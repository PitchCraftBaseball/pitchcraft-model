import numpy as np 
import pandas as pd
import torch
import json
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from model_shared.feature_engineering.pitch_constants import *
from model_shared.feature_engineering.feature_calculator import pitch_to_family, add_batter_count_split_features, add_pitcher_count_split_features, calculate_game_state_features
from sklearn.preprocessing import StandardScaler
from pitch_rnn.encoder import build_vocab, encode_df
from pitch_rnn.sequence_builder import PitchSeqDS
from torch.utils.data import DataLoader
from model_shared.rnn_definition import PitchRNN
from pitch_rnn.early_stopping import EarlyStopping
from pitch_rnn.export_artifacts import *
from model_shared.feature_engineering.location import *


def calculate_target_variable(data: pd.DataFrame) -> pd.DataFrame:
    data["is_real_pitch"] = data["pitch_type"].notna() & (~data["pitch_type"].isin(IGNORE))
    data["target_is_real_pitch"] = data.groupby("pa_id")["is_real_pitch"].shift(-1)
    data["y_next_pitch_type"]  = data.groupby("pa_id")["pitch_type"].shift(-1)
    data["y_next_pitch_group"] = data["y_next_pitch_type"].map(lambda x: pitch_to_family(x))

    # new — next pitch location targets
    data["y_next_horiz"] = data.groupby("pa_id")["horiz_bucket"].shift(-1)
    data["y_next_vert"]  = data.groupby("pa_id")["vert_bucket"].shift(-1)

    data_train = data[data["target_is_real_pitch"] == True].copy()
    data_train = data_train[data_train["y_next_pitch_group"].notna()].copy()

    return data_train

# randomly select plate appearances to be a part of training and test sets

def apply_pitcher_lookup(train_df, test_df):
    pitcher_cols = ["pitcher", "count_situation", "pitcher_sit_n",
                    "pitcher_sit_fb_rate", "pitcher_sit_br_rate",
                    "pitcher_sit_os_rate", "pitcher_sit_whiff_rate"]
    lookup = train_df[pitcher_cols].drop_duplicates(subset=["pitcher", "count_situation"])
    return test_df.merge(lookup, on=["pitcher", "count_situation"], how="left")

def apply_batter_lookup(train_df, test_df):
    batter_cols = ["batter", "count_situation", "batter_sit_n",
                   "batter_sit_swing_rate", "batter_sit_whiff_rate"]
    lookup = train_df[batter_cols].drop_duplicates(subset=["batter", "count_situation"])
    return test_df.merge(lookup, on=["batter", "count_situation"], how="left")

def split_by_pa_id(df: pd.DataFrame, pa_col="pa_id", ratios=(0.8, 0.2), seed: int=42):
    r_train, r_test = ratios
    assert abs((r_train + r_test) - 1.0) < 1e-9

    pa_ids = df[pa_col].dropna().unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(pa_ids)

    n = len(pa_ids)
    n_train = int(n*r_train)

    train_ids = set(pa_ids[:n_train])
    test_ids = set(pa_ids[n_train:])

    train_df = df[df[pa_col].isin(train_ids)].copy()
    test_df = df[df[pa_col].isin(test_ids)].copy()

    return train_df, test_df, train_ids, test_ids

def split_by_year(df: pd.DataFrame, test_year: int = 2025) -> tuple:
    train_df = df[df["game_year"] < test_year].copy()
    test_df  = df[df["game_year"] == test_year].copy()

    print(f"Train: {len(train_df):,} rows ({df['game_year'].min()}-{test_year - 1})")
    print(f"Test:  {len(test_df):,} rows ({test_year})")

    return train_df, test_df

def make_fixed_sequences(df, feature_spec, pa_col="pa_id", max_len=8):
    CAT_COLS    = feature_spec["cat_cols"]
    NUM_COLS    = feature_spec["num_cols"]
    cat_id_cols = [c + "_id" for c in CAT_COLS]

    df = df.copy()
    df["_pitch_pos"] = df.groupby(pa_col).cumcount()
    df = df[df["_pitch_pos"] < max_len]

    pa_ids        = df[pa_col].unique()
    df["_pa_idx"] = df[pa_col].map({pa: i for i, pa in enumerate(pa_ids)})

    n_pa  = len(pa_ids)
    n_cat = len(cat_id_cols)
    n_num = len(NUM_COLS)

    X_cat   = np.full((n_pa, max_len, n_cat), PAD_ID, dtype=np.int64)
    X_num   = np.zeros((n_pa, max_len, n_num),         dtype=np.float32)
    Y_pitch = np.full((n_pa, max_len),         PAD_ID, dtype=np.int64)
    Y_horiz = np.full((n_pa, max_len),         PAD_ID, dtype=np.int64)
    Y_vert  = np.full((n_pa, max_len),         PAD_ID, dtype=np.int64)

    pa_idx    = df["_pa_idx"].to_numpy(np.int64)
    pitch_pos = df["_pitch_pos"].to_numpy(np.int64)

    X_cat[pa_idx, pitch_pos]   = df[cat_id_cols].to_numpy(np.int64)
    X_num[pa_idx, pitch_pos]   = df[NUM_COLS].to_numpy(np.float32)
    Y_pitch[pa_idx, pitch_pos] = df["y_id"].to_numpy(np.int64)
    Y_horiz[pa_idx, pitch_pos] = df["y_horiz_id"].to_numpy(np.int64)
    Y_vert[pa_idx, pitch_pos]  = df["y_vert_id"].to_numpy(np.int64)

    return (
        torch.from_numpy(X_cat),
        torch.from_numpy(X_num),
        torch.from_numpy(Y_pitch),
        torch.from_numpy(Y_horiz),
        torch.from_numpy(Y_vert),
    )

def calculate_class_weights(y_train, num_classes, pad_id=0, smoothing=0.35):
    y_flat = y_train.flatten()
    y_flat = y_flat[y_flat != pad_id]

    unique, counts = np.unique(y_flat, return_counts=True)

    total = len(y_flat)
    frequencies = counts / total
    weights = (1.0 / frequencies) ** smoothing

    weight_tensor = torch.zeros(num_classes)
    for class_id, weight in zip (unique, weights):
        weight_tensor[class_id] = weight
    return weight_tensor

def build_arsenal_masks(arsenals, cat_vocabs, y_vocab, num_classes, year):
    pitcher_vocab = cat_vocabs["pitcher"]
        
    masks = torch.ones(len(pitcher_vocab) + 1, num_classes)

    for pitcher_str, data in arsenals.items():
        pitcher_id_int = int(pitcher_str)

        if pitcher_id_int not in pitcher_vocab:
            continue

        enc_id = pitcher_vocab[pitcher_id_int]

        if year in data:
            allowed_pitches = data[year]["arsenal_mask"]
        elif str(int(year) - 1) in data:
            allowed_pitches = data[str(int(year) - 1)]["arsenal_mask"]
        else:
            continue

        masks[enc_id] = 0
        for pitch in allowed_pitches:
            if pitch in y_vocab:
                masks[enc_id, y_vocab[pitch]] = 1

    return masks

def train_model(
    model, train_loader, test_loader,
    criterion, optimizer, early_stopping,
    device, num_classes,
    horiz_weight: float = 0.3,
    vert_weight:  float = 0.3,
):
    criterion_loc = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    epochs = 20

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x_cat, x_num, y, y_horiz, y_vert in train_loader:
            x_cat   = x_cat.to(device)
            x_num   = x_num.to(device)
            y       = y.to(device)
            y_horiz = y_horiz.to(device)
            y_vert  = y_vert.to(device)

            logits_pitch, logits_horiz, logits_vert = model(x_cat, x_num)

            loss_pitch = criterion(
                logits_pitch.reshape(-1, num_classes), y.reshape(-1)
            )
            loss_horiz = criterion_loc(
                logits_horiz.reshape(-1, 3), y_horiz.reshape(-1)
            )
            loss_vert = criterion_loc(
                logits_vert.reshape(-1, 3), y_vert.reshape(-1)
            )

            loss = loss_pitch + horiz_weight * loss_horiz + vert_weight * loss_vert

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss_pitch.item() * x_cat.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_cat, x_num, y, y_horiz, y_vert in test_loader:
                x_cat = x_cat.to(device)
                x_num = x_num.to(device)
                y = y.to(device)
                y_horiz = y_horiz.to(device)
                y_vert  = y_vert.to(device)

                logits_pitch, logits_horiz, logits_vert = model(x_cat, x_num)

                loss = criterion(
                    logits_pitch.reshape(-1, num_classes), y.reshape(-1)
                )
                test_loss += loss.item() * x_cat.size(0)

        test_loss /= len(test_loader.dataset)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

    early_stopping.load_best_model(model)

def rnn_training_handler(data: pd.DataFrame, feature_spec, custom_emb_dims, model_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TARGET_COL = feature_spec["target"]
    CAT_COLS = feature_spec["cat_cols"]
    NUM_COLS = feature_spec["num_cols"]

    boundaries = compute_bucket_boundaries(data)
    data       = add_location_targets(data, boundaries)
    data       = add_prev_location_features(data)

    data_train = calculate_target_variable(data)

    train_df, test_df = split_by_year(data_train, test_year=2025)
    print("Split Training Data")

    train_df = calculate_game_state_features(train_df)
    test_df  = calculate_game_state_features(test_df)

    train_df = add_pitcher_count_split_features(train_df)
    train_df = add_batter_count_split_features(train_df)

    test_df = apply_pitcher_lookup(train_df, test_df)
    test_df = apply_batter_lookup(train_df, test_df)

    cat_vocabs = {c: build_vocab(train_df[c]) for c in CAT_COLS}
    y_vocab = build_vocab(train_df[TARGET_COL])
    cat_vocab_sizes = {c: len(cat_vocabs[c]) + 1 for c in CAT_COLS}
    num_classes = len(y_vocab) + 1
    scaler = StandardScaler()
    scaler.fit(train_df[NUM_COLS].fillna(0))

    train_enc = encode_df(train_df, feature_spec, cat_vocabs, y_vocab, scaler)
    test_enc = encode_df(test_df, feature_spec, cat_vocabs, y_vocab, scaler)
    print("Encoded Data")

    Xc_tr, Xn_tr, Y_tr, Yh_tr, Yv_tr = make_fixed_sequences(
        train_enc, feature_spec, max_len=MAX_LEN
    )
    Xc_te, Xn_te, Y_te, Yh_te, Yv_te = make_fixed_sequences(
        test_enc, feature_spec, max_len=MAX_LEN
    )
    print("Made Fixed Sequences")

    train_loader = DataLoader(
        PitchSeqDS(Xc_tr, Xn_tr, Y_tr, Yh_tr, Yv_tr),
        batch_size=model_params['batch_size'], shuffle=True
    )
    test_loader = DataLoader(
        PitchSeqDS(Xc_te, Xn_te, Y_te, Yh_te, Yv_te),
        batch_size=model_params['batch_size'], shuffle=False
    )
    print("Loaded Data")

    class_weights = calculate_class_weights(Y_tr, num_classes, PAD_ID, model_params['smoothing_weights'])

    model = PitchRNN(
        cat_vocab_sizes = cat_vocab_sizes,
        num_features    = len(NUM_COLS),
        emb_dims        = custom_emb_dims,
        num_classes     = num_classes,
        num_layers      = model_params['model_layers']
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=model_params['optimizer_lr'])
    early_stopping = EarlyStopping(patience=model_params['stopping_patience'], delta=model_params['stopping_delta'])

    train_model(model, train_loader, test_loader, criterion, optimizer, early_stopping, device, num_classes)

    export_model(model)
    export_vocabs(cat_vocabs, y_vocab, feature_spec)
    export_test_tensors(Xc_te, Xn_te, Y_te, Yh_te, Yv_te)