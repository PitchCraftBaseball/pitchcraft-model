import pandas as pd
import numpy as np
from model_shared.feature_engineering.pitch_constants import PAD_ID
from sklearn.preprocessing import StandardScaler

def build_vocab(values):
    uniq = pd.Series(values.dropna().unique())
    return {v: i for i, v in enumerate(uniq, start=1)}

def encode(series, vocab):
    return series.map(vocab).fillna(PAD_ID).astype(int)

def encode_df(df, feature_spec, cat_vocabs, y_vocab, scaler):
    TARGET_COL = feature_spec["target"]
    CAT_COLS = feature_spec["cat_cols"]
    NUM_COLS = feature_spec["num_cols"]

    out = df.copy()
    for c in CAT_COLS:
        out[c + "_id"] = encode(out[c], cat_vocabs[c])
    out["y_id"] = encode(out[TARGET_COL], y_vocab)

    out["y_horiz_id"] = out["y_next_horiz"].fillna(PAD_ID).astype(np.int64)
    out["y_vert_id"]  = out["y_next_vert"].fillna(PAD_ID).astype(np.int64)

    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.float32)
    out[NUM_COLS] = scaler.transform(out[NUM_COLS])
    return out


    

