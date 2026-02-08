# %%
# %pip install --upgrade --quiet torch pandas numpy matplotlib scikit-learn seaborn

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import seaborn as sns


# %% [markdown]
# ### Data Preparation

# %%
data = pd.read_csv('rnn_data.csv')

# %%
data.head()

# %%
data["is_real_pitch"] =  data["pitch_type"].notna() & (data["pitch_type"] != "ABS")

data["target_is_real_pitch"] = data.groupby("pa_id")["is_real_pitch"].shift(-1)

data["y_next_pitch_type"] = data.groupby("pa_id")["pitch_type"].shift(-1)

data_train = data[data["target_is_real_pitch"] == True].copy() 

# %%
# randomly select plate appearances to be a part of training and test sets

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

# %%
train_df, test_df, train_ids, test_ids = split_by_pa_id(
    data_train, pa_col="pa_id", ratios=(0.8, 0.2), seed=7
)

# %%
# Features to be used by the model
FEATURE_SPEC = {
    "target": "y_next_pitch_type",
    "cat_cols": [
        "pitcher", "batter", "stand", "p_throws", "inning_topbot",
        "count_state", "prev_pitch_type"
    ],
    "num_cols": [
        "balls", "strikes", "outs_when_up", "inning", "score_diff_bat",
        "on_1b", "on_2b", "on_3b"
    ],
}

TARGET_COL = FEATURE_SPEC["target"]
CAT_COLS = FEATURE_SPEC["cat_cols"]
NUM_COLS = FEATURE_SPEC["num_cols"]

# %% [markdown]
# ### RNN Setup

# %%
PAD_ID = 0

def build_vocab(values):
    uniq = pd.Series(values.dropna().unique())
    return {v: i for i, v in enumerate(uniq, start=1)}

def encode(series, vocab):
    return series.map(vocab).fillna(PAD_ID).astype(int)

cat_vocabs = {c: build_vocab(train_df[c]) for c in CAT_COLS}
y_vocab    = build_vocab(train_df[TARGET_COL])

def encode_df(df):
    out = df.copy()
    for c in CAT_COLS:
        out[c + "_id"] = encode(out[c], cat_vocabs[c])
    out["y_id"] = encode(out[TARGET_COL], y_vocab)
    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.float32)
    return out

train_enc = encode_df(train_df)
test_enc  = encode_df(test_df)

# %%
# make all pitch sequences the same length
def make_fixed_sequences(df, pa_col="pa_id", max_len=8):
    X_cat, X_num, Y = [], [], []

    for _, g in df.groupby(pa_col, sort=False):

        cat = g[[c + "_id" for c in CAT_COLS]].to_numpy(np.int64)     
        num = g[NUM_COLS].to_numpy(np.float32)                        
        y   = g["y_id"].to_numpy(np.int64)                            

        L = min(len(g), max_len)
        cat, num, y = cat[:L], num[:L], y[:L]

        pad = max_len - L
        if pad > 0:
            cat = np.pad(cat, ((0,pad),(0,0)), constant_values=PAD_ID)
            num = np.pad(num, ((0,pad),(0,0)), constant_values=0.0)
            y   = np.pad(y,   (0,pad),         constant_values=PAD_ID)

        X_cat.append(cat); X_num.append(num); Y.append(y)

    return (
        torch.tensor(np.stack(X_cat), dtype=torch.long),
        torch.tensor(np.stack(X_num), dtype=torch.float32),
        torch.tensor(np.stack(Y),     dtype=torch.long),
    )

MAX_LEN = 8
Xc_tr, Xn_tr, Y_tr = make_fixed_sequences(train_enc, max_len=MAX_LEN)
Xc_te, Xn_te, Y_te = make_fixed_sequences(test_enc,  max_len=MAX_LEN)

# %%
# Create the Dataset
class PitchSeqDS(Dataset):
    def __init__(self, Xc, Xn, Y):
        self.Xc, self.Xn, self.Y = Xc, Xn, Y
    def __len__(self): return self.Y.size(0)
    def __getitem__(self, i): return self.Xc[i], self.Xn[i], self.Y[i]

train_loader = DataLoader(PitchSeqDS(Xc_tr, Xn_tr, Y_tr), batch_size=64, shuffle=True)
test_loader  = DataLoader(PitchSeqDS(Xc_te, Xn_te, Y_te), batch_size=64, shuffle=False)

# %%
# Model Creation
class SimplePitchRNN(nn.Module):
    def __init__(self, cat_vocab_sizes, num_features, emb_dim=16, hidden=128, num_classes=16, pad_id=0):
        super().__init__()
        self.cat_cols = list(cat_vocab_sizes.keys())

        self.embs = nn.ModuleDict({
            col: nn.Embedding(cat_vocab_sizes[col], emb_dim, padding_idx=pad_id)
            for col in self.cat_cols
        })
        
        in_dim = len(self.cat_cols) * emb_dim + num_features
        self.rnn = nn.RNN(in_dim, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, num_classes)

    def forward(self, x_cat, x_num):
        embs = []
        for j, col in enumerate(self.cat_cols):
            embs.append(self.embs[col](x_cat[:, :, j]))  
        x = torch.cat(embs + [x_num], dim=-1)           
        h, _ = self.rnn(x)                               
        return self.fc(h)                                

cat_vocab_sizes = {c: len(cat_vocabs[c]) + 1 for c in CAT_COLS}  
num_classes = len(y_vocab) + 1                                   

# Model Initialization
model = SimplePitchRNN(cat_vocab_sizes, num_features=len(NUM_COLS), num_classes=num_classes)

# %%
from datetime import datetime
from pathlib import Path
import pandas as pd

# Export vocabularies and feature lists with a date-stamped filename.
today = datetime.now().strftime("%Y%m%d")
vocab_dir = Path.cwd().parent / "vocab"
feature_dir = Path.cwd().parent / "feature-list"
vocab_dir.mkdir(parents=True, exist_ok=True)
feature_dir.mkdir(parents=True, exist_ok=True)

rows = []
for feature, vocab in cat_vocabs.items():
    for value, idx in vocab.items():
        rows.append({"feature": feature, "value": value, "id": idx, "kind": "categorical"})
for value, idx in y_vocab.items():
    rows.append({"feature": TARGET_COL, "value": value, "id": idx, "kind": "target"})

vocab_path = vocab_dir / f"rnn_vocab_{today}.csv"
pd.DataFrame(rows).to_csv(vocab_path, index=False)

feature_rows = []
for c in CAT_COLS:
    feature_rows.append({"feature": c, "kind": "categorical"})
for c in NUM_COLS:
    feature_rows.append({"feature": c, "kind": "numerical"})
feature_rows.append({"feature": TARGET_COL, "kind": "target"})

feature_path = feature_dir / f"rnn_vocab_{today}.csv"
pd.DataFrame(feature_rows).to_csv(feature_path, index=False)

print(f"Wrote vocab to {vocab_path}")
print(f"Wrote feature list to {feature_path}")


# %% [markdown]
# ### Training and Evaluation

# %%
# Simplified the training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for x_cat, x_num, y in train_loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        y     = y.to(device)
        logits = model(x_cat, x_num)  

        loss = criterion(
            logits.reshape(-1, num_classes),
            y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

# %%
PAD_ID = 0

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x_cat, x_num, y in test_loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        y     = y.to(device)

        outputs = model(x_cat, x_num)      
        predicted = outputs.argmax(dim=-1) 

        mask = (y != PAD_ID)
        correct += ((predicted == y) & mask).sum().item()
        total += mask.sum().item()

accuracy = 100 * correct / total
print(f"Token Accuracy (no PAD): {accuracy:.2f}%")


# %%
from collections import Counter

model.eval()
counts = Counter()

with torch.no_grad():
    for x_cat, x_num, y in test_loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)

        preds = model(x_cat, x_num).argmax(dim=-1)
        for p in preds.view(-1).tolist():
            if p != 0:
                counts[p] += 1

print(counts.most_common(5))


# %%
# reverse vocab
id_to_pitch = {v: k for k, v in y_vocab.items()}

for pid, cnt in counts.most_common(5):
    print(id_to_pitch.get(pid, "PAD"), cnt)


# %%
model.eval()
correct = 0
total = 0
K = 3

with torch.no_grad():
    for x_cat, x_num, y in test_loader:
        x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
        logits = model(x_cat, x_num)
        topk = logits.topk(K, dim=-1).indices 

        mask = (y != PAD_ID)
        match = (topk == y.unsqueeze(-1)).any(dim=-1)

        correct += (match & mask).sum().item()
        total += mask.sum().item()

print(f"Top-{K} Accuracy: {100*correct/total:.2f}%")


# %%
# Confusion Matrix
from sklearn.metrics import confusion_matrix

model.eval()
all_preds = [] 
all_true = []

with torch.no_grad():
    for x_cat, x_num, y in test_loader:
        logits = model(x_cat, x_num)        # (B, L, C)
        preds  = logits.argmax(dim=-1)      # (B, L)

        mask = (y != PAD_ID)

        # mask == 1 for valid tokens, 0 for PAD
        valid = mask.bool()

        all_preds.append(preds[valid].cpu().numpy())
        all_true.append(y[valid].cpu().numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_true)


cm = confusion_matrix(y_true, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_norm,
    cmap="Blues",
    fmt="d"
)
plt.xlabel("Predicted Pitch Type")
plt.ylabel("True Pitch Type")
plt.title("Pitch Type Confusion Matrix (Token-Level)")
plt.show()


# %%
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=list(id_to_pitch.values()))
print(report)


