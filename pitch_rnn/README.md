# Pitch RNN

A recurrent neural network that predicts the next pitch type thrown within a plate appearance, given the sequence of pitches thrown so far.

## Overview

The model is a GRU-based RNN that reads a plate appearance pitch-by-pitch and outputs a probability distribution over pitch types at each step. Training is orchestrated by `trainer_engine.py`, which calls into this package.

There are four model variants, each with a different prediction target:

| Model Type | Target | Description |
|---|---|---|
| `group` | Pitch family (fastball / breaking / offspeed) | First-level classification |
| `fastball` | Specific fastball type (FF, SI, FC) | Second-level within fastballs |
| `breaking` | Specific breaking ball type (SL, CU, KC, SV, ST) | Second-level within breaking balls |
| `offspeed` | Specific offspeed type (CH, FS) | Second-level within offspeed |

---

## Pipeline

### 1. Data Loading (`trainer_engine.py`)

Training data is loaded from a Parquet file or database via `get_training_data()`, which pulls from the `historical_pitches` table. A feature list file (passed as a CLI argument) specifies which columns to pull.

### 2. Preprocessing (`model_shared/feature_engineering/`)

`clean_data()` filters out non-pitch events and pitch codes in `IGNORE` (`ABS`, `PO`, `FA`, `EP`). `get_rnn_features()` adds engineered features from the feature repository.

### 3. Target Variable Calculation (`pitch_rnn_trainer.py` → `calculate_target_variable`)

For each pitch in a plate appearance, the target is the **next** pitch thrown. Rows where the following pitch is not a real pitch are dropped. The resulting columns are:

- `y_next_pitch_group` — pitch family of the next pitch
- `y_next_pitch_fastball/breaking/offspeed` — specific type within each family

### 4. Train/Test Split (`split_by_year`)

Data is split temporally — years 2023–2024 are used for training and 2025 is held out as the test set. This prevents leakage from future games into model training.

### 5. Feature Engineering (`pitch_rnn_trainer.py`)

Aggregate features are computed on the training set and then joined onto the test set via lookup to prevent leakage:

- `add_pitcher_count_split_features` — pitcher pitch mix rates by count situation
- `add_batter_count_split_features` — batter swing/whiff rates by count situation
- `add_pitcher_family_rate_features` — pitcher overall family-level pitch rates
- `calculate_game_state_features` — base state, inning, score differential, etc.

### 6. Encoding (`pitch_rnn/encoder.py`)

**Categorical columns** (pitcher, batter, pitch type, count state, etc.) are integer-encoded via `build_vocab`, which assigns a 1-based index to each unique value seen in the training set. Unknown values at inference time are mapped to `PAD_ID = 0`.

**Numerical columns** are standardized using `StandardScaler` fit on the training set only.

### 7. Sequence Construction (`pitch_rnn_trainer.py` → `make_fixed_sequences`)

Each plate appearance becomes a fixed-length tensor of shape `(max_len, n_features)` where `MAX_LEN = 8`. Plate appearances with fewer than 8 pitches are zero-padded. The three tensors produced are:

- `X_cat` — categorical feature IDs, shape `(n_pa, 8, n_cat_cols)`
- `X_num` — normalized numerical features, shape `(n_pa, 8, n_num_cols)`
- `Y` — target pitch ID at each step, shape `(n_pa, 8)`

### 8. Class Weighting (`calculate_class_weights`)

Pitch types are imbalanced (fastballs dominate). Inverse-frequency class weights with a smoothing exponent are computed and passed to the loss function to prevent the model from collapsing to a majority-class predictor.

### 9. Model Architecture (`model_shared/rnn_definition.py` → `PitchRNN`)

```
Categorical inputs → Embedding layers (per column) → Dropout
                                                        ↓
Numerical inputs ──────────────────────────────────→ Concat
                                                        ↓
                                                    GRU (2 layers, hidden=128)
                                                        ↓
                                                    Dropout → Linear → logits
```

Each categorical column gets its own `nn.Embedding` with a tunable dimension (e.g., pitcher/batter = 32 dims, count state = 8 dims). These are concatenated with the scaled numerical features and fed into the GRU at each time step. The GRU outputs a hidden state at every position, and a linear head converts it to per-class logits.

Default hyperparameters:

| Parameter | Value |
|---|---|
| Hidden size | 128 |
| GRU layers | 2 |
| Dropout | 0.35 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Max epochs | 20 |

### 10. Loss Function (`FocalLoss`)

[Focal Loss](https://arxiv.org/abs/1708.02002) is used instead of standard cross-entropy. It down-weights easy/common examples (fastballs) by a factor of `(1 - p)^γ`, concentrating training signal on pitches the model finds hard to predict. Class weights are also applied, and `PAD_ID = 0` positions are masked out of the loss.

### 11. Training Loop (`train_model`)

Each epoch:
1. Forward pass → compute Focal Loss over all non-pad positions
2. Backprop with gradient clipping (`max_norm=1.0`)
3. Adam optimizer step
4. Evaluate on the held-out test set
5. `ReduceLROnPlateau` halves the learning rate if test loss stagnates for 2 epochs
6. `EarlyStopping` (patience=3) saves the best model state and halts training if no improvement

### 12. Export (`pitch_rnn/export_artifacts.py`)

After training, three artifacts are written to `model_shared/`:

| Artifact | Path | Contents |
|---|---|---|
| Model weights | `trained-parameters/pitch_rnn_{type}_{date}.pt` | `model.state_dict()` |
| Vocabularies | `vocab/rnn_vocab_{type}_{date}.json` | Cat vocabs, y vocab, feature spec |
| Test tensors | `test_data/test_tensors_{type}_{date}.pt` | `Xc`, `Xn`, `Y` tensors for evaluation |

A `temperature.json` file is also written (default 1.0), which the inference engine uses to scale logits at prediction time.

---

## Running Training

```bash
python trainer_engine.py <feature_list_file>
```

The `MODEL_TYPE` constant at the top of `trainer_engine.py` selects which variant to train (`group`, `fastball`, `breaking`, or `offspeed`).

---

## Module Reference

| File | Responsibility |
|---|---|
| `pitch_rnn_trainer.py` | Orchestrates the full training pipeline |
| `encoder.py` | Vocabulary building and integer encoding |
| `sequence_builder.py` | `PitchSeqDS` PyTorch Dataset wrapper |
| `early_stopping.py` | Tracks best val loss and saves best model state |
| `export_artifacts.py` | Saves model, vocabs, and test tensors to disk |
| `setup_data.py` | CLI script to pull raw features from the DB into CSV |
