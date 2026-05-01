# Pitch RNN Training Diagnostics

## Is rising test loss after 3 epochs a problem?

Not necessarily — it's too early to panic. Early in training the model is rapidly adjusting weights and test loss can lag or temporarily spike before falling. With `patience=5` the early-stopping safety net has not yet engaged. That said, 3 epochs of a rising trend is worth investigating because the underlying causes listed below could prevent test loss from ever recovering.

---

## Weak Points

### 1. New 2025 players get wrong statistical baseline (data bug)

**File:** `pitch_rnn_trainer.py` → `apply_pitcher_lookup` / `apply_batter_lookup`, then `encoder.py` → `encode_df`

When a 2025 pitcher or batter was not in training, the left-join leaves `pitcher_sit_*` and `batter_sit_*` as `NaN`. `encode_df` fills those with raw `0` before calling `scaler.transform`. The scaler was fit on training data whose natural zero is not the mean. After scaling, raw 0 maps to *below* the mean, not *at* the mean. A new pitcher appears to the model as having zero fastball rate, zero whiff rate, etc., rather than an average pitcher.

**Recommendation:** Fill NaN with each column's training mean before scaling, not with literal 0.

```python
train_means = train_df[NUM_COLS].mean()
test_df[NUM_COLS] = test_df[NUM_COLS].fillna(train_means)
```

Pass `train_means` into `encode_df` (or compute it from the scaler's `mean_` attribute after fitting) and use it instead of `fillna(0)`.

---

### 2. Arsenal mask uses year=2026 for all training data

**File:** `pitch_rnn_trainer.py:267`

```python
arsenal_masks = build_arsenal_masks(arsenals, cat_vocabs, y_vocab, num_classes, year=2026)
```

The training set spans 2021–2024. Applying a 2026 arsenal mask to those observations is a temporal mismatch — a pitcher's declared arsenal in 2026 may not reflect what they threw in 2022. This can mask pitches that were actually thrown in training, causing the `apply_arsenal_mask_train` true-label override to fire frequently, which weakens the arsenal constraint the model is supposed to learn.

**Recommendation:** Either build per-year masks and select the mask that matches each sequence's game year, or limit the mask to pitchers whose arsenal is well-established and consistent. At minimum, consider using `year=2025` so the mask is contemporary with the most recent training data.

---

### 3. Pitcher/batter count-split stats include the target pitch (mild leakage)

**File:** `feature_calculator.py` → `add_pitcher_count_split_features`, `add_batter_count_split_features`

The aggregate stats (e.g., `pitcher_sit_fb_rate`) are computed from all rows in `train_df`, including the row being predicted. Because the target is `y_next_pitch_type` (shifted), the current row's `pitch_type` is part of the aggregate used as a feature on the same row. For large datasets this is a small bias, but it means the model sees slightly inflated accuracy on its own training statistics.

**Recommendation:** Compute the lookup table from a *leave-one-out* or rolling prior window to remove each row's own contribution. The simplest fix is computing the lookup from the previous season's data only.

---

### 4. Year-based split creates real distribution shift

**File:** `pitch_rnn_trainer.py` → `split_by_year`

Training on 2021–2024 and testing on 2025 is good practice for avoiding temporal leakage, but it means the model is genuinely generalizing across seasons. Roster turnover, pitch mix evolution (e.g., sweeper adoption), and rule changes (shift ban, pitch clock) all make 2025 systematically different from prior seasons. A gap between train and test loss is expected and not necessarily fixable through architecture changes alone.

**Recommendation:** Monitor whether test loss is converging (even if higher than train loss) or diverging. If diverging, evaluate adding 2024 data weight or recency-weighting samples so recent seasons matter more during training.

---

### 5. No learning rate scheduler

**File:** `trainer_engine.py` → `MODEL_HYPERPARAMETERS`

Adam with a fixed `lr=0.001` can overshoot flat minima as training progresses. A scheduler would reduce the LR when test loss plateaus, giving finer convergence.

**Recommendation:** Add `ReduceLROnPlateau` tied to test loss:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
# inside train_model, after computing test_loss each epoch:
scheduler.step(test_loss)
```

---

### 6. Loss metric is PA-count normalized, not pitch-count normalized

**File:** `pitch_rnn_trainer.py` → `train_model` (lines 201, 219)

```python
train_loss += loss.item() * x_cat.size(0)   # scales by batch size in PAs
train_loss /= len(train_loader.dataset)      # divides by total PAs
```

`CrossEntropyLoss` with `ignore_index` already returns the mean loss per *valid token* (pitch). Multiplying by batch size then dividing by dataset size gives a PA-weighted average of per-pitch-averages. If the training and test sets have different average PA lengths, the two loss numbers are not on the same scale, making the comparison misleading.

**Recommendation:** Track total valid tokens alongside loss:

```python
n_valid = (y != PAD_ID).sum().item()
train_loss += loss.item() * n_valid
train_tokens += n_valid
# ...
train_loss /= train_tokens
```

---

### 7. Cold-start for unseen 2025 pitchers and batters in embeddings

**File:** `encoder.py` → `encode`

Unknown 2025 players map to `PAD_ID = 0` via `fillna(PAD_ID)`. The embedding at index 0 is the padding embedding, which is frozen at zero by `padding_idx=pad_id` in `nn.Embedding`. So unseen pitchers and batters contribute zero to the embedding input, which means the model falls back entirely on numerical features for those players.

**Recommendation:** Assign unknown entities their own dedicated `<UNK>` embedding index (separate from PAD). This lets the model learn a meaningful average representation for unseen players rather than silencing their signal entirely.

```python
# In build_vocab, reserve index 1 for UNK, push real values to start=2
UNK_ID = 1

def encode(series, vocab):
    return series.map(vocab).fillna(UNK_ID).astype(int)
```

---

## Summary Table

| # | Issue | Severity | Fix Complexity |
|---|-------|----------|----------------|
| 1 | New player stats filled with raw 0 instead of training mean | High | Low |
| 2 | Arsenal mask year=2026 applied to 2021-2024 training data | Medium | Low |
| 3 | Count-split stats include target pitch (leakage) | Medium | Medium |
| 4 | Year-based split distribution shift | Expected | N/A |
| 5 | No LR scheduler | Medium | Low |
| 6 | Loss metric normalized by PA count not pitch count | Low | Low |
| 7 | Unseen 2025 players silenced via PAD embedding | Medium | Low |
