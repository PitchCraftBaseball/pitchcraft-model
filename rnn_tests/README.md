# RNN Test Suite

Unit tests for the PitchCraft RNN model pipeline, covering data preprocessing, feature
engineering, encoding, sequence building, model construction, training utilities, and
artifact export.

---

## Running the Tests

```bash
pytest rnn_tests/
```

To see coverage:

```bash
pytest rnn_tests/ --cov=pitch_rnn --cov=model_shared --cov-report=term-missing
```

---

## Test Modules

### `conftest.py` — Shared Fixtures

Defines fixtures and helper factory functions shared across all test modules:

- **`feature_spec`** — a minimal 2-feature-spec dict (`pitcher`, `stand`, `prev_pitch_type` as cat cols; `outs_when_up`, `inning` as num cols)
- **`sample_cat_vocabs` / `sample_y_vocab`** — small hand-crafted vocabularies used to keep tests fast and deterministic
- **`fitted_scaler`** — a `StandardScaler` pre-fit on 3-row training data, ready to pass into `encode_df`
- **`make_sort_df` / `make_universal_df` / `make_target_df` / `make_encoded_df`** — factories that build the minimal DataFrames required by each layer of the pipeline

---

### `test_model.py` — Model Architecture (`PitchRNN`)

Verifies that the model is constructed correctly and behaves as expected during both
forward and backward passes, without running any training.

| Test | What it checks |
|---|---|
| `test_model_output_shape` | Forward pass returns `(batch, seq_len, num_classes)` |
| `test_model_default_emb_dims` | When `emb_dims=None`, every embedding defaults to dim 16 |
| `test_model_padding_idx_is_zero` | Every embedding uses `padding_idx=0` (the PAD token) |
| `test_model_num_layers` | `num_layers` is forwarded correctly to the GRU |
| `test_model_fc_out_dim` | The final linear layer outputs `num_classes` logits |
| `test_model_eval_deterministic` | Identical inputs produce identical outputs in eval mode |
| `test_model_pad_token_gradient_is_zero` | The PAD embedding row receives no gradient during backprop |

---

### `test_encoding.py` — Vocabulary and Encoding (`encoder.py`)

Covers `build_vocab`, `encode`, and `encode_df` — the functions that turn raw string
and numeric columns into integer and float tensors.

| Test | What it checks |
|---|---|
| `test_build_vocab_ids_start_at_1` | IDs are always ≥ 1; 0 is reserved for PAD |
| `test_build_vocab_excludes_NaN` | NaN values are never assigned a vocab entry |
| `test_build_vocab_unique_ids` | Duplicate input values collapse to one entry with one ID |
| `test_encode_known_values` | Mapped values match the vocab and produce int dtype |
| `test_encode_OOV_maps_to_PAD` | Out-of-vocabulary values encode to `PAD_ID` (0) |
| `test_encode_NaN_maps_to_PAD` | NaN in the input series encodes to `PAD_ID` |
| `test_encode_df_cat_id_columns_exist` | `encode_df` produces a `{col}_id` column for every cat col |
| `test_encode_df_y_id_encodes_target` | Unknown target values map to `PAD_ID` |
| `test_encode_df_num_cols_float32_no_NaN` | All numeric columns become `float32` with no NaN remaining |
| `test_encode_df_scaler_not_refit_on_test` | The scaler is fit on train data only; test data is transformed, not refit |

---

### `test_data_prep.py` — Data Cleaning and Preprocessing

Covers `sort_statcast`, `universal_features`, `data_remapping`, `drop_unused_cols`,
`clean_data`, `calculate_target_variable`, and `split_by_pa_id`.

| Test | What it checks |
|---|---|
| `test_sort_statcast_ascending_order` | Rows sort game_date → inning → Top before Bot → at-bat → pitch number |
| `test_universal_features_game_type_filter` | Only regular season (`game_type == 'R'`) rows survive |
| `test_universal_features_drops_UN_pitch_type` | `UN` pitch types are removed |
| `test_universal_features_pa_id_construction` | `pa_id` is formatted as `'{game_pk}_{at_bat_number}'` |
| `test_universal_features_prev_pitch_type_START_for_first_pitch` | First pitch of every PA gets `prev_pitch_type == 'START'` |
| `test_universal_features_prev_pitch_skips_ABS` | Automatic ball/strike pitches are skipped when computing `prev_pitch_type` |
| `test_universal_features_prev_pitch_stays_within_PA` | `prev_pitch_type` never bleeds across plate appearances |
| `test_universal_features_seq_len` | Every row in a PA carries the correct total sequence length |
| `test_calculate_target_*` | Target shift is correct; IGNORE types are excluded; no cross-PA bleed |
| `test_split_by_pa_id_*` | Ratio validation, disjoint sets, and seed reproducibility |
| `test_data_remapping_*` | `SC/CS → CU`, `FO → FS`, `automatic_ball/strike → ABS` |
| `test_drop_unused_cols_*` | Drop-list columns removed; absent columns handled without error |
| `test_clean_data_*` | End-to-end: remapping, column drops, game type filter, `pa_id` creation |

---

### `test_sequences.py` — Sequence Building (`make_fixed_sequences`, `PitchSeqDS`)

Verifies the tensor construction step that pads or truncates each plate appearance
into a fixed-length sequence before it enters the DataLoader.

| Test | What it checks |
|---|---|
| `test_make_fixed_sequences_output_shapes` | `X_cat(N, max_len, K)`, `X_num(N, max_len, M)`, `Y(N, max_len)` |
| `test_make_fixed_sequences_count_equals_unique_pa_ids` | First dimension equals the number of unique `pa_id`s |
| `test_make_fixed_sequences_long_PA_truncated` | A 12-pitch PA with `max_len=8` is cut to 8 timesteps |
| `test_make_fixed_sequences_short_PA_padded` | A 3-pitch PA with `max_len=8` gets PAD in positions 3–7 |
| `test_make_fixed_sequences_padding_is_suffix` | Real data occupies the prefix; padding is always at the end |
| `test_make_fixed_sequences_dtypes` | `X_cat → int64`, `X_num → float32`, `Y → int64` |
| `test_pitch_seq_ds_len` | `len(PitchSeqDS)` equals the number of sequences |
| `test_pitch_seq_ds_getitem` | `ds[i]` returns the correct slices with the expected shapes |

---

### `test_feature_calculator.py` — Feature Engineering

Covers the helper functions that compute pitch-level, game-state, and historical
rate features used as model inputs.

| Test group | What it checks |
|---|---|
| `pitch_to_family` | Pitch type classification into fastball / breaking / offspeed / `None` |
| `count_situation` | Returns `'ahead'`, `'behind'`, or `'even'` from balls/strikes |
| `calculate_pitch_features` | Adds `is_swing`, `is_whiff`, `in_zone`, `pitch_group`, etc. |
| `calculate_game_state_features` | Computes bitfield `base_state`, `count_state` string, `count_situation` |
| `add_pitcher_count_split_features` | Pitcher situational rates added; auto-injects `count_situation` if absent |
| `add_batter_count_split_features` | Batter situational swing/whiff rates; same auto-inject logic |
| `pitcher_family_lookup` / `add_pitcher_family_rate_features` | Overall FB/BR/OS rates per pitcher; stale pre-existing columns are recomputed |
| `get_vs_pitcher_stats` | Batter-vs-pitcher history lookup; empty result for unknown pairs |
| `situational_split` | Per-pitcher situational pitch-mix and whiff rates |
| `calculate_woba` | wOBA per batter using standard linear weights |

---

### `test_training.py` — Loss, Weights, Early Stopping, and Input Validation

Covers `FocalLoss`, `calculate_class_weights`, `EarlyStopping`, `_validate_feature_columns`,
and the guard logic at the entry point of `rnn_training_handler`.

| Test | What it checks |
|---|---|
| `test_focal_loss_ignores_pad_index` | All-PAD targets produce zero loss |
| `test_focal_loss_is_scalar` | Output is always a scalar tensor |
| `test_focal_loss_reduces_easy_examples` | Focal loss < cross-entropy when prediction is highly confident and correct |
| `test_focal_loss_non_negative` | Loss is always ≥ 0 |
| `test_calculate_class_weights_*` | PAD excluded; shape correct; rare classes weighted higher; smoothing=0 gives raw inverse |
| `test_early_stopping_*` | Triggers after `patience` non-improving epochs; resets on improvement; restores best weights; `delta` prevents premature trigger |
| `test_validate_*` | Missing cat col, num col, or target raises a descriptive `KeyError` listing all absent columns |
| `test_handler_missing_*_raises_before_training` | `rnn_training_handler` validates columns before any feature engineering or training begins |

---

### `test_trainer_utils.py` — Training Split and Sampling Utilities

Covers year-based splits, per-split feature lookups, the balanced sampler, and
arsenal mask construction.

| Test | What it checks |
|---|---|
| `test_split_by_year_*` | Rows split correctly by year; `train_start_year` filters old data; disjoint `pa_id` sets |
| `test_apply_pitcher_lookup_*` | Known pitcher gets merged stats; unknown pitcher row gets NaN |
| `test_apply_batter_lookup_*` | Same pattern for batter-side lookup |
| `test_apply_pitcher_family_lookup_*` | Overall pitch-mix rates merged by pitcher; NaN for unknown pitchers |
| `test_build_balanced_sampler_*` | Rare class gets proportionally higher weight; all-PAD sequences get weight 0; `num_samples` correct |
| `test_build_arsenal_masks_*` | Known pitcher has correct pitch blocked/allowed; unknown pitcher stays all-ones; year fallback to year-1 works; no matching year leaves row unchanged |

---

### `test_export_artifacts.py` — Artifact Persistence

Covers `export_model`, `export_vocabs`, `export_test_tensors`, `load_vocabs`, and
`get_latest_file`.

| Test group | What it checks |
|---|---|
| `export_model` | Custom path; datestamped default filename; default directory relative to module; nested dir creation; saved state dict is loadable |
| `export_vocabs` | Missing cat col raises `ValueError`; happy path creates file; default datestamped filename; default directory; payload has required keys; int keys are stringified for JSON |
| `export_test_tensors` | Custom path; datestamped filename; default directory; saved file contains `Xc`, `Xn`, `Y` keys |
| `load_vocabs` | `pitcher`/`batter` keys restored as `int`; other columns stay `str`; `y_vocab` and `feature_spec` round-trip correctly |
| `get_latest_file` | Returns most recently modified file; raises `FileNotFoundError` when no match |

---

## What Is Not Covered (and Why)

### Training Loop

The epoch-level training loop — iterating over batches, calling `loss.backward()`,
stepping the optimizer, and logging metrics — is intentionally excluded. This is not
appropriate for a unit test because:

- It requires real or realistic data at training scale to produce meaningful results.
- Correctness is not binary: loss going down is a stochastic property, not an assertion.
- Any meaningful check would be slow and flaky (GPU/CPU timing, randomness).

The components *around* the training loop (loss function, class weights, early stopping,
sampler, input validation) are all tested individually. That gives high confidence that
the loop will behave correctly when the tested pieces are wired together.

### Data Fetching

Pulling raw pitch data from Baseball Savant / `pybaseball` is a network operation with
no stable fixture. It is not tested here.

### Hyperparameter Search

Grid searches and manual tuning runs are notebook-driven workflows, not importable
functions. They don't have a testable unit boundary.

### Model Evaluation Metrics

Top-k accuracy, per-pitch-type confusion matrices, and calibration curves are computed
in evaluation notebooks after training. They depend on a trained model checkpoint, which
is not part of the test setup.
