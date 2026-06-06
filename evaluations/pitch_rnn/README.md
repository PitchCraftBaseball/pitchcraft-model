# Pitch RNN Evaluation

Post-training evaluation of the `PitchRNN` model. Runs automatically at the end of `trainer_engine.py` via `evaluate_rnn()`, or can be run standalone against any saved checkpoint.

## Overview

Evaluation loads the most recently saved model weights, vocabulary, and held-out test tensors from `model_shared/`, runs inference on the 2025 test set, calibrates the output temperature, then computes a suite of metrics and saves charts to `eval_output/`.

---

## Entry Point

```python
from evaluations.pitch_rnn.evaluate_rnn import evaluate_rnn

evaluate_rnn(
    emb_dims=EMB_DIMS,
    num_layers=2,
    use_arsenal_mask=False,
    hidden=128,
)
```

Called automatically by `trainer_engine.py` after training completes.

---

## Pipeline

### 1. Load Artifacts (`load_model_and_vocabs`, `load_test_loader`)

Three files are resolved by finding the most recently modified file matching each pattern in `model_shared/`:

| Artifact | Pattern | Contents |
|---|---|---|
| Model weights | `trained-parameters/pitch_rnn_*.pt` | `state_dict` from training |
| Vocabulary | `vocab/rnn_vocab_*.json` | Cat vocabs, y vocab, feature spec |
| Test tensors | `test_data/test_tensors_*.pt` | `Xc`, `Xn`, `Y` tensors saved at train time |

The model is reconstructed from the vocabulary (which contains the feature spec), loaded in `eval` mode, and the test tensors are wrapped in a `DataLoader`.

### 2. Arsenal Masks (`build_arsenal_masks`)

When `use_arsenal_mask=True`, a per-pitcher mask is built from `pitch_arsenal/arsenals_all.json`. For each pitcher in the vocabulary, classes for pitches **not** in their known arsenal are set to `-inf` before `argmax`, so the model can only predict pitch types the pitcher actually throws.

If a pitcher's 2025 arsenal is not available, their 2024 arsenal is used as a fallback.

### 3. Temperature Calibration (`find_optimal_temperature`)

After training, raw logits tend to be overconfident. Temperature scaling divides all logits by a scalar `T` before softmax — higher T flattens the distribution, lower T sharpens it.

The optimal temperature is found by minimizing cross-entropy on the test set using L-BFGS with `T` as a single learnable parameter (initialized at 1.5, clamped to ≥ 0.1). The result is written back to `model_shared/vocab/temperature.json` so the inference engine picks it up automatically.

### 4. Evaluation Suite (`evaluate_model_complete`)

All metrics are computed with PAD positions masked out. Arsenal masks and the calibrated temperature are applied throughout.

#### Token Accuracy
Percentage of individual pitch predictions that exactly match the ground truth, across all non-PAD positions in the test set.

#### Top-K Accuracy (`get_top_k_accuracy`)
Percentage of positions where the true label appears in the model's top-K predictions. Reported at K=2.

#### Most Common Predictions (`get_most_common_pitches`)
Distribution of what the model most frequently predicts — a sanity check for mode collapse toward common pitch types.

#### Classification Report (`print_classification_report`)
Per-class precision, recall, and F1 using `sklearn.metrics.classification_report`. Useful for identifying which pitch types the model struggles with most.

#### Confusion Matrix (`generate_confusion_matrix`)
Row-normalized confusion matrix (each row sums to 1, so diagonal = recall per class). Plotted as a heatmap and saved to `eval_output/<timestamp>/confusion_matrix_<timestamp>.png`.

#### Calibration Curves (`generate_calibration_curves`)
Reliability diagram comparing mean predicted probability vs. actual positive rate for each pitch family (fastball, breaking, offspeed). A perfectly calibrated model sits on the diagonal. Saved to `eval_output/<timestamp>/calibration_curves_<timestamp>.png`.

#### Positional Accuracy (`get_positional_accuracy`)
Accuracy broken down by position within the plate appearance (pitch 1 through pitch 8). Shows whether the model improves as it sees more pitches in the sequence.

---

## Output Files

Each evaluation run creates a timestamped subdirectory under `eval_output/`:

```
eval_output/
└── YYYYMMDD_HHMMSS/
    ├── confusion_matrix_YYYYMMDD_HHMMSS.png
    └── calibration_curves_YYYYMMDD_HHMMSS.png
```

Metric JSON files (accuracy, top-K, classification report) are written in some runs alongside the plots.

---

## Module Reference

| Function | Description |
|---|---|
| `evaluate_rnn` | Top-level entry point; resolves artifacts, calibrates temperature, runs full suite |
| `load_model_and_vocabs` | Reconstructs `PitchRNN` from vocab file and loads weights |
| `load_test_loader` | Wraps saved test tensors in a `DataLoader` |
| `load_arsenals` | Reads the pitcher arsenal JSON |
| `build_arsenal_masks` | Builds per-pitcher class masks from arsenal data |
| `get_all_predictions` | Runs inference and returns flat `(y_true, y_pred)` arrays |
| `find_optimal_temperature` | L-BFGS temperature search, writes result to `temperature.json` |
| `get_accuracy` | Overall token accuracy |
| `get_top_k_accuracy` | Top-K accuracy |
| `get_most_common_pitches` | Prediction frequency distribution |
| `print_classification_report` | Per-class precision/recall/F1 |
| `generate_confusion_matrix` | Row-normalized heatmap, saved to disk |
| `generate_calibration_curves` | Reliability diagram by pitch family, saved to disk |
| `get_positional_accuracy` | Accuracy by pitch position within the plate appearance |
| `evaluate_model_complete` | Runs all of the above in sequence |
