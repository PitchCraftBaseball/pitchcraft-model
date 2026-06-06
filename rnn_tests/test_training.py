import numpy as np
import pytest
import torch
import torch.nn as nn

import pandas as pd

from pitch_rnn.pitch_rnn_trainer import FocalLoss, calculate_class_weights, rnn_training_handler, _validate_feature_columns
from pitch_rnn.early_stopping import EarlyStopping
from model_shared.feature_engineering.pitch_constants import PAD_ID


def test_focal_loss_ignores_pad_index():
    """FocalLoss with ignore_index=PAD_ID should not penalise PAD positions."""
    loss_fn = FocalLoss(gamma=2.0, ignore_index=PAD_ID)

    num_classes = 5
    # All-PAD targets → loss should be 0
    logits = torch.randn(8, num_classes)
    targets = torch.zeros(8, dtype=torch.long)  # all PAD

    loss = loss_fn(logits, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_focal_loss_is_scalar():
    """FocalLoss always returns a scalar tensor."""
    loss_fn = FocalLoss(gamma=2.0, ignore_index=PAD_ID)
    logits = torch.randn(16, 8)
    targets = torch.randint(1, 8, (16,))
    loss = loss_fn(logits, targets)
    assert loss.shape == torch.Size([])


def test_focal_loss_reduces_easy_examples():
    """
    Focal loss down-weights easy examples relative to cross-entropy.
    When gamma>0 and a prediction is very confident and correct, focal
    weight (1-pt)^gamma is small → focal < CE.
    """
    loss_fn_focal = FocalLoss(gamma=2.0, ignore_index=PAD_ID)
    loss_fn_ce = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # Construct logits where the correct class has very high probability.
    # Values must be moderate enough that float32 CE is non-zero (10.0 saturates to 0.0).
    num_classes = 5
    logits = torch.full((10, num_classes), -5.0)
    logits[:, 1] = 5.0  # class 1 is almost certainly predicted
    targets = torch.ones(10, dtype=torch.long)  # correct class is 1

    focal = loss_fn_focal(logits, targets)
    ce = loss_fn_ce(logits, targets)

    assert focal.item() < ce.item()


def test_focal_loss_non_negative():
    """Focal loss is always >= 0."""
    loss_fn = FocalLoss(gamma=2.0, ignore_index=PAD_ID)
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0.0



def test_calculate_class_weights_pads_excluded():
    """PAD positions (value==pad_id) are excluded from the weight computation."""
    Y = torch.tensor([[0, 1, 2, 0]])  # 0 is PAD
    weights = calculate_class_weights(Y, num_classes=4, pad_id=PAD_ID)
    assert weights[PAD_ID].item() == pytest.approx(0.0)


def test_calculate_class_weights_shape():
    """Output tensor has length == num_classes."""
    Y = torch.tensor([[1, 2, 1, 3]])
    weights = calculate_class_weights(Y, num_classes=6, pad_id=PAD_ID)
    assert weights.shape == (6,)


def test_calculate_class_weights_rare_class_gets_higher_weight():
    """A rare class receives more weight than a common class."""
    # Class 1 appears once, class 2 appears nine times
    Y = torch.tensor([[1, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    weights = calculate_class_weights(Y, num_classes=4, pad_id=PAD_ID, smoothing=0.35)
    assert weights[1].item() > weights[2].item()


def test_calculate_class_weights_smoothing_zero_gives_raw_inverse():
    """With smoothing=0, weight = (1/freq)^0 = 1.0 for all present classes."""
    Y = torch.tensor([[1, 1, 2, 2, 2]])
    weights = calculate_class_weights(Y, num_classes=4, pad_id=PAD_ID, smoothing=0.0)
    # Both classes present → weight = 1.0
    assert weights[1].item() == pytest.approx(1.0)
    assert weights[2].item() == pytest.approx(1.0)
    # Absent class → 0
    assert weights[3].item() == pytest.approx(0.0)


class _DummyModel(nn.Module):
    """Minimal model whose state_dict can be saved/restored."""
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))


def test_early_stopping_triggers_after_patience():
    """early_stop flag is set after `patience` non-improving epochs."""
    es = EarlyStopping(patience=3, delta=0.0)
    model = _DummyModel()

    es(1.0, model)  # epoch 1 — new best
    assert not es.early_stop

    for _ in range(3):  # 3 non-improving epochs
        es(1.1, model)

    assert es.early_stop


def test_early_stopping_resets_counter_on_improvement():
    """Counter resets to 0 when validation loss improves."""
    es = EarlyStopping(patience=3, delta=0.0)
    model = _DummyModel()

    es(1.0, model)   # best = 1.0
    es(1.1, model)   # counter = 1
    es(1.1, model)   # counter = 2
    es(0.5, model)   # improvement → counter resets
    assert es.counter == 0
    assert not es.early_stop


def test_early_stopping_loads_best_model():
    """load_best_model restores the weights from the best epoch."""
    es = EarlyStopping(patience=5, delta=0.0)
    model = _DummyModel()

    # Epoch 1: best (param==1.0)
    es(0.5, model)
    best_param = model.param.item()

    # Change the model weights and record a worse loss
    with torch.no_grad():
        model.param.fill_(99.0)
    es(1.0, model)

    es.load_best_model(model)
    assert model.param.item() == pytest.approx(best_param)


def test_early_stopping_delta_prevents_premature_trigger():
    """With delta=0.05, a loss improvement < delta does not reset the counter."""
    es = EarlyStopping(patience=2, delta=0.05)
    model = _DummyModel()

    es(1.0, model)              # best
    es(0.98, model)             # improvement of 0.02 < delta → counter increments
    assert es.counter == 1
    es(0.96, model)             # improvement of 0.02 < delta → counter increments
    assert es.counter == 2
    assert es.early_stop        # patience=2 exhausted


# Feature spec used for handler validation tests; includes count_state so we
# can exercise both the cat_col and target missing-column paths.
_HANDLER_SPEC = {
    "target": "y_next_pitch_type",
    "cat_cols": ["pitcher", "stand", "count_state"],
    "num_cols": ["outs_when_up"],
}


def _valid_handler_df() -> pd.DataFrame:
    """Minimal DataFrame containing every column in _HANDLER_SPEC."""
    return pd.DataFrame({
        "pitcher":          [100],
        "stand":            ["R"],
        "count_state":      ["0-0"],
        "outs_when_up":     [1],
        "y_next_pitch_type": ["FF"],
    })


def test_validate_no_missing_columns_passes():
    """All required columns present → no exception raised."""
    _validate_feature_columns(_valid_handler_df(), _HANDLER_SPEC)


def test_validate_missing_cat_col_raises_key_error():
    """A cat_col absent from the DataFrame raises a descriptive KeyError."""
    df = _valid_handler_df().drop(columns=["count_state"])
    with pytest.raises(KeyError, match="count_state"):
        _validate_feature_columns(df, _HANDLER_SPEC)


def test_validate_missing_target_col_raises_key_error():
    """The target column absent from the DataFrame raises a descriptive KeyError."""
    df = _valid_handler_df().drop(columns=["y_next_pitch_type"])
    with pytest.raises(KeyError, match="y_next_pitch_type"):
        _validate_feature_columns(df, _HANDLER_SPEC)


def test_validate_multiple_missing_cols_all_listed():
    """All missing columns are named together in the error message."""
    df = _valid_handler_df().drop(columns=["count_state", "y_next_pitch_type"])
    with pytest.raises(KeyError) as exc_info:
        _validate_feature_columns(df, _HANDLER_SPEC)
    assert "count_state" in str(exc_info.value)
    assert "y_next_pitch_type" in str(exc_info.value)


# --- rnn_training_handler integration: error raised before training loop ---

def test_handler_missing_cat_col_raises_before_training():
    """TEST_MISSING_CAT_COLS: dropping a cat_col causes the handler to raise
    immediately — no feature engineering or training loop is entered."""
    df = _valid_handler_df().drop(columns=["count_state"])
    with pytest.raises(KeyError, match="count_state"):
        rnn_training_handler(df, _HANDLER_SPEC, custom_emb_dims=None, model_params=None)


def test_handler_missing_target_col_raises_before_training():
    """Dropping the target column causes the handler to raise immediately —
    vocabulary construction and the training loop are never reached."""
    df = _valid_handler_df().drop(columns=["y_next_pitch_type"])
    with pytest.raises(KeyError, match="y_next_pitch_type"):
        rnn_training_handler(df, _HANDLER_SPEC, custom_emb_dims=None, model_params=None)


def test_validate_missing_num_col_raises_key_error():
    """A num_col absent from the DataFrame raises a descriptive KeyError."""
    df = _valid_handler_df().drop(columns=["outs_when_up"])
    with pytest.raises(KeyError, match="outs_when_up"):
        _validate_feature_columns(df, _HANDLER_SPEC)


def test_handler_missing_num_col_raises_before_training():
    """TEST_MISSING_NUM_COLS: dropping a num_col causes the handler to raise
    immediately — the missing column is not silently zeroed out."""
    df = _valid_handler_df().drop(columns=["outs_when_up"])
    with pytest.raises(KeyError, match="outs_when_up"):
        rnn_training_handler(df, _HANDLER_SPEC, custom_emb_dims=None, model_params=None)
