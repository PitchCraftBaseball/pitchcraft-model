"""
Tests for training utilities: FocalLoss, calculate_class_weights, EarlyStopping,
and the calculate_class_weights helper.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from pitch_rnn.pitch_rnn_trainer import FocalLoss, calculate_class_weights
from pitch_rnn.early_stopping import EarlyStopping
from model_shared.feature_engineering.pitch_constants import PAD_ID


# ===========================================================================
# FocalLoss
# ===========================================================================


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

    # Construct logits where the correct class has very high probability
    num_classes = 5
    logits = torch.full((10, num_classes), -10.0)
    logits[:, 1] = 10.0  # class 1 is almost certainly predicted
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


# ===========================================================================
# calculate_class_weights
# ===========================================================================


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


# ===========================================================================
# EarlyStopping
# ===========================================================================


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
