"""
Tests for the PitchRNN model definition.
Covers instantiation, forward-pass shapes, and embedding behaviour.
"""

import torch
import pytest

from model_shared.rnn_definition import PitchRNN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CAT_VOCAB_SIZES = {"pitcher": 50, "stand": 3, "prev_pitch_type": 20}
_EMB_DIMS = {"pitcher": 16, "stand": 4, "prev_pitch_type": 8}
_NUM_FEATURES = 5
_NUM_CLASSES = 12
_HIDDEN = 64
_BATCH = 4
_SEQ = 8


@pytest.fixture
def model():
    return PitchRNN(
        cat_vocab_sizes=_CAT_VOCAB_SIZES,
        num_features=_NUM_FEATURES,
        emb_dims=_EMB_DIMS,
        hidden=_HIDDEN,
        num_classes=_NUM_CLASSES,
        dropout=0.0,
        num_layers=1,
    )


def _make_batch(batch=_BATCH, seq=_SEQ, n_cat=len(_CAT_VOCAB_SIZES), n_num=_NUM_FEATURES):
    x_cat = torch.zeros(batch, seq, n_cat, dtype=torch.int64)
    x_num = torch.randn(batch, seq, n_num)
    return x_cat, x_num


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_model_output_shape(model):
    """Forward pass returns (batch, seq_len, num_classes)."""
    x_cat, x_num = _make_batch()
    with torch.no_grad():
        out = model(x_cat, x_num)
    assert out.shape == (_BATCH, _SEQ, _NUM_CLASSES)


def test_model_default_emb_dims():
    """When emb_dims is None every embedding defaults to dim 16."""
    m = PitchRNN(
        cat_vocab_sizes=_CAT_VOCAB_SIZES,
        num_features=_NUM_FEATURES,
        emb_dims=None,
        hidden=_HIDDEN,
        num_classes=_NUM_CLASSES,
    )
    for col in _CAT_VOCAB_SIZES:
        assert m.embs[col].embedding_dim == 16


def test_model_padding_idx_is_zero(model):
    """Every embedding layer uses padding_idx=0."""
    for col in _CAT_VOCAB_SIZES:
        assert model.embs[col].padding_idx == 0


def test_model_num_layers():
    """num_layers is forwarded correctly to the GRU."""
    m = PitchRNN(
        cat_vocab_sizes=_CAT_VOCAB_SIZES,
        num_features=_NUM_FEATURES,
        hidden=_HIDDEN,
        num_classes=_NUM_CLASSES,
        num_layers=2,
    )
    assert m.rnn.num_layers == 2


def test_model_fc_out_dim(model):
    """The final linear layer outputs num_classes logits."""
    assert model.fc.out_features == _NUM_CLASSES


def test_model_eval_deterministic(model):
    """The same input produces identical outputs in eval mode (no dropout noise)."""
    model.eval()
    x_cat, x_num = _make_batch()
    with torch.no_grad():
        out1 = model(x_cat, x_num)
        out2 = model(x_cat, x_num)
    assert torch.allclose(out1, out2)


def test_model_pad_token_gradient_is_zero(model):
    """
    Embedding gradient for padding_idx=0 remains zero after a forward/backward pass,
    confirming the model does not learn from PAD positions.
    """
    model.train()
    x_cat, x_num = _make_batch()
    out = model(x_cat, x_num)
    loss = out.sum()
    loss.backward()

    for col in _CAT_VOCAB_SIZES:
        grad = model.embs[col].weight.grad
        if grad is not None:
            assert grad[0].abs().sum().item() == pytest.approx(0.0)
