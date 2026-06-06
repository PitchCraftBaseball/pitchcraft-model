import numpy as np
import pandas as pd
import pytest
import torch

from pitch_rnn.pitch_rnn_trainer import make_fixed_sequences
from pitch_rnn.sequence_builder import PitchSeqDS
from model_shared.feature_engineering.pitch_constants import PAD_ID

from rnn_tests.conftest import MINI_FEATURE_SPEC, make_encoded_df

_SPEC = MINI_FEATURE_SPEC
_CAT_COLS = _SPEC["cat_cols"]  # ["pitcher", "stand", "prev_pitch_type"]
_NUM_COLS = _SPEC["num_cols"]  # ["outs_when_up", "inning"]
_K = len(_CAT_COLS)  # 3
_M = len(_NUM_COLS)  # 2


def _uniform_pa_df(n_pa: int, pitches_per_pa: int, y_val: int = 1) -> pd.DataFrame:
    """Encoded DataFrame with n_pa PAs each having pitches_per_pa pitches."""
    pa_pitches = {f"pa{i}": pitches_per_pa for i in range(n_pa)}
    return make_encoded_df(pa_pitches, feature_spec=_SPEC, y_val=y_val)


def test_make_fixed_sequences_output_shapes():
    """X_cat(N,8,K), X_num(N,8,M), Y(N,8) for N PAs with K cat and M num features."""
    N, MAX = 5, 8
    df = _uniform_pa_df(N, pitches_per_pa=4)
    Xc, Xn, Y = make_fixed_sequences(df, _SPEC, max_len=MAX)

    assert Xc.shape == (N, MAX, _K)
    assert Xn.shape == (N, MAX, _M)
    assert Y.shape == (N, MAX)


def test_make_fixed_sequences_count_equals_unique_pa_ids():
    """First dimension of X_cat equals the number of unique pa_ids."""
    N = 50
    df = _uniform_pa_df(N, pitches_per_pa=3)
    Xc, _, _ = make_fixed_sequences(df, _SPEC, max_len=8)

    assert Xc.shape[0] == N


def test_make_fixed_sequences_long_PA_truncated():
    """A 12-pitch PA with max_len=8 → only 8 timesteps; pitches 9-12 are absent."""
    df = make_encoded_df({"pa0": 12}, feature_spec=_SPEC)
    Xc, Xn, Y = make_fixed_sequences(df, _SPEC, max_len=8)

    # Shape dimension 1 must be 8
    assert Xc.shape[1] == 8
    assert Xn.shape[1] == 8
    assert Y.shape[1] == 8

    # All 8 positions should be filled (not PAD), because the PA has >=8 real pitches
    assert (Y[0] != PAD_ID).all()


def test_make_fixed_sequences_short_PA_padded():
    """3-pitch PA with max_len=8: positions 3-7 in Y and X_cat are PAD_ID; X_num==0.0."""
    df = make_encoded_df({"pa0": 3}, feature_spec=_SPEC, y_val=1)
    Xc, Xn, Y = make_fixed_sequences(df, _SPEC, max_len=8)

    # Positions 3-7 should be PAD
    assert (Y[0, 3:] == PAD_ID).all()
    assert (Xc[0, 3:] == PAD_ID).all()
    assert (Xn[0, 3:] == 0.0).all()


def test_make_fixed_sequences_padding_is_suffix():
    """2-pitch PA with max_len=8: positions 0-1 hold real data; 2-7 are zero/PAD."""
    df = make_encoded_df({"pa0": 2}, feature_spec=_SPEC, y_val=2)
    Xc, Xn, Y = make_fixed_sequences(df, _SPEC, max_len=8)

    # Real data is in prefix positions
    assert Y[0, 0] == 2
    assert Y[0, 1] == 2

    # Suffix is PAD
    assert (Y[0, 2:] == PAD_ID).all()
    assert (Xc[0, 2:] == PAD_ID).all()
    assert (Xn[0, 2:] == 0.0).all()


def test_make_fixed_sequences_dtypes():
    """X_cat → torch.int64, X_num → torch.float32, Y → torch.int64."""
    df = _uniform_pa_df(3, pitches_per_pa=4)
    Xc, Xn, Y = make_fixed_sequences(df, _SPEC, max_len=8)

    assert Xc.dtype == torch.int64
    assert Xn.dtype == torch.float32
    assert Y.dtype == torch.int64


def test_pitch_seq_ds_len():
    """len(PitchSeqDS) equals the number of PAs (first dimension of Y)."""
    N = 6
    df = _uniform_pa_df(N, pitches_per_pa=4)
    Xc, Xn, Y = make_fixed_sequences(df, _SPEC, max_len=8)
    ds = PitchSeqDS(Xc, Xn, Y)
    assert len(ds) == N


def test_pitch_seq_ds_getitem():
    """ds[i] returns (Xc[i], Xn[i], Y[i]) with correct shapes."""
    N, MAX = 4, 8
    df = _uniform_pa_df(N, pitches_per_pa=3)
    Xc, Xn, Y = make_fixed_sequences(df, _SPEC, max_len=MAX)
    ds = PitchSeqDS(Xc, Xn, Y)
    xc_i, xn_i, y_i = ds[0]
    assert xc_i.shape == Xc[0].shape
    assert xn_i.shape == Xn[0].shape
    assert y_i.shape == Y[0].shape
