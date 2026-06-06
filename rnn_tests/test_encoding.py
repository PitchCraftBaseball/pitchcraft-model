import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from pitch_rnn.encoder import build_vocab, encode, encode_df
from model_shared.feature_engineering.pitch_constants import PAD_ID

from rnn_tests.conftest import MINI_FEATURE_SPEC


def test_build_vocab_ids_start_at_1():
    """All assigned vocabulary IDs are >= 1; 0 is never assigned."""
    series = pd.Series(["FF", "SL", "CH", "CU"])
    vocab = build_vocab(series)
    assert all(v >= 1 for v in vocab.values())
    assert 0 not in vocab.values()


def test_build_vocab_excludes_NaN():
    """NaN values are dropped; the resulting dict has no NaN key and only real entries."""
    series = pd.Series(["FF", np.nan, "SL", np.nan])
    vocab = build_vocab(series)
    assert len(vocab) == 2
    assert not any(k != k for k in vocab)  # NaN != NaN


def test_build_vocab_unique_ids():
    """Duplicate input values collapse to a single entry with a unique ID."""
    series = pd.Series(["FF", "FF", "SL", "CH", "SL"])
    vocab = build_vocab(series)
    assert len(vocab) == 3
    assert len(set(vocab.values())) == 3  # all IDs distinct


def test_encode_known_values():
    """Mapped values match the vocab; result dtype is int."""
    vocab = {"FF": 1, "SL": 2}
    series = pd.Series(["FF", "SL", "FF"])
    result = encode(series, vocab)
    assert list(result) == [1, 2, 1]
    assert result.dtype == int


def test_encode_OOV_maps_to_PAD():
    """Out-of-vocabulary values encode to PAD_ID (0)."""
    vocab = {"FF": 1, "SL": 2}
    series = pd.Series(["CH"])  # 'CH' not in vocab
    result = encode(series, vocab)
    assert result.iloc[0] == PAD_ID


def test_encode_NaN_maps_to_PAD():
    """NaN in the input series encodes to PAD_ID (0)."""
    vocab = {"FF": 1, "SL": 2}
    series = pd.Series(["FF", np.nan, "SL"])
    result = encode(series, vocab)
    assert result.iloc[1] == PAD_ID


def _make_encode_df_input(n=10):
    """Return a minimal DataFrame and feature spec for encode_df tests."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "pitcher": rng.choice([200, 201], size=n),
            "stand": rng.choice(["R", "L"], size=n),
            "prev_pitch_type": rng.choice(["FF", "SL", "START"], size=n),
            "outs_when_up": rng.integers(0, 3, size=n).astype(float),
            "inning": rng.integers(1, 10, size=n).astype(float),
            "y_next_pitch_type": rng.choice(["FF", "SL", "CH"], size=n),
        }
    )
    return df


def test_encode_df_cat_id_columns_exist(feature_spec, sample_cat_vocabs, sample_y_vocab, fitted_scaler):
    """encode_df produces a '{col}_id' column for every cat col; all values >= 0."""
    df = _make_encode_df_input()
    out = encode_df(df, feature_spec, sample_cat_vocabs, sample_y_vocab, fitted_scaler)

    for col in feature_spec["cat_cols"]:
        id_col = col + "_id"
        assert id_col in out.columns, f"Missing column {id_col}"
        assert (out[id_col] >= 0).all(), f"{id_col} has negative values"


def test_encode_df_y_id_encodes_target(feature_spec, sample_cat_vocabs, sample_y_vocab, fitted_scaler):
    """y_vocab={'FF':1,'SL':2,'CH':3}; unknown target values map to PAD_ID (0)."""
    df = _make_encode_df_input()
    # Inject an unknown target value
    df.loc[0, "y_next_pitch_type"] = "UNKNOWN"

    out = encode_df(df, feature_spec, sample_cat_vocabs, sample_y_vocab, fitted_scaler)

    assert "y_id" in out.columns
    assert out.loc[0, "y_id"] == PAD_ID  # UNKNOWN → PAD


def test_encode_df_num_cols_float32_no_NaN(feature_spec, sample_cat_vocabs, sample_y_vocab, fitted_scaler):
    """All num cols become float32 with no NaN; non-parseable values become 0.0."""
    df = _make_encode_df_input()
    # Inject a NaN into a numeric column
    df.loc[0, "outs_when_up"] = np.nan

    out = encode_df(df, feature_spec, sample_cat_vocabs, sample_y_vocab, fitted_scaler)

    for col in feature_spec["num_cols"]:
        assert out[col].dtype == np.float32, f"{col} is not float32"
        assert not out[col].isna().any(), f"{col} contains NaN"


def test_encode_df_scaler_not_refit_on_test(feature_spec, sample_cat_vocabs, sample_y_vocab):
    """
    The scaler is fitted on train_df only; test_df is transformed (not refitted).
    After transform, the test column mean will not equal 0 when the distributions differ.
    """
    rng = np.random.default_rng(42)

    train_df = pd.DataFrame(
        {
            "pitcher": [200] * 20,
            "stand": ["R"] * 20,
            "prev_pitch_type": ["FF"] * 20,
            "outs_when_up": rng.normal(0, 1, 20),
            "inning": rng.normal(5, 1, 20),
            "y_next_pitch_type": ["FF"] * 20,
        }
    )
    # Test distribution is shifted significantly from train
    test_df = pd.DataFrame(
        {
            "pitcher": [200] * 20,
            "stand": ["R"] * 20,
            "prev_pitch_type": ["FF"] * 20,
            "outs_when_up": rng.normal(10, 1, 20),  # very different mean
            "inning": rng.normal(50, 1, 20),
            "y_next_pitch_type": ["FF"] * 20,
        }
    )

    scaler = StandardScaler()
    scaler.fit(train_df[feature_spec["num_cols"]].fillna(0))

    train_enc = encode_df(train_df, feature_spec, sample_cat_vocabs, sample_y_vocab, scaler)
    test_enc = encode_df(test_df, feature_spec, sample_cat_vocabs, sample_y_vocab, scaler)

    # Train mean ≈ 0 (fitted on train); test mean should be far from 0
    train_mean = train_enc["outs_when_up"].mean()
    test_mean = test_enc["outs_when_up"].mean()

    assert abs(train_mean) < 0.1, "Train mean should be ~0 after StandardScaler"
    assert abs(test_mean) > 1.0, "Test mean should differ from 0 (scaler not refitted)"
