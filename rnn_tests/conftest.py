import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

# Ensure the project root is importable when pytest is invoked from any cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Shared feature spec (minimal – 3 cat cols, 2 num cols)
# ---------------------------------------------------------------------------

MINI_FEATURE_SPEC = {
    "target": "y_next_pitch_type",
    "cat_cols": ["pitcher", "stand", "prev_pitch_type"],
    "num_cols": ["outs_when_up", "inning"],
}


@pytest.fixture
def feature_spec():
    return MINI_FEATURE_SPEC


# ---------------------------------------------------------------------------
# Vocabulary fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cat_vocabs():
    return {
        "pitcher": {200: 1, 201: 2},
        "stand": {"R": 1, "L": 2},
        "prev_pitch_type": {"FF": 1, "SL": 2, "CH": 3, "START": 4},
    }


@pytest.fixture
def sample_y_vocab():
    return {"FF": 1, "SL": 2, "CH": 3}


# ---------------------------------------------------------------------------
# Scaler fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_scaler():
    scaler = StandardScaler()
    train = pd.DataFrame({"outs_when_up": [0.0, 1.0, 2.0], "inning": [1.0, 5.0, 9.0]})
    scaler.fit(train)
    return scaler


# ---------------------------------------------------------------------------
# DataFrame builder helpers (importable by individual test modules)
# ---------------------------------------------------------------------------


def make_sort_df(rows):
    """rows: list of dicts with sort-key columns."""
    return pd.DataFrame(rows)


def make_universal_df(pitches, *, game_pk=1, at_bat_numbers=None, game_type="R"):
    """Minimal DataFrame accepted by universal_features."""
    n = len(pitches)
    if at_bat_numbers is None:
        at_bat_numbers = [1] * n
    return pd.DataFrame(
        {
            "game_type": game_type,
            "pitch_type": pitches,
            "game_pk": game_pk,
            "at_bat_number": at_bat_numbers,
            "pitch_number": range(1, n + 1),
        }
    )


def make_target_df(pitches_by_pa: dict):
    """
    pitches_by_pa: {pa_id: [pitch_type, ...]}
    Returns a DataFrame with pa_id and pitch_type already set,
    ready to pass directly to calculate_target_variable.
    """
    rows = []
    for pa_id, pitches in pitches_by_pa.items():
        for i, pt in enumerate(pitches):
            rows.append({"pa_id": pa_id, "pitch_type": pt, "pitch_number": i + 1})
    return pd.DataFrame(rows)


def make_encoded_df(pa_pitches: dict, feature_spec=None, y_val=1):
    """
    Build an already-encoded DataFrame for make_fixed_sequences tests.

    pa_pitches: {pa_id: n_pitches}
    """
    spec = feature_spec or MINI_FEATURE_SPEC
    cat_id_cols = [c + "_id" for c in spec["cat_cols"]]
    num_cols = spec["num_cols"]

    rows = []
    for pa_id, n in pa_pitches.items():
        for _ in range(n):
            row = {"pa_id": pa_id, "y_id": y_val}
            for col in cat_id_cols:
                row[col] = 1
            for col in num_cols:
                row[col] = 0.5
            rows.append(row)
    return pd.DataFrame(rows)
