import numpy as np
import pandas as pd
import pytest

from model_shared.feature_engineering.data_preprocessor import (
    clean_data,
    data_remapping,
    drop_unused_cols,
    sort_statcast,
    universal_features,
)
from pitch_rnn.pitch_rnn_trainer import calculate_target_variable, split_by_pa_id

from rnn_tests.conftest import make_target_df, make_universal_df


def _make_clean_data_df(pitches, at_bat_numbers=None):
    """Minimal DataFrame that satisfies all four steps of clean_data."""
    n = len(pitches)
    if at_bat_numbers is None:
        at_bat_numbers = [1] * n
    return pd.DataFrame({
        "game_date": "2024-04-01",
        "game_pk": 1,
        "game_type": "R",
        "inning": 1,
        "inning_topbot": "Top",
        "at_bat_number": at_bat_numbers,
        "pitch_number": range(1, n + 1),
        "pitch_type": pitches,
        "description": "ball",
        "player_name": "John Doe",
        "sv_id": "abc123",
    })


def test_sort_statcast_ascending_order():
    """Rows sort game_date→game_pk→inning→Top before Bot→at_bat→pitch; index resets."""
    df = pd.DataFrame(
        [
            # Wrong order: later date first, Bot before Top, pitch 2 before 1
            {
                "game_date": "2024-04-02",
                "game_pk": 1,
                "inning": 1,
                "inning_topbot": "Bot",
                "at_bat_number": 3,
                "pitch_number": 1,
            },
            {
                "game_date": "2024-04-01",
                "game_pk": 1,
                "inning": 1,
                "inning_topbot": "Top",
                "at_bat_number": 1,
                "pitch_number": 2,
            },
            {
                "game_date": "2024-04-01",
                "game_pk": 1,
                "inning": 1,
                "inning_topbot": "Top",
                "at_bat_number": 1,
                "pitch_number": 1,
            },
            {
                "game_date": "2024-04-01",
                "game_pk": 1,
                "inning": 1,
                "inning_topbot": "Bot",
                "at_bat_number": 2,
                "pitch_number": 1,
            },
        ]
    )
    result = sort_statcast(df)

    # Index must be a clean 0-based range
    assert list(result.index) == [0, 1, 2, 3]

    # game_date ascending
    assert list(result["game_date"]) == [
        "2024-04-01",
        "2024-04-01",
        "2024-04-01",
        "2024-04-02",
    ]

    # Within the same inning, Top (descending sort → "Top" > "Bot") comes first
    assert list(result["inning_topbot"]) == ["Top", "Top", "Bot", "Bot"]

    # Pitch numbers within an at-bat are ascending
    assert list(result["pitch_number"][:2]) == [1, 2]


def test_universal_features_game_type_filter():
    """Only rows with game_type=='R' survive."""
    df = pd.DataFrame(
        {
            "game_type": ["R", "S", "R", "E"],
            "pitch_type": ["FF", "FF", "SL", "CH"],
            "game_pk": [1, 1, 1, 1],
            "at_bat_number": [1, 2, 3, 4],
            "pitch_number": [1, 1, 1, 1],
        }
    )
    result = universal_features(df)
    assert set(result["game_type"].unique()) == {"R"}
    assert len(result) == 2


def test_universal_features_drops_UN_pitch_type():
    """Rows with pitch_type=='UN' are removed."""
    df = make_universal_df(["FF", "UN", "SL", "UN"])
    result = universal_features(df)
    assert "UN" not in result["pitch_type"].values
    assert len(result) == 2


def test_universal_features_pa_id_construction():
    """game_pk=12345, at_bat_number=7 → pa_id=='12345_7'."""
    df = make_universal_df(["FF"], game_pk=12345, at_bat_numbers=[7])
    result = universal_features(df)
    assert result.iloc[0]["pa_id"] == "12345_7"


def test_universal_features_prev_pitch_type_START_for_first_pitch():
    """First pitch of every PA gets prev_pitch_type=='START'."""
    # Two at-bats in the same game
    df = make_universal_df(
        ["FF", "SL", "CH", "FF"],
        game_pk=1,
        at_bat_numbers=[1, 1, 2, 2],
    )
    result = universal_features(df)

    # First pitch of at-bat 1
    pa1_rows = result[result["at_bat_number"] == 1].sort_values("pitch_number")
    assert pa1_rows.iloc[0]["prev_pitch_type"] == "START"

    # First pitch of at-bat 2
    pa2_rows = result[result["at_bat_number"] == 2].sort_values("pitch_number")
    assert pa2_rows.iloc[0]["prev_pitch_type"] == "START"


def test_universal_features_prev_pitch_skips_ABS():
    """In sequence FF→ABS→SL, SL's prev_pitch_type is 'FF', not 'ABS'."""
    df = make_universal_df(["FF", "ABS", "SL"], game_pk=1, at_bat_numbers=[1, 1, 1])
    result = universal_features(df).sort_values("pitch_number").reset_index(drop=True)

    sl_row = result[result["pitch_type"] == "SL"].iloc[0]
    assert sl_row["prev_pitch_type"] == "FF"


def test_universal_features_prev_pitch_stays_within_PA():
    """First pitch of PA 2 gets 'START'; it never inherits the last pitch of PA 1."""
    df = make_universal_df(
        ["FF", "SL", "CH", "SL"],
        game_pk=1,
        at_bat_numbers=[1, 1, 2, 2],
    )
    result = universal_features(df)

    pa2_rows = result[result["at_bat_number"] == 2].sort_values("pitch_number")
    assert pa2_rows.iloc[0]["prev_pitch_type"] == "START"


def test_universal_features_seq_len():
    """Every row in a PA carries seq_len equal to total pitches in that PA."""
    df = make_universal_df(
        ["FF", "SL", "CH", "FF", "SL"],
        game_pk=1,
        at_bat_numbers=[1, 1, 1, 2, 2],
    )
    result = universal_features(df)

    pa1 = result[result["at_bat_number"] == 1]
    assert (pa1["seq_len"] == 3).all()

    pa2 = result[result["at_bat_number"] == 2]
    assert (pa2["seq_len"] == 2).all()



def test_calculate_target_is_real_pitch_false_for_IGNORE():
    """IGNORE pitches get is_real_pitch==False; valid pitches get True."""
    df = make_target_df({"pa1": ["FF", "ABS", "PO", "FA", "EP", "SL"]})
    calculate_target_variable(df)  # mutates df in-place

    ignore_mask = df["pitch_type"].isin({"ABS", "PO", "FA", "EP"})
    assert df.loc[ignore_mask, "is_real_pitch"].eq(False).all()
    assert df.loc[~ignore_mask, "is_real_pitch"].eq(True).all()


def test_calculate_target_y_next_pitch_type_shift():
    """3-pitch PA FF→SL→CH: FF's y==SL, SL's y==CH, CH's y==NaN (and filtered out)."""
    df = make_target_df({"pa1": ["FF", "SL", "CH"]})
    result = calculate_target_variable(df)

    # The returned data_train excludes the last pitch (no next real pitch)
    assert set(result["pitch_type"]) == {"FF", "SL"}

    ff_row = result[result["pitch_type"] == "FF"].iloc[0]
    sl_row = result[result["pitch_type"] == "SL"].iloc[0]

    assert ff_row["y_next_pitch_type"] == "SL"
    assert sl_row["y_next_pitch_type"] == "CH"

    # The original df (mutated) shows NaN for the last pitch
    ch_mask = df["pitch_type"] == "CH"
    assert df.loc[ch_mask, "y_next_pitch_type"].isna().all()


def test_calculate_target_no_cross_PA_bleed():
    """Last pitch of PA 1 gets y_next_pitch_type==NaN, not the first pitch of PA 2."""
    df = make_target_df({"pa1": ["FF", "SL"], "pa2": ["CH", "FF"]})
    calculate_target_variable(df)  # mutate in-place to inspect all rows

    # Last pitch of pa1 (SL at pitch_number==2)
    last_pa1 = df[(df["pa_id"] == "pa1") & (df["pitch_number"] == 2)]
    assert last_pa1["y_next_pitch_type"].isna().all()


def test_calculate_target_excludes_non_real_targets():
    """A pitch whose next pitch is an IGNORE type is excluded from the training set."""
    # FF → ABS (IGNORE) → SL
    # FF's target_is_real_pitch is False (ABS is not real) → FF excluded
    df = make_target_df({"pa1": ["FF", "ABS", "SL"]})
    result = calculate_target_variable(df)

    pitch_types_in_result = set(result["pitch_type"].values)
    assert "FF" not in pitch_types_in_result  # followed by IGNORE → excluded


def test_split_by_pa_id_ratio_assertion():
    """Ratios that don't sum to 1.0 raise AssertionError."""
    df = pd.DataFrame({"pa_id": [f"pa{i}" for i in range(10)]})
    with pytest.raises(AssertionError):
        split_by_pa_id(df, ratios=(0.7, 0.25))


def test_split_by_pa_id_disjoint_sets():
    """100 unique pa_ids → train ∩ test == ∅ and |train| + |test| == 100."""
    pa_ids = [f"pa{i}" for i in range(100)]
    df = pd.DataFrame({"pa_id": pa_ids})

    _, _, train_ids, test_ids = split_by_pa_id(df, ratios=(0.8, 0.2), seed=42)

    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == 100


def test_split_by_pa_id_reproducibility():
    """The same seed produces identical splits on two independent calls."""
    pa_ids = [f"pa{i}" for i in range(200)]
    df = pd.DataFrame({"pa_id": pa_ids})

    _, _, train_ids_a, test_ids_a = split_by_pa_id(df, ratios=(0.8, 0.2), seed=42)
    _, _, train_ids_b, test_ids_b = split_by_pa_id(df, ratios=(0.8, 0.2), seed=42)

    assert train_ids_a == train_ids_b
    assert test_ids_a == test_ids_b


def test_data_remapping_pitch_types():
    """SC→'CU', CS→'CU', FO→'FS'; unrelated types are unchanged."""
    df = pd.DataFrame(
        {
            "pitch_type": ["SC", "CS", "FO", "FF", "SL"],
            "description": ["ball"] * 5,
        }
    )
    result = data_remapping(df)
    assert list(result["pitch_type"]) == ["CU", "CU", "FS", "FF", "SL"]


def test_data_remapping_ABS_override():
    """pitch_type overridden to 'ABS' when description is automatic_ball/automatic_strike."""
    df = pd.DataFrame(
        {
            "pitch_type": ["FF", "FF", "SL"],
            "description": ["automatic_ball", "automatic_strike", "ball"],
        }
    )
    result = data_remapping(df)
    assert result.iloc[0]["pitch_type"] == "ABS"
    assert result.iloc[1]["pitch_type"] == "ABS"
    assert result.iloc[2]["pitch_type"] == "SL"  # untouched


def test_drop_unused_cols_removes_known_columns():
    """Columns in the hard-coded drop list are removed from the output."""
    df = pd.DataFrame({
        "pitch_type": ["FF"],
        "player_name": ["John Doe"],
        "sv_id": ["abc"],
        "spin_dir": [0.5],
    })
    result = drop_unused_cols(df)
    assert "player_name" not in result.columns
    assert "sv_id" not in result.columns
    assert "spin_dir" not in result.columns
    assert "pitch_type" in result.columns


def test_drop_unused_cols_ignores_absent_columns():
    """No error is raised when drop-list columns are absent (errors='ignore')."""
    df = pd.DataFrame({"pitch_type": ["FF", "SL"], "game_pk": [1, 1]})
    result = drop_unused_cols(df)
    assert set(result.columns) == {"pitch_type", "game_pk"}


def test_drop_unused_cols_preserves_non_drop_columns():
    """Columns not in the drop list are kept intact."""
    df = pd.DataFrame({
        "pitch_type": ["FF"],
        "balls": [1],
        "strikes": [2],
        "player_name": ["Jane"],
    })
    result = drop_unused_cols(df)
    assert "balls" in result.columns
    assert "strikes" in result.columns


def test_clean_data_remaps_pitch_types():
    """clean_data remaps SC→CU via data_remapping."""
    df = _make_clean_data_df(["SC", "FF"])
    result = clean_data(df)
    assert "SC" not in result["pitch_type"].values
    assert "CU" in result["pitch_type"].values


def test_clean_data_drops_unused_columns():
    """clean_data removes drop-list columns (player_name, sv_id)."""
    df = _make_clean_data_df(["FF"])
    result = clean_data(df)
    assert "player_name" not in result.columns
    assert "sv_id" not in result.columns


def test_clean_data_filters_non_regular_games():
    """clean_data removes non-regular-season rows via universal_features."""
    df = _make_clean_data_df(["FF"])
    df["game_type"] = "S"  # spring training
    result = clean_data(df)
    assert len(result) == 0


def test_clean_data_adds_pa_id():
    """clean_data produces a pa_id column formatted as '{game_pk}_{at_bat_number}'."""
    df = _make_clean_data_df(["FF"])
    result = clean_data(df)
    assert "pa_id" in result.columns
    assert result.iloc[0]["pa_id"] == "1_1"
