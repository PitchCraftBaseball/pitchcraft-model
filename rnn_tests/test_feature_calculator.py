import numpy as np
import pandas as pd
import pytest

from model_shared.feature_engineering.feature_calculator import (
    add_batter_count_split_features,
    add_pitcher_count_split_features,
    add_pitcher_family_rate_features,
    calculate_game_state_features,
    calculate_pitch_features,
    calculate_woba,
    count_situation,
    get_vs_pitcher_stats,
    pitch_to_family,
    pitcher_family_lookup,
    situational_split,
)


def _make_pitch_df():
    return pd.DataFrame({
        "description": ["swinging_strike", "ball", "called_strike", "hit_into_play"],
        "zone":        [5,                 11,     3,               14],
        "pitch_type":  ["FF",              "SL",   "CU",            "CH"],
    })


def _make_game_state_df():
    return pd.DataFrame({
        "on_1b":   [12345.0, None,    None],
        "on_2b":   [None,    67890.0, None],
        "on_3b":   [None,    None,    None],
        "balls":   [1,       2,       0],
        "strikes": [2,       0,       0],
    })


def _make_count_df(include_count_situation=True):
    df = pd.DataFrame({
        "pitcher":     [100, 100, 100, 200, 200],
        "batter":      [1,   1,   2,   3,   3],
        "balls":       [0,   1,   2,   0,   3],
        "strikes":     [0,   1,   0,   2,   0],
        "pitch_type":  ["FF", "SL", "FF", "CH", "CU"],
        "description": [
            "swinging_strike", "ball", "called_strike",
            "hit_into_play",   "ball",
        ],
        # Required by calculate_game_state_features when count_situation is absent
        "on_1b": [None, None, None, None, None],
        "on_2b": [None, None, None, None, None],
        "on_3b": [None, None, None, None, None],
    })
    if include_count_situation:
        df["count_situation"] = ["even", "even", "behind", "ahead", "behind"]
    return df


def _make_history_df():
    return pd.DataFrame({
        "batter":      [1,               1,      2,               2],
        "pitcher":     [10,              10,     10,              20],
        "balls":       [0,               1,      0,               2],
        "strikes":     [0,               2,      1,               0],
        "pitch_type":  ["FF",            "SL",   "FF",            "CH"],
        "description": ["swinging_strike", "ball", "called_strike", "hit_into_play"],
    })

def test_pitch_to_family_fastball():
    assert pitch_to_family("FF") == "fastball"

def test_pitch_to_family_breaking():
    assert pitch_to_family("SL") == "breaking"

def test_pitch_to_family_offspeed():
    assert pitch_to_family("CH") == "offspeed"

def test_pitch_to_family_nan_returns_none():
    assert pitch_to_family(np.nan) is None

def test_pitch_to_family_unknown_returns_none():
    assert pitch_to_family("XX") is None


def test_count_situation_pitcher_ahead():
    assert count_situation(0, 2) == "ahead"

def test_count_situation_pitcher_behind():
    assert count_situation(3, 1) == "behind"

def test_count_situation_even():
    assert count_situation(1, 1) == "even"


def test_calculate_pitch_features_adds_columns():
    result = calculate_pitch_features(_make_pitch_df())
    for col in ("is_swing", "is_whiff", "is_called_strike", "is_ball", "in_zone", "out_zone", "pitch_group"):
        assert col in result.columns, f"Missing column: {col}"


def test_calculate_pitch_features_is_swing():
    result = calculate_pitch_features(_make_pitch_df())
    assert result.loc[result["description"] == "swinging_strike", "is_swing"].all()
    assert not result.loc[result["description"] == "ball", "is_swing"].any()


def test_calculate_pitch_features_is_whiff():
    result = calculate_pitch_features(_make_pitch_df())
    assert result.loc[result["description"] == "swinging_strike", "is_whiff"].all()
    assert not result.loc[result["description"] == "called_strike", "is_whiff"].any()


def test_calculate_pitch_features_in_zone():
    result = calculate_pitch_features(_make_pitch_df())
    assert result.loc[result["zone"] == 5,  "in_zone"].all()
    assert result.loc[result["zone"] == 14, "out_zone"].all()


def test_calculate_pitch_features_pitch_group():
    result = calculate_pitch_features(_make_pitch_df())
    assert result.loc[result["pitch_type"] == "FF", "pitch_group"].iloc[0] == "fastball"
    assert result.loc[result["pitch_type"] == "SL", "pitch_group"].iloc[0] == "breaking"
    assert result.loc[result["pitch_type"] == "CH", "pitch_group"].iloc[0] == "offspeed"


def test_calculate_game_state_features_adds_columns():
    result = calculate_game_state_features(_make_game_state_df())
    for col in ("base_state", "count_state", "count_situation"):
        assert col in result.columns


def test_calculate_game_state_base_state_encoding():
    result = calculate_game_state_features(_make_game_state_df())
    assert result.loc[0, "base_state"] == 1   # on_1b only → bit 0 set
    assert result.loc[1, "base_state"] == 2   # on_2b only → bit 1 set
    assert result.loc[2, "base_state"] == 0   # bases empty


def test_calculate_game_state_count_state_format():
    result = calculate_game_state_features(_make_game_state_df())
    assert result.loc[0, "count_state"] == "1-2"


def test_calculate_game_state_count_situation():
    result = calculate_game_state_features(_make_game_state_df())
    assert result.loc[0, "count_situation"] == "ahead"    # 1 ball, 2 strikes → pitcher ahead
    assert result.loc[1, "count_situation"] == "behind"   # 2 balls, 0 strikes → pitcher behind
    assert result.loc[2, "count_situation"] == "even"     # 0-0


def test_add_pitcher_count_split_features_adds_columns():
    result = add_pitcher_count_split_features(_make_count_df())
    for col in ("pitcher_sit_n", "pitcher_sit_fb_rate", "pitcher_sit_br_rate",
                "pitcher_sit_os_rate", "pitcher_sit_whiff_rate"):
        assert col in result.columns


def test_add_pitcher_count_split_features_injects_count_situation():
    """Function adds count_situation automatically when the column is absent."""
    df = _make_count_df(include_count_situation=False)
    assert "count_situation" not in df.columns
    result = add_pitcher_count_split_features(df)
    assert "count_situation" in result.columns


def test_add_pitcher_count_split_features_skips_recalc_when_present():
    """When count_situation already exists, the if-branch is not entered."""
    df = _make_count_df(include_count_situation=True)
    original = df["count_situation"].tolist()
    result = add_pitcher_count_split_features(df)
    assert result["count_situation"].tolist() == original

def test_add_batter_count_split_features_adds_columns():
    result = add_batter_count_split_features(_make_count_df())
    for col in ("batter_sit_n", "batter_sit_swing_rate", "batter_sit_whiff_rate"):
        assert col in result.columns


def test_add_batter_count_split_features_injects_count_situation():
    """Function adds count_situation automatically when the column is absent."""
    df = _make_count_df(include_count_situation=False)
    assert "count_situation" not in df.columns
    result = add_batter_count_split_features(df)
    assert "count_situation" in result.columns

def test_pitcher_family_lookup_columns():
    df = pd.DataFrame({"pitcher": [100, 100, 100], "pitch_type": ["FF", "SL", "CH"]})
    result = pitcher_family_lookup(df)
    assert set(result.columns) == {
        "pitcher", "pitcher_family_fb_rate",
        "pitcher_family_br_rate", "pitcher_family_os_rate",
    }


def test_pitcher_family_lookup_rates():
    df = pd.DataFrame({
        "pitcher":    [100, 100, 100, 100],
        "pitch_type": ["FF", "FF", "SL", "CH"],  # 50% FB, 25% BR, 25% OS
    })
    result = pitcher_family_lookup(df)
    row = result[result["pitcher"] == 100].iloc[0]
    assert row["pitcher_family_fb_rate"] == pytest.approx(0.5)
    assert row["pitcher_family_br_rate"] == pytest.approx(0.25)
    assert row["pitcher_family_os_rate"] == pytest.approx(0.25)


def test_add_pitcher_family_rate_features_adds_columns():
    df = pd.DataFrame({"pitcher": [100, 100, 200], "pitch_type": ["FF", "SL", "CH"]})
    result = add_pitcher_family_rate_features(df)
    for col in ("pitcher_family_fb_rate", "pitcher_family_br_rate", "pitcher_family_os_rate"):
        assert col in result.columns


def test_add_pitcher_family_rate_features_replaces_stale_values():
    """Pre-existing family rate columns are dropped and recomputed, not duplicated."""
    df = pd.DataFrame({
        "pitcher":                [100, 100],
        "pitch_type":             ["FF", "SL"],
        "pitcher_family_fb_rate": [0.99, 0.99],  # stale
    })
    result = add_pitcher_family_rate_features(df)
    assert result["pitcher_family_fb_rate"].iloc[0] == pytest.approx(0.5)


def test_get_vs_pitcher_stats_no_history_returns_empty():
    result = get_vs_pitcher_stats(999, 888, _make_history_df())
    assert result.empty


def test_get_vs_pitcher_stats_returns_stats_row():
    # batter=1 vs pitcher=10: 2 pitches; swinging_strike → swing + whiff
    result = get_vs_pitcher_stats(1, 10, _make_history_df())
    assert not result.empty
    assert result["n_pitches"].iloc[0] == 2
    assert result["swing_rate"].iloc[0] == pytest.approx(0.5)
    assert result["whiff_rate"].iloc[0] == pytest.approx(0.5)


def test_get_vs_pitcher_stats_columns():
    result = get_vs_pitcher_stats(1, 10, _make_history_df())
    for col in ("n_pitches", "swing_rate", "whiff_rate", "called_k_rate"):
        assert col in result.columns


def test_situational_split_unknown_pitcher_returns_empty():
    result = situational_split(999, _make_history_df())
    assert result.empty


def test_situational_split_returns_rows_for_known_pitcher():
    result = situational_split(10, _make_history_df())
    assert not result.empty
    assert (result["pitcher"] == 10).all()


def test_situational_split_columns():
    result = situational_split(10, _make_history_df())
    for col in ("count_situation", "fastball_rate", "breaking_rate",
                "offspeed_rate", "swing_rate", "whiff_rate", "called_k_rate"):
        assert col in result.columns


def test_situational_split_pitcher_id_prepended():
    result = situational_split(10, _make_history_df())
    assert result.columns[0] == "pitcher"


# ===========================================================================
# calculate_woba
# ===========================================================================

def test_calculate_woba_returns_woba_column():
    df = pd.DataFrame({"batter": [1, 1], "events": ["single", "strikeout"]})
    result = calculate_woba(df)
    assert "wOBA" in result.columns


def test_calculate_woba_single_batter_value():
    df = pd.DataFrame({
        "batter": [1, 1, 1],
        "events": ["single", "home_run", "strikeout"],
    })
    result = calculate_woba(df)
    # AB=3, 1B=1, HR=1 → wOBA = (0.882 + 2.037) / 3
    expected = (0.882 + 2.037) / 3
    assert result.loc[1, "wOBA"] == pytest.approx(expected, rel=1e-4)


def test_calculate_woba_multiple_batters():
    df = pd.DataFrame({
        "batter": [1, 1, 2, 2],
        "events": ["single", "strikeout", "home_run", "walk"],
    })
    result = calculate_woba(df)
    assert 1 in result.index
    assert 2 in result.index
