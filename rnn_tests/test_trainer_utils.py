import pandas as pd
import pytest
import torch

from pitch_rnn.pitch_rnn_trainer import (
    apply_batter_lookup,
    apply_pitcher_family_lookup,
    apply_pitcher_lookup,
    build_arsenal_masks,
    build_balanced_sampler,
    split_by_year,
)
from model_shared.feature_engineering.pitch_constants import PAD_ID


def _make_year_df(year_pa_map: dict) -> pd.DataFrame:
    """Build a DataFrame with game_year and pa_id columns."""
    rows = []
    for year, pa_ids in year_pa_map.items():
        for pa in pa_ids:
            rows.append({"game_year": year, "pa_id": pa})
    return pd.DataFrame(rows)



def test_split_by_year_basic_split():
    """Rows with game_year < test_year go to train; == test_year go to test."""
    df = _make_year_df({2023: ["pa1", "pa2"], 2024: ["pa3"], 2025: ["pa4", "pa5"]})
    train_df, test_df, _, _ = split_by_year(df, test_year=2025)
    assert set(train_df["game_year"].unique()) == {2023, 2024}
    assert set(test_df["game_year"].unique()) == {2025}


def test_split_by_year_with_train_start_year():
    """train_start_year excludes rows older than the cutoff from the training set."""
    df = _make_year_df({2022: ["pa0"], 2023: ["pa1"], 2024: ["pa2"], 2025: ["pa3"]})
    train_df, _, _, _ = split_by_year(df, test_year=2025, train_start_year=2023)
    assert 2022 not in train_df["game_year"].values
    assert 2023 in train_df["game_year"].values
    assert 2024 in train_df["game_year"].values


def test_split_by_year_no_start_year_includes_all_prior():
    """Without train_start_year, all years before test_year appear in train."""
    df = _make_year_df({2020: ["pa0"], 2021: ["pa1"], 2025: ["pa2"]})
    train_df, _, _, _ = split_by_year(df, test_year=2025)
    assert {2020, 2021} == set(train_df["game_year"].unique())


def test_split_by_year_disjoint_pa_ids():
    """train_ids and test_ids share no plate appearance IDs."""
    df = _make_year_df({2023: ["pa1", "pa2"], 2025: ["pa3", "pa4"]})
    _, _, train_ids, test_ids = split_by_year(df, test_year=2025)
    assert train_ids.isdisjoint(test_ids)


def test_split_by_year_returns_pa_id_sets():
    """The returned id sets contain the correct pa_ids."""
    df = _make_year_df({2023: ["a", "b"], 2025: ["c"]})
    _, _, train_ids, test_ids = split_by_year(df, test_year=2025)
    assert train_ids == {"a", "b"}
    assert test_ids == {"c"}


def _pitcher_lookup_dfs():
    train_df = pd.DataFrame({
        "pitcher":             [100,   100,    200],
        "count_situation":     ["even", "ahead", "even"],
        "pitcher_sit_n":       [10,    5,      8],
        "pitcher_sit_fb_rate": [0.6,   0.4,    0.5],
        "pitcher_sit_br_rate": [0.3,   0.4,    0.3],
        "pitcher_sit_os_rate": [0.1,   0.2,    0.2],
        "pitcher_sit_whiff_rate": [0.2, 0.3,   0.25],
    })
    test_df = pd.DataFrame({
        "pitcher":         [100,    999],
        "count_situation": ["even", "even"],
    })
    return train_df, test_df


def test_apply_pitcher_lookup_merges_known_pitcher():
    train_df, test_df = _pitcher_lookup_dfs()
    result = apply_pitcher_lookup(train_df, test_df)
    row = result[result["pitcher"] == 100].iloc[0]
    assert row["pitcher_sit_n"] == 10
    assert row["pitcher_sit_fb_rate"] == pytest.approx(0.6)


def test_apply_pitcher_lookup_unknown_pitcher_gets_nan():
    train_df, test_df = _pitcher_lookup_dfs()
    result = apply_pitcher_lookup(train_df, test_df)
    row = result[result["pitcher"] == 999].iloc[0]
    assert pd.isna(row["pitcher_sit_n"])


def _batter_lookup_dfs():
    train_df = pd.DataFrame({
        "batter":               [1,      2],
        "count_situation":      ["even", "ahead"],
        "batter_sit_n":         [20,     15],
        "batter_sit_swing_rate":[0.45,   0.55],
        "batter_sit_whiff_rate":[0.10,   0.20],
    })
    test_df = pd.DataFrame({
        "batter":          [1,      999],
        "count_situation": ["even", "even"],
    })
    return train_df, test_df


def test_apply_batter_lookup_merges_known_batter():
    train_df, test_df = _batter_lookup_dfs()
    result = apply_batter_lookup(train_df, test_df)
    row = result[result["batter"] == 1].iloc[0]
    assert row["batter_sit_n"] == 20


def test_apply_batter_lookup_unknown_batter_gets_nan():
    train_df, test_df = _batter_lookup_dfs()
    result = apply_batter_lookup(train_df, test_df)
    row = result[result["batter"] == 999].iloc[0]
    assert pd.isna(row["batter_sit_n"])


def test_apply_pitcher_family_lookup_merges_known_pitcher():
    train_df = pd.DataFrame({
        "pitcher":                  [100,  200],
        "pitcher_family_fb_rate":   [0.6,  0.4],
        "pitcher_family_br_rate":   [0.3,  0.4],
        "pitcher_family_os_rate":   [0.1,  0.2],
    })
    test_df = pd.DataFrame({"pitcher": [100, 999]})
    result = apply_pitcher_family_lookup(train_df, test_df)
    known = result[result["pitcher"] == 100].iloc[0]
    assert known["pitcher_family_fb_rate"] == pytest.approx(0.6)


def test_apply_pitcher_family_lookup_unknown_pitcher_gets_nan():
    train_df = pd.DataFrame({
        "pitcher":                [100],
        "pitcher_family_fb_rate": [0.6],
        "pitcher_family_br_rate": [0.3],
        "pitcher_family_os_rate": [0.1],
    })
    test_df = pd.DataFrame({"pitcher": [999]})
    result = apply_pitcher_family_lookup(train_df, test_df)
    assert pd.isna(result.iloc[0]["pitcher_family_fb_rate"])


def test_build_balanced_sampler_rare_class_higher_weight():
    """Class 2 appears once; class 1 appears nine times → weight ratio is 9:1."""
    Y = torch.zeros(10, 4, dtype=torch.long)
    Y[:9, 0] = 1   # 9 sequences labelled class 1
    Y[9,  0] = 2   # 1 sequence labelled class 2
    sampler = build_balanced_sampler(Y)
    weights = list(sampler.weights)
    assert weights[9] == pytest.approx(weights[0] * 9, rel=1e-5)


def test_build_balanced_sampler_all_pad_sequence_gets_zero_weight():
    """Sequences whose every target is PAD receive weight 0."""
    Y = torch.zeros(3, 4, dtype=torch.long)  # all PAD
    Y[1, 0] = 1  # only sequence 1 has a real label
    sampler = build_balanced_sampler(Y)
    weights = list(sampler.weights)
    assert weights[0] == pytest.approx(0.0)
    assert weights[2] == pytest.approx(0.0)
    assert weights[1] > 0.0


def test_build_balanced_sampler_length():
    """Sampler num_samples equals the number of sequences."""
    Y = torch.ones(8, 4, dtype=torch.long)
    sampler = build_balanced_sampler(Y)
    assert sampler.num_samples == 8


def test_build_arsenal_masks_known_pitcher_limits_classes():
    """Allowed pitches are set to 1; all others are 0 for a known pitcher."""
    cat_vocabs = {"pitcher": {100: 1}}
    y_vocab    = {"FF": 1, "SL": 2, "CH": 3}
    arsenals   = {"100": {2025: {"arsenal_mask": ["FF", "SL"]}}}
    masks = build_arsenal_masks(arsenals, cat_vocabs, y_vocab, num_classes=4, year=2025)
    assert masks[1, 1].item() == 1.0   # FF allowed
    assert masks[1, 2].item() == 1.0   # SL allowed
    assert masks[1, 3].item() == 0.0   # CH blocked


def test_build_arsenal_masks_unknown_pitcher_stays_all_ones():
    """A pitcher not in the vocab dict is skipped; their row remains all-ones."""
    cat_vocabs = {"pitcher": {100: 1}}
    y_vocab    = {"FF": 1, "SL": 2}
    arsenals   = {"999": {2025: {"arsenal_mask": ["FF"]}}}  # 999 not in vocab
    masks = build_arsenal_masks(arsenals, cat_vocabs, y_vocab, num_classes=3, year=2025)
    assert (masks == 1.0).all()


def test_build_arsenal_masks_falls_back_to_prior_year():
    """When the exact year is absent, year-1 data is used."""
    cat_vocabs = {"pitcher": {100: 1}}
    y_vocab    = {"FF": 1, "SL": 2, "CH": 3}
    arsenals   = {"100": {"2024": {"arsenal_mask": ["CH"]}}}  # only 2024 available
    masks = build_arsenal_masks(arsenals, cat_vocabs, y_vocab, num_classes=4, year=2025)
    assert masks[1, 3].item() == 1.0   # CH allowed (from 2024 fallback)
    assert masks[1, 1].item() == 0.0   # FF blocked
    assert masks[1, 2].item() == 0.0   # SL blocked


def test_build_arsenal_masks_no_matching_year_stays_all_ones():
    """When neither the year nor year-1 exists, the pitcher row is left unchanged."""
    cat_vocabs = {"pitcher": {100: 1}}
    y_vocab    = {"FF": 1}
    arsenals   = {"100": {"2020": {"arsenal_mask": ["FF"]}}}  # too old
    masks = build_arsenal_masks(arsenals, cat_vocabs, y_vocab, num_classes=2, year=2025)
    assert (masks == 1.0).all()
