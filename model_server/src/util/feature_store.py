"""
Class that includes an interface (I think they call it a template here) 
that allows us to change which data source we want to query for our historical 
data.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Protocol

import pandas as pd

from model_shared.feature_engineering.feature_calculator import (
    batter_situation_lookup,
    pitcher_family_lookup,
    pitcher_situation_lookup,
)


_PITCHER_SIT_COLS = [
    "pitcher_sit_fb_rate",
    "pitcher_sit_br_rate",
    "pitcher_sit_os_rate",
    "pitcher_sit_whiff_rate",
]
_BATTER_SIT_COLS = [
    "batter_sit_swing_rate",
    "batter_sit_whiff_rate",
]
_PITCHER_FAMILY_COLS = [
    "pitcher_family_fb_rate",
    "pitcher_family_br_rate",
    "pitcher_family_os_rate",
]

_HISTORY_COLS = ["balls", "strikes", "pitch_type", "description"]

_HISTORICAL_PITCHES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "historical_pitches.parquet"
)


class FeatureStore(Protocol):
    def get_pitcher_situation_splits(
        self, pitcher_id: str, count_situation: str
    ) -> Dict[str, float]: ...

    def get_batter_situation_splits(
        self, batter_id: str, count_situation: str
    ) -> Dict[str, float]: ...

    def get_pitcher_family_splits(
        self, pitcher_id: str
    ) -> Dict[str, float]: ...


class _BaseHistoricalPitchesFeatureStore:
    """Shared situation-split logic. Subclasses only need to provide
    ``_load_player_history`` — everything else (the lookup wiring and the
    empty-frame zero default) lives here so SQL and parquet variants stay
    in lockstep.
    """

    @lru_cache(maxsize=1024)
    def _get_pitcher_lookup(self, pitcher_id: str) -> pd.DataFrame:
        df = self._load_player_history("pitcher", pitcher_id)
        if df.empty:
            return pd.DataFrame()
        return pitcher_situation_lookup(df)

    @lru_cache(maxsize=1024)
    def _get_batter_lookup(self, batter_id: str) -> pd.DataFrame:
        df = self._load_player_history("batter", batter_id)
        if df.empty:
            return pd.DataFrame()
        return batter_situation_lookup(df)

    def get_pitcher_situation_splits(
        self, pitcher_id: str, count_situation: str
    ) -> Dict[str, float]:
        lookup = self._get_pitcher_lookup(pitcher_id)
        if lookup.empty:
            return {col: 0.0 for col in _PITCHER_SIT_COLS}
        match = lookup[lookup["count_situation"] == count_situation]
        if match.empty:
            return {col: 0.0 for col in _PITCHER_SIT_COLS}
        row = match.iloc[0]
        return {col: float(row[col]) for col in _PITCHER_SIT_COLS}

    def get_batter_situation_splits(
        self, batter_id: str, count_situation: str
    ) -> Dict[str, float]:
        lookup = self._get_batter_lookup(batter_id)
        if lookup.empty:
            return {col: 0.0 for col in _BATTER_SIT_COLS}
        match = lookup[lookup["count_situation"] == count_situation]
        if match.empty:
            return {col: 0.0 for col in _BATTER_SIT_COLS}
        row = match.iloc[0]
        return {col: float(row[col]) for col in _BATTER_SIT_COLS}

    def get_pitcher_family_splits(self, pitcher_id: str) -> Dict[str, float]:
        df = self._load_player_history("pitcher", pitcher_id)
        if df.empty:
            return {col: 0.0 for col in _PITCHER_FAMILY_COLS}
        lookup = pitcher_family_lookup(df)
        if lookup.empty:
            return {col: 0.0 for col in _PITCHER_FAMILY_COLS}
        row = lookup.iloc[0]
        return {col: float(row[col]) for col in _PITCHER_FAMILY_COLS}

    def _load_player_history(self, role: str, player_id: str) -> pd.DataFrame:
        raise NotImplementedError


@lru_cache(maxsize=1)
def _load_historical_pitches() -> pd.DataFrame:
    if not _HISTORICAL_PITCHES_PATH.exists():
        raise FileNotFoundError(
            f"No cached historical pitches parquet found at {_HISTORICAL_PITCHES_PATH}"
        )

    df = pd.read_parquet(
        _HISTORICAL_PITCHES_PATH,
        columns=["batter", "pitcher", *_HISTORY_COLS],
    )
    df = df.copy()
    df["_batter_id"] = df["batter"].astype("Int64").astype(str)
    df["_pitcher_id"] = df["pitcher"].astype("Int64").astype(str)
    return df


class ParquetHistoricalPitchesFeatureStore(_BaseHistoricalPitchesFeatureStore):
    """Compute situational splits from ``data/historical_pitches.parquet``.

    The parquet is read once (lru_cache) and reused across requests, so
    per-request work is just a hash-equality filter on the cached frame —
    no DB round-trip.
    """

    @lru_cache(maxsize=1024)
    def _load_player_history(self, role: str, player_id: str) -> pd.DataFrame:
        df = _load_historical_pitches()
        id_col = "_batter_id" if role == "batter" else "_pitcher_id"
        matched = df[df[id_col] == str(player_id)]
        return matched[[role, *_HISTORY_COLS]].reset_index(drop=True)
