from __future__ import annotations

from typing import Dict, Protocol

import pandas as pd

from model_shared.db import get_read_cursor
from model_shared.feature_engineering.feature_calculator import (
    batter_situation_lookup,
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

_HISTORY_COLS = ["balls", "strikes", "pitch_type", "description"]


class FeatureStore(Protocol):
    def get_pitcher_situation_splits(
        self, pitcher_id: str, count_situation: str
    ) -> Dict[str, float]: ...

    def get_batter_situation_splits(
        self, batter_id: str, count_situation: str
    ) -> Dict[str, float]: ...


class SqlHistoricalPitchesFeatureStore:
    """Compute situational splits live from ``historical_pitches``.

    Loads one player's pitch history into a DataFrame and delegates to
    ``pitcher_situation_lookup`` / ``batter_situation_lookup`` in
    ``feature_calculator.py`` — the same functions training uses — so
    inference-time rates are guaranteed to match training-time rates.
    """

    def get_pitcher_situation_splits(
        self, pitcher_id: str, count_situation: str
    ) -> Dict[str, float]:
        df = self._load_player_history("pitcher", pitcher_id)
        if df.empty:
            return {col: 0.0 for col in _PITCHER_SIT_COLS}
        lookup = pitcher_situation_lookup(df)
        match = lookup[lookup["count_situation"] == count_situation]
        if match.empty:
            return {col: 0.0 for col in _PITCHER_SIT_COLS}
        row = match.iloc[0]
        return {col: float(row[col]) for col in _PITCHER_SIT_COLS}

    def get_batter_situation_splits(
        self, batter_id: str, count_situation: str
    ) -> Dict[str, float]:
        df = self._load_player_history("batter", batter_id)
        if df.empty:
            return {col: 0.0 for col in _BATTER_SIT_COLS}
        lookup = batter_situation_lookup(df)
        match = lookup[lookup["count_situation"] == count_situation]
        if match.empty:
            return {col: 0.0 for col in _BATTER_SIT_COLS}
        row = match.iloc[0]
        return {col: float(row[col]) for col in _BATTER_SIT_COLS}

    @staticmethod
    def _load_player_history(role: str, player_id: str) -> pd.DataFrame:
        columns = [role, *_HISTORY_COLS]
        select_list = ", ".join(columns)
        query = (
            f"SELECT {select_list} FROM historical_pitches WHERE {role} = %s"
        )
        with get_read_cursor() as cursor:
            cursor.execute(query, (player_id,))
            rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=columns)
