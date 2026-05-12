"""Parquet-backed accessor for the support feature tables that used to be
queried live from Postgres in ``model_server.src.util.feature_db_accessor``.

Each season-aggregated source table (``batted_ball_profile``,
``plate_discipline``, ``quality_of_contact``, ``pitch_tracking``,
``zone_metrics``, ``players``) is bootstrapped to its own parquet under
``data/`` by ``model_shared.setup``. These accessors return dicts with the
exact same shape the SQL versions did, so callers (the inference helpers)
can swap the import without changing their downstream rename/normalization
logic.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"

_PITCH_TRACKING_RENAMES: Dict[str, str] = {
    "whiff_percentage": "pitch_whiff_percentage",
    "launch_angle": "average_launch_angle",
    "exit_velocity": "average_exit_velocity",
    "mph": "average_mph",
}


@lru_cache(maxsize=8)
def _load_table(name: str) -> pd.DataFrame:
    path = _DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature table parquet at {path}. "
            "Run model_shared/setup.py to bootstrap the support tables."
        )
    df = pd.read_parquet(path).copy()
    if "player_id" in df.columns:
        df["_player_id_str"] = df["player_id"].astype("Int64").astype(str)
    if "id" in df.columns and "player_id" not in df.columns:
        df["_player_id_str"] = df["id"].astype("Int64").astype(str)
    return df


def _row_or_empty(
    df: pd.DataFrame,
    player_id: str,
    *,
    year: Optional[int] = None,
    position: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    pid = str(player_id)
    mask = df["_player_id_str"] == pid
    if year is not None and "year" in df.columns:
        mask &= df["year"] == year
    if position is not None and "position" in df.columns:
        mask &= df["position"] == position
    if extra:
        for col, val in extra.items():
            mask &= df[col] == val
    return df[mask]


def _series_to_dict(df: pd.DataFrame, *, drop: tuple[str, ...] = ()) -> Dict[str, Any]:
    if df.empty:
        return {}
    row = df.iloc[0]
    out: Dict[str, Any] = {}
    for col in df.columns:
        if col in drop or col.startswith("_"):
            continue
        value = row[col]
        if isinstance(value, float) and np.isnan(value):
            out[col] = None
        elif pd.isna(value):
            out[col] = None
        else:
            out[col] = value
    return out


def _rename_pitch_tracking_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {_PITCH_TRACKING_RENAMES.get(k, k): v for k, v in d.items()}


@lru_cache(maxsize=1024)
def fetch_player_out_type_historical_features(
    player_id: str,
    year: int,
    pitch_type: str,
    *,
    is_batter: bool,
) -> Dict[str, Optional[float]]:
    position = "B" if is_batter else "P"

    bb = _row_or_empty(_load_table("batted_ball_profile"), player_id, year=year, position=position)
    pd_tbl = _row_or_empty(_load_table("plate_discipline"), player_id, year=year, position=position)
    qoc = _row_or_empty(_load_table("quality_of_contact"), player_id, year=year, position=position)
    pt = _row_or_empty(
        _load_table("pitch_tracking"),
        player_id,
        year=year,
        position=position,
        extra={"pitch_type": pitch_type},
    )
    players = _row_or_empty(_load_table("players"), player_id)

    if bb.empty or pd_tbl.empty or qoc.empty or pt.empty or players.empty:
        return {}

    out: Dict[str, Any] = {}
    out.update(_series_to_dict(bb, drop=("player_id", "position", "year")))
    out.update(_series_to_dict(pd_tbl, drop=("player_id", "position", "year")))
    out.update(_series_to_dict(qoc, drop=("player_id", "position", "year")))
    out.update(
        _rename_pitch_tracking_keys(
            _series_to_dict(pt, drop=("player_id", "position", "year"))
        )
    )
    out.update(_series_to_dict(players, drop=("id",)))
    return out


@lru_cache(maxsize=1024)
def fetch_player_transition_historical_features(
    player_id: str,
    year: int,
    pitch_type: str,
    *,
    is_batter: bool,
) -> Dict[str, Optional[float]]:
    position = "B" if is_batter else "P"

    bb = _row_or_empty(_load_table("batted_ball_profile"), player_id, year=year, position=position)
    pd_tbl = _row_or_empty(_load_table("plate_discipline"), player_id, year=year, position=position)
    pt = _row_or_empty(
        _load_table("pitch_tracking"),
        player_id,
        year=year,
        position=position,
        extra={"pitch_type": pitch_type},
    )
    players = _row_or_empty(_load_table("players"), player_id)

    if bb.empty or pd_tbl.empty or pt.empty or players.empty:
        return {}

    out: Dict[str, Any] = {}
    out.update(_series_to_dict(bb, drop=("player_id", "position", "year")))
    out.update(_series_to_dict(pd_tbl, drop=("player_id", "position", "year")))
    out.update(
        _rename_pitch_tracking_keys(
            _series_to_dict(pt, drop=("player_id", "position", "year"))
        )
    )
    out.update(_series_to_dict(players, drop=("id",)))
    return out


@lru_cache(maxsize=1024)
def fetch_player_location_features(
    player_id: str,
    year: int,
    *,
    is_batter: bool,
    metrics: Tuple[str, ...],
) -> Dict[str, Optional[float]]:
    position = "B" if is_batter else "P"
    df = _load_table("location_metrics")
    rows = df[
        (df["_player_id_str"] == str(player_id))
        & (df["year"] == year)
        & (df["position"] == position)
        & (df["metric"].isin(metrics))
    ]
    if rows.empty:
        return {}

    loc_cols = [c for c in rows.columns if c.startswith("loc")]
    out: Dict[str, Any] = {}
    for _, row in rows.iterrows():
        metric = row["metric"]
        for lc in loc_cols:
            value = row[lc]
            if pd.isna(value):
                out[f"{metric}_{lc}"] = None
            else:
                out[f"{metric}_{lc}"] = value
    return out
