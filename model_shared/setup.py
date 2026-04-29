"""
Setup script. Might call this in an entrypoint script to set up a historical cache for the inference model. TBD. - Dylan
"""
import logging
from pathlib import Path

import pandas as pd

from .db import query_historical_pitches_by_year, query_table_dataframe
from .out_type_features import (
    enrich_with_out_type_features,
    with_out_type_historical_pitch_columns,
)
from .parquet import save_training_data
from model_shared.feature_list import validate_feature_list_file

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data")

_SUPPORT_TABLE_START_YEAR = 2024
_SUPPORT_TABLE_END_YEAR = 2025

_ZONE_METRICS = (
    "batting_average",
    "average_exit_velocity",
    "average_launch_angle",
    "contact_batting_average",
    "hard_hit_bip_percentage",
    "expected_batting_average",
    "strikeout_percentage",
    "whiff_percentage",
    "walk_percentage",
    "ground_ball_percentage",
    "line_drive_percentage",
    "fly_ball_percentage",
    "popup_percentage",
    "swing_percentage",
)

_SUPPORT_TABLE_QUERIES: dict[str, tuple[str, tuple]] = {
    "batted_ball_profile": (
        """
        SELECT player_id, position, year,
               ground_ball_percentage, air_ball_percentage
        FROM batted_ball_profile
        WHERE year BETWEEN %s AND %s
        """,
        (_SUPPORT_TABLE_START_YEAR, _SUPPORT_TABLE_END_YEAR),
    ),
    "plate_discipline": (
        """
        SELECT player_id, position, year,
               whiff_percentage, chase_percentage,
               zone_percentage, zone_swing_percentage, zone_contact_percentage,
               first_pitch_swing_percentage, meatball_swing_percentage
        FROM plate_discipline
        WHERE year BETWEEN %s AND %s
        """,
        (_SUPPORT_TABLE_START_YEAR, _SUPPORT_TABLE_END_YEAR),
    ),
    "quality_of_contact": (
        """
        SELECT player_id, position, year,
               weak_percentage, under_percentage, topped_percentage,
               flareburner_percentage, solid_percentage,
               barrel_percentage, barrels_per_pa
        FROM quality_of_contact
        WHERE year BETWEEN %s AND %s
        """,
        (_SUPPORT_TABLE_START_YEAR, _SUPPORT_TABLE_END_YEAR),
    ),
    "pitch_tracking": (
        """
        SELECT player_id, position, year, pitch_type,
               pitch_count, strikeouts, batted_ball_events,
               batting_average, putaway_percentage, whiff_percentage,
               launch_angle, exit_velocity, expected_batting_average, mph
        FROM pitch_tracking
        WHERE year BETWEEN %s AND %s
        """,
        (_SUPPORT_TABLE_START_YEAR, _SUPPORT_TABLE_END_YEAR),
    ),
    "zone_metrics": (
        """
        SELECT player_id, position, year, metric,
               zone1, zone2, zone3, zone4, zone5,
               zone6, zone7, zone8, zone9,
               zone11, zone12, zone13, zone14
        FROM zone_metrics
        WHERE year BETWEEN %s AND %s
          AND metric = ANY(%s::zone_metric_type[])
        """,
        (_SUPPORT_TABLE_START_YEAR, _SUPPORT_TABLE_END_YEAR, list(_ZONE_METRICS)),
    ),
    "players": (
        """
        SELECT id, sz_top, sz_bot
        FROM players
        """,
        (),
    ),
}


def _bootstrap_support_tables() -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    for name, (query, params) in _SUPPORT_TABLE_QUERIES.items():
        df = query_table_dataframe(query, params)
        path = _DATA_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        print(f"Saved {len(df):,} rows to {path}")


if __name__ == "__main__":
    logger.info("Running setup script")
    feature_list_dict = validate_feature_list_file("/home/shakotan/git-linux/pitchcraft-repos/pitchcraft-model/feature_list")
    if not feature_list_dict:
        raise ValueError("Feature list validation failed")

    features = feature_list_dict.get("historical_pitches")
    if not features:
        raise ValueError("No features found for historical_pitches")

    features = with_out_type_historical_pitch_columns(features)
    df = query_historical_pitches_by_year("historical_pitches", features, start_year=2024, end_year=2025)
    df = enrich_with_out_type_features(df)

    # Ensure stable parquet typing for date-like values returned as Python objects.
    if "game_date" in df.columns:
        df["game_date"] = df["game_date"].astype("string")

    save_training_data(df)

    _bootstrap_support_tables()
