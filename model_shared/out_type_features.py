from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


OUT_TYPE_HISTORICAL_PITCH_COLUMNS = [
    "events",
    "bb_type",
    "launch_speed",
    "launch_angle",
]

OUT_TYPE_FEATURE_COLUMNS = [
    "batter_prev_whiff_rate",
    "batter_prev_gb_rate",
    "batter_prev_fb_rate",
    "batter_prev_chase_rate",
    "batter_prev_weak_rate",
    "batter_prev_under_rate",
    "batter_prev_topped_rate",
    "batter_prev_flareburner_rate",
    "batter_prev_solid_rate",
    "batter_prev_barrel_rate",
    "batter_prev_barrels_per_pa",
    "batter_prev_looking_strike_rate",
    "batter_prev_zone_contact_rate",
    "pitcher_prev_fb_rate",
    "pitcher_prev_gb_rate",
    "pitcher_prev_whiff_rate",
    "pitcher_prev_chase_rate",
    "pitcher_prev_weak_rate",
    "pitcher_prev_under_rate",
    "pitcher_prev_topped_rate",
    "pitcher_prev_flareburner_rate",
    "pitcher_prev_solid_rate",
    "pitcher_prev_barrel_rate",
    "pitcher_prev_barrels_per_pa",
    "pitcher_pitch_putaway_rate",
    "batter_pitch_putaway_rate",
    "pitcher_pitch_whiff_rate",
    "batter_pitch_whiff_rate",
    "pitcher_pitch_average_launch_angle",
    "pitcher_pitch_average_exit_velocity",
    "pitcher_pitch_expected_batting_average",
    "batter_pitch_average_launch_angle",
    "batter_pitch_average_exit_velocity",
    "batter_pitch_expected_batting_average",
    "pitcher_pitch_batting_average",
    "batter_pitch_batting_average",
    "pitcher_zone_batting_average",
    "batter_zone_batting_average",
    "pitcher_zone_average_exit_velocity",
    "batter_zone_average_exit_velocity",
    "pitcher_zone_average_launch_angle",
    "batter_zone_average_launch_angle",
    "pitcher_zone_contact_batting_average",
    "batter_zone_contact_batting_average",
    "pitcher_zone_hard_hit_bip_percentage",
    "batter_zone_hard_hit_bip_percentage",
    "pitcher_zone_expected_batting_average",
    "batter_zone_expected_batting_average",
    "batter_zone_strikeout_percentage",
    "pitcher_zone_strikeout_percentage",
    "batter_zone_whiff_percentage",
    "pitcher_zone_whiff_percentage",
    "batter_zone_fly_ball_rate",
    "pitcher_zone_fly_ball_rate",
    "batter_zone_walk_percentage",
    "pitcher_zone_walk_percentage",
    "batter_zone_ground_ball_percentage",
    "pitcher_zone_ground_ball_percentage",
    "batter_zone_swing_percentage",
    "pitcher_zone_swing_percentage",
    "pitcher_pitch_average_mph",
    "batter_pitch_average_mph",
    "batter_prev_first_pitch_swing_rate",
    "batter_prev_meatball_swing_rate",
    "pitcher_prev_first_pitch_swing_rate",
    "pitcher_prev_meatball_swing_rate",
]

_PREV_COLS = [
    "player_id",
    "position",
    "ground_ball_percentage",
    "air_ball_percentage",
    "whiff_percentage",
    "chase_percentage",
    "weak_percentage",
    "under_percentage",
    "topped_percentage",
    "zone_contact_percentage",
    "zone_percentage",
    "zone_swing_percentage",
    "first_pitch_swing_percentage",
    "meatball_swing_percentage",
    "flareburner_percentage",
    "solid_percentage",
    "barrel_percentage",
    "barrels_per_pa",
]

_PITCH_TRACKING_COLS = [
    "player_id",
    "position",
    "pitch_type",
    "pitch_count",
    "strikeouts",
    "batted_ball_events",
    "batting_average",
    "putaway_percentage",
    "pitch_whiff_percentage",
    "average_launch_angle",
    "average_exit_velocity",
    "expected_batting_average",
    "average_mph",
]

_SQL_PCT_COLS = [
    "batter_prev_fb_rate",
    "batter_prev_gb_rate",
    "batter_prev_whiff_rate",
    "batter_prev_chase_rate",
    "batter_prev_weak_rate",
    "batter_prev_under_rate",
    "batter_prev_topped_rate",
    "batter_prev_flareburner_rate",
    "batter_prev_solid_rate",
    "batter_prev_barrel_rate",
    "batter_prev_barrels_per_pa",
    "batter_prev_looking_strike_rate",
    "batter_prev_zone_contact_rate",
    "batter_pitch_putaway_rate",
    "batter_pitch_whiff_rate",
    "pitcher_prev_fb_rate",
    "pitcher_prev_gb_rate",
    "pitcher_prev_whiff_rate",
    "pitcher_prev_chase_rate",
    "pitcher_prev_weak_rate",
    "pitcher_prev_under_rate",
    "pitcher_prev_topped_rate",
    "pitcher_prev_flareburner_rate",
    "pitcher_prev_solid_rate",
    "pitcher_prev_barrel_rate",
    "pitcher_prev_barrels_per_pa",
    "pitcher_pitch_putaway_rate",
    "pitcher_pitch_whiff_rate",
    "batter_prev_first_pitch_swing_rate",
    "batter_prev_meatball_swing_rate",
    "pitcher_prev_first_pitch_swing_rate",
    "pitcher_prev_meatball_swing_rate",
]


def with_out_type_historical_pitch_columns(features: Iterable[str]) -> list[str]:
    combined = list(dict.fromkeys([*features, *OUT_TYPE_HISTORICAL_PITCH_COLUMNS]))
    return combined


def enrich_with_out_type_features(pitches: pd.DataFrame) -> pd.DataFrame:
    if pitches.empty or "game_year" not in pitches.columns:
        return pitches

    enriched_years = []
    for data_year in sorted(pitches["game_year"].dropna().unique()):
        current_pitch_data = pitches[pitches["game_year"] == data_year].copy()
        historical_year = int(data_year) - 1
        historical_df = query_out_type_historical_data(historical_year)
        zone_metrics_df = query_out_type_zone_metrics(historical_year)

        if historical_df.empty:
            enriched_years.append(current_pitch_data)
            continue

        enriched_years.append(
            build_out_type_season_features(
                current_pitch_data,
                historical_df,
                zone_metrics_df,
            )
        )

    return pd.concat(enriched_years, ignore_index=True)


def query_out_type_historical_data(year: int) -> pd.DataFrame:
    query = """
        SELECT
            bb.player_id,
            bb.position,
            bb.ground_ball_percentage,
            bb.air_ball_percentage,
            pd.whiff_percentage,
            pd.chase_percentage,
            pd.zone_percentage,
            pd.zone_swing_percentage,
            pd.zone_contact_percentage,
            pd.first_pitch_swing_percentage,
            pd.meatball_swing_percentage,
            qoc.weak_percentage,
            qoc.under_percentage,
            qoc.topped_percentage,
            qoc.flareburner_percentage,
            qoc.solid_percentage,
            qoc.barrel_percentage,
            qoc.barrels_per_pa,
            pt.pitch_type,
            pt.pitch_count,
            pt.strikeouts,
            pt.batted_ball_events,
            pt.batting_average,
            pt.putaway_percentage,
            pt.whiff_percentage AS pitch_whiff_percentage,
            pt.launch_angle AS average_launch_angle,
            pt.exit_velocity AS average_exit_velocity,
            pt.expected_batting_average,
            pt.mph AS average_mph
        FROM batted_ball_profile bb
        JOIN plate_discipline pd
            ON bb.player_id = pd.player_id
            AND bb.position = pd.position
            AND bb.year = pd.year
        JOIN quality_of_contact qoc
            ON bb.player_id = qoc.player_id
            AND bb.position = qoc.position
            AND bb.year = qoc.year
        JOIN pitch_tracking pt
            ON bb.player_id = pt.player_id
            AND bb.position = pt.position
            AND bb.year = pt.year
        WHERE bb.year = %s;
    """
    return _query_dataframe(query, (year,))


def query_out_type_zone_metrics(year: int) -> pd.DataFrame:
    query = """
        SELECT
            player_id,
            position,
            metric,
            zone1,
            zone2,
            zone3,
            zone4,
            zone5,
            zone6,
            zone7,
            zone8,
            zone9,
            zone11,
            zone12,
            zone13,
            zone14
        FROM zone_metrics
        WHERE year = %s
            AND metric IN (
                'batting_average',
                'average_exit_velocity',
                'average_launch_angle',
                'contact_batting_average',
                'hard_hit_bip_percentage',
                'expected_batting_average',
                'strikeout_percentage',
                'whiff_percentage',
                'walk_percentage',
                'ground_ball_percentage',
                'line_drive_percentage',
                'fly_ball_percentage',
                'popup_percentage',
                'swing_percentage'
            );
    """
    df = _query_dataframe(query, (year,))
    if df.empty:
        return df

    df_long = pd.melt(
        df,
        id_vars=["player_id", "position", "metric"],
        var_name="zone",
        value_name="metric_value",
    )
    df_long["zone"] = df_long["zone"].str.replace("zone", "").astype(int)
    return (
        df_long.pivot_table(
            index=["player_id", "position", "zone"],
            columns="metric",
            values="metric_value",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )


def build_out_type_season_features(
    current_pitch_data: pd.DataFrame,
    historical_df: pd.DataFrame,
    zone_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    pitch_data = current_pitch_data.copy()

    batter_baselines = (
        historical_df[historical_df["position"] == "B"][_PREV_COLS]
        .drop_duplicates()
        .add_prefix("batter_prev_")
    )
    pitcher_baselines = (
        historical_df[historical_df["position"] == "P"][_PREV_COLS]
        .drop_duplicates()
        .add_prefix("pitcher_prev_")
    )

    pitch_data = pitch_data.merge(
        batter_baselines,
        left_on="batter",
        right_on="batter_prev_player_id",
        how="left",
    )
    pitch_data = pitch_data.merge(
        pitcher_baselines,
        left_on="pitcher",
        right_on="pitcher_prev_player_id",
        how="left",
    )

    batter_pitch = (
        historical_df[historical_df["position"] == "B"][_PITCH_TRACKING_COLS]
        .drop_duplicates()
        .add_prefix("batter_pitch_")
    )
    pitcher_pitch = (
        historical_df[historical_df["position"] == "P"][_PITCH_TRACKING_COLS]
        .drop_duplicates()
        .add_prefix("pitcher_pitch_")
    )

    pitch_data = pitch_data.merge(
        batter_pitch,
        left_on=["batter", "pitch_type"],
        right_on=["batter_pitch_player_id", "batter_pitch_pitch_type"],
        how="left",
    )
    pitch_data = pitch_data.merge(
        pitcher_pitch,
        left_on=["pitcher", "pitch_type"],
        right_on=["pitcher_pitch_player_id", "pitcher_pitch_pitch_type"],
        how="left",
    )

    if not zone_metrics_df.empty:
        batter_zone = zone_metrics_df[zone_metrics_df["position"] == "B"].add_prefix("batter_zone_")
        pitcher_zone = zone_metrics_df[zone_metrics_df["position"] == "P"].add_prefix("pitcher_zone_")

        pitch_data = pitch_data.merge(
            batter_zone,
            left_on=["batter", "zone"],
            right_on=["batter_zone_player_id", "batter_zone_zone"],
            how="left",
        )
        pitch_data = pitch_data.merge(
            pitcher_zone,
            left_on=["pitcher", "zone"],
            right_on=["pitcher_zone_player_id", "pitcher_zone_zone"],
            how="left",
        )

    pitch_data = pitch_data.rename(columns=_out_type_renames())

    numeric_cols = [
        "batter_prev_zone_percentage",
        "batter_prev_zone_swing_percentage",
        "batter_zone_line_drive_percentage",
        "batter_zone_fly_ball_percentage",
        "batter_zone_popup_percentage",
        "pitcher_zone_line_drive_percentage",
        "pitcher_zone_fly_ball_percentage",
        "pitcher_zone_popup_percentage",
        *_SQL_PCT_COLS,
    ]
    for col in numeric_cols:
        if col in pitch_data.columns:
            pitch_data[col] = pd.to_numeric(pitch_data[col], errors="coerce")

    pitch_data["batter_prev_looking_strike_rate"] = (
        pitch_data["batter_prev_zone_percentage"]
        * (1 - pitch_data["batter_prev_zone_swing_percentage"] / 100.0)
    )

    for prefix in ("batter", "pitcher"):
        line_drive_col = f"{prefix}_zone_line_drive_percentage"
        fly_ball_col = f"{prefix}_zone_fly_ball_percentage"
        popup_col = f"{prefix}_zone_popup_percentage"
        target_col = f"{prefix}_zone_fly_ball_rate"
        if all(col in pitch_data.columns for col in (line_drive_col, fly_ball_col, popup_col)):
            pitch_data[target_col] = (
                pitch_data[line_drive_col]
                + pitch_data[fly_ball_col]
                + pitch_data[popup_col]
            )

    pitch_data = pitch_data.drop(
        columns=[
            "batter_prev_zone_swing_percentage",
            "batter_prev_zone_percentage",
            "pitcher_prev_zone_swing_percentage",
            "pitcher_prev_zone_percentage",
            "pitcher_prev_zone_contact_percentage",
        ],
        errors="ignore",
    )

    for col in _SQL_PCT_COLS:
        if col in pitch_data.columns:
            pitch_data[col] = pitch_data[col] / 100.0

    return pitch_data


def _query_dataframe(query: str, params: tuple) -> pd.DataFrame:
    from .db import get_read_cursor

    with get_read_cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _out_type_renames() -> Dict[str, str]:
    return {
        "batter_prev_ground_ball_percentage": "batter_prev_gb_rate",
        "batter_prev_air_ball_percentage": "batter_prev_fb_rate",
        "batter_prev_whiff_percentage": "batter_prev_whiff_rate",
        "batter_prev_chase_percentage": "batter_prev_chase_rate",
        "batter_prev_weak_percentage": "batter_prev_weak_rate",
        "batter_prev_under_percentage": "batter_prev_under_rate",
        "batter_prev_topped_percentage": "batter_prev_topped_rate",
        "batter_prev_flareburner_percentage": "batter_prev_flareburner_rate",
        "batter_prev_solid_percentage": "batter_prev_solid_rate",
        "batter_prev_barrel_percentage": "batter_prev_barrel_rate",
        "batter_prev_barrels_per_pa": "batter_prev_barrels_per_pa",
        "batter_prev_zone_contact_percentage": "batter_prev_zone_contact_rate",
        "batter_prev_first_pitch_swing_percentage": "batter_prev_first_pitch_swing_rate",
        "batter_prev_meatball_swing_percentage": "batter_prev_meatball_swing_rate",
        "batter_pitch_putaway_percentage": "batter_pitch_putaway_rate",
        "batter_pitch_pitch_whiff_percentage": "batter_pitch_whiff_rate",
        "pitcher_prev_ground_ball_percentage": "pitcher_prev_gb_rate",
        "pitcher_prev_air_ball_percentage": "pitcher_prev_fb_rate",
        "pitcher_prev_whiff_percentage": "pitcher_prev_whiff_rate",
        "pitcher_prev_chase_percentage": "pitcher_prev_chase_rate",
        "pitcher_prev_weak_percentage": "pitcher_prev_weak_rate",
        "pitcher_prev_under_percentage": "pitcher_prev_under_rate",
        "pitcher_prev_topped_percentage": "pitcher_prev_topped_rate",
        "pitcher_prev_flareburner_percentage": "pitcher_prev_flareburner_rate",
        "pitcher_prev_solid_percentage": "pitcher_prev_solid_rate",
        "pitcher_prev_barrel_percentage": "pitcher_prev_barrel_rate",
        "pitcher_prev_barrels_per_pa": "pitcher_prev_barrels_per_pa",
        "pitcher_prev_first_pitch_swing_percentage": "pitcher_prev_first_pitch_swing_rate",
        "pitcher_prev_meatball_swing_percentage": "pitcher_prev_meatball_swing_rate",
        "pitcher_pitch_putaway_percentage": "pitcher_pitch_putaway_rate",
        "pitcher_pitch_pitch_whiff_percentage": "pitcher_pitch_whiff_rate",
    }
