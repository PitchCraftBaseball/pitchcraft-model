from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

# Half width of strikezone in feet (set 17 inches wide)
ZONE_HALF_WIDTH = 17 / 12 / 2

def assign_location_zone(plate_x, plate_z, sz_top, sz_bot) -> int | None:
    if pd.isna(plate_x) or pd.isna(plate_z) or pd.isna(sz_top) or pd.isna(sz_bot):
        return None

    is_left = plate_x < 0
    mid_z = (sz_top + sz_bot) / 2
    is_upper = plate_z >= mid_z
    
    in_zone = (
        -ZONE_HALF_WIDTH <= plate_x <= ZONE_HALF_WIDTH and # in strikezone horizontally
        sz_bot <= plate_z <= sz_top # in strikezone vertically
    )

    if is_upper:
        zone_base = 1 if is_left else 2
    else:
        zone_base = 3 if is_left else 4

    if in_zone:
        return zone_base
    else:
        return zone_base + 4

OUT_TYPE_HISTORICAL_PITCH_COLUMNS = [
    "events",
    "bb_type",
    "launch_speed",
    "launch_angle",
    "description",
]

OUT_TYPE_FEATURE_COLUMNS = [
    "batter_prev_whiff_percentage",
    "batter_prev_gb_percentage",
    "batter_prev_fb_percentage",
    "batter_prev_chase_percentage",
    "batter_prev_weak_percentage",
    "batter_prev_under_percentage",
    "batter_prev_topped_percentage",
    "batter_prev_flareburner_percentage",
    "batter_prev_solid_percentage",
    "batter_prev_barrel_percentage",
    "batter_prev_barrels_per_pa",
    "batter_prev_looking_strike_percentage",
    "batter_prev_zone_contact_percentage",
    "pitcher_prev_fb_percentage",
    "pitcher_prev_gb_percentage",
    "pitcher_prev_whiff_percentage",
    "pitcher_prev_chase_percentage",
    "pitcher_prev_weak_percentage",
    "pitcher_prev_under_percentage",
    "pitcher_prev_topped_percentage",
    "pitcher_prev_flareburner_percentage",
    "pitcher_prev_solid_percentage",
    "pitcher_prev_barrel_percentage",
    "pitcher_prev_barrels_per_pa",
    "pitcher_pitch_putaway_percentage",
    "batter_pitch_putaway_percentage",
    "pitcher_pitch_whiff_percentage",
    "batter_pitch_whiff_percentage",
    "pitcher_pitch_average_launch_angle",
    "pitcher_pitch_average_exit_velocity",
    "pitcher_pitch_expected_batting_average",
    "batter_pitch_average_launch_angle",
    "batter_pitch_average_exit_velocity",
    "batter_pitch_expected_batting_average",
    "pitcher_pitch_batting_average",
    "batter_pitch_batting_average",
    "pitcher_loc_batting_average",
    "batter_loc_batting_average",
    "pitcher_loc_average_exit_velocity",
    "batter_loc_average_exit_velocity",
    "pitcher_loc_average_launch_angle",
    "batter_loc_average_launch_angle",
    "pitcher_loc_contact_batting_average",
    "batter_loc_contact_batting_average",
    "pitcher_loc_hard_hit_bip_percentage",
    "batter_loc_hard_hit_bip_percentage",
    "batter_loc_strikeout_percentage",
    "pitcher_loc_strikeout_percentage",
    "batter_loc_whiff_percentage",
    "pitcher_loc_whiff_percentage",
    "batter_loc_fly_ball_percentage",
    "pitcher_loc_fly_ball_percentage",
    "batter_loc_walk_percentage",
    "pitcher_loc_walk_percentage",
    "batter_loc_ground_ball_percentage",
    "pitcher_loc_ground_ball_percentage",
    "batter_loc_swing_percentage",
    "pitcher_loc_swing_percentage",
    "batter_loc_foul_percentage",
    "pitcher_loc_foul_percentage",
    "pitcher_pitch_average_mph",
    "batter_pitch_average_mph",
    "batter_prev_first_pitch_swing_percentage",
    "batter_prev_meatball_swing_percentage",
    "pitcher_prev_first_pitch_swing_percentage",
    "pitcher_prev_meatball_swing_percentage",
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
        location_metrics_df = query_out_type_location_metrics(historical_year)

        if historical_df.empty:
            enriched_years.append(current_pitch_data)
            continue

        enriched_years.append(
            build_out_type_season_features(
                current_pitch_data,
                historical_df,
                location_metrics_df,
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


def query_out_type_location_metrics(year: int) -> pd.DataFrame:
    query = """
        SELECT
            player_id,
            position,
            metric,
            loc1,
            loc2,
            loc3,
            loc4,
            loc5,
            loc6,
            loc7,
            loc8
        FROM location_metrics
        WHERE year = %s
            AND metric IN (
                'batting_average',
                'average_exit_velocity',
                'average_launch_angle',
                'contact_batting_average',
                'hard_hit_bip_percentage',
                'strikeout_percentage',
                'whiff_percentage',
                'walk_percentage',
                'ground_ball_percentage',
                'line_drive_percentage',
                'fly_ball_percentage',
                'popup_percentage',
                'swing_percentage',
                'foul_percentage'
            );
    """
    df = _query_dataframe(query, (year,))
    if df.empty:
        return df

    df_long = pd.melt(
        df,
        id_vars=["player_id", "position", "metric"],
        var_name="location",
        value_name="metric_value",
    )
    df_long["location"] = df_long["location"].str.replace("loc", "").astype(int)
    return (
        df_long.pivot_table(
            index=["player_id", "position", "location"],
            columns="metric",
            values="metric_value",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )


def build_out_type_season_features(
    current_pitch_data: pd.DataFrame,
    historical_df: pd.DataFrame,
    location_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    pitch_data = current_pitch_data.copy()

    if "location" not in pitch_data.columns:
        if all(c in pitch_data.columns for c in ("plate_x", "plate_z", "sz_top", "sz_bot")):
            pitch_data["location"] = pitch_data.apply(
                lambda r: assign_location_zone(r["plate_x"], r["plate_z"], r["sz_top"], r["sz_bot"]),
                axis=1,
            )

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

    if not location_metrics_df.empty:
        batter_loc = location_metrics_df[location_metrics_df["position"] == "B"].add_prefix("batter_loc_")
        pitcher_loc = location_metrics_df[location_metrics_df["position"] == "P"].add_prefix("pitcher_loc_")

        pitch_data = pitch_data.merge(
            batter_loc,
            left_on=["batter", "location"],
            right_on=["batter_loc_player_id", "batter_loc_location"],
            how="left",
        )
        pitch_data = pitch_data.merge(
            pitcher_loc,
            left_on=["pitcher", "location"],
            right_on=["pitcher_loc_player_id", "pitcher_loc_location"],
            how="left",
        )

    pitch_data = pitch_data.rename(columns=_out_type_renames())

    for col in ["batter_prev_zone_percentage", "batter_prev_zone_swing_percentage"]:
        if col in pitch_data.columns:
            pitch_data[col] = pd.to_numeric(pitch_data[col], errors="coerce")

    pitch_data["batter_prev_looking_strike_percentage"] = (
        pitch_data["batter_prev_zone_percentage"]
        * (1 - pitch_data["batter_prev_zone_swing_percentage"] / 100.0)
    )

    for prefix in ("batter", "pitcher"):
        line_drive_col = f"{prefix}_loc_line_drive_percentage"
        fly_ball_col = f"{prefix}_loc_fly_ball_percentage"
        popup_col = f"{prefix}_loc_popup_percentage"
        target_col = f"{prefix}_loc_fly_ball_percentage"
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
        "batter_prev_ground_ball_percentage": "batter_prev_gb_percentage",
        "batter_prev_air_ball_percentage": "batter_prev_fb_percentage",
        "batter_prev_first_pitch_swing_percentage": "batter_prev_first_pitch_swing_percentage",
        "batter_prev_meatball_swing_percentage": "batter_prev_meatball_swing_percentage",
        "batter_pitch_pitch_whiff_percentage": "batter_pitch_whiff_percentage",
        "pitcher_prev_ground_ball_percentage": "pitcher_prev_gb_percentage",
        "pitcher_prev_air_ball_percentage": "pitcher_prev_fb_percentage",
        "pitcher_prev_first_pitch_swing_percentage": "pitcher_prev_first_pitch_swing_percentage",
        "pitcher_prev_meatball_swing_percentage": "pitcher_prev_meatball_swing_percentage",
        "pitcher_pitch_pitch_whiff_percentage": "pitcher_pitch_whiff_percentage",
        "batter_pitch_putaway_percentage": "batter_pitch_putaway_percentage",
        "pitcher_pitch_putaway_percentage": "pitcher_pitch_putaway_percentage",
    }
