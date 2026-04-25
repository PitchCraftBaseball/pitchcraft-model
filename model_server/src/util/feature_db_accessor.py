from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from model_shared.db import find_table_for_column, get_read_cursor


def fetch_player_features(
    player_id: str,
    feature_names: List[str],
    *,
    entity: str,
    is_batter: bool,
) -> Dict[str, Optional[str]]:
    # Query across tables to find each feature for the given player_ID.

    with get_read_cursor() as cursor:
        features: Dict[str, Optional[str]] = {}
        for feature in feature_names:
            table = find_table_for_column("public", feature)
            if table is None:
                print(f"no table found for: {feature}")
                features[feature] = None
                continue
            id_column = "player_id"
            if table == "players":
                # TODO: Consider renaming players.id to player_id for consistency.
                id_column = "id"
            if table == "historical_pitches":
                # Use batter/pitcher IDs for historical pitch rows.
                id_column = "batter" if is_batter else "pitcher"
            cursor.execute(
                f"SELECT {feature} FROM {table} WHERE {id_column} = %s LIMIT 1",
                (player_id,),
            )
            row = cursor.fetchone()
            features[feature] = (
                row[0] if row else None
            )  # gets the only thing that should be in the tuple because we are requesting for one column from the table
        # Debug: write out retrieved feature values to a dated log file.
        _log_player_feature_retrieval(player_id, feature_names, entity, features)
        return features

def fetch_player_out_type_historical_features(
        player_id: str,
        year: int,
        pitch_type: str,
        entity: str,
        is_batter: bool
) -> Dict[str, Optional[str]]:
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
            pt.mph AS average_mph,
            p.sz_top,
            p.sz_bot
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
            AND pt.pitch_type = %s
        JOIN players p 
            ON bb.player_id = p.id
        WHERE bb.player_id = %s AND bb.year = %s AND bb.position = %s;
    """

    with get_read_cursor() as cursor:
        cursor.execute(query, (pitch_type, player_id, year, "B" if is_batter else "P"))
        row = cursor.fetchone()

        if not row: 
            return {}
        
        columns = [desc[0] for desc in cursor.description]
        features = dict(zip(columns, row))
        _log_player_feature_retrieval(player_id, columns, entity, features)
        return features

def fetch_player_out_type_zone_features(
        player_id: str,
        year: int,
        entity: str,
        is_batter: bool,
) -> Dict [str, Optional[str]]:
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
        WHERE player_id = %s AND year = %s AND position = %s
            AND metric IN ('batting_average', 'average_exit_velocity', 
                          'average_launch_angle', 'contact_batting_average',
                          'hard_hit_bip_percentage', 'expected_batting_average',
                          'strikeout_percentage', 'whiff_percentage', 'walk_percentage', 'ground_ball_percentage',
                          'line_drive_percentage', 'fly_ball_percentage', 'popup_percentage', 'swing_percentage'
                          )
    """
    
    with get_read_cursor() as cursor:
        cursor.execute(query, (player_id, year, "B" if is_batter else "P"))
        rows = cursor.fetchall()

        if not rows: 
            return {}
        
        columns = [desc[0] for desc in cursor.description]
        features = {}

        for row in rows:
            row_dict = dict(zip(columns, row))
            metric = row_dict.pop('metric')

            for key, val in row_dict.items():
                if key.startswith('zone'):
                    features[f"{metric}_{key}"] = val

        _log_player_feature_retrieval(player_id, list(features.keys()), entity, features)
        return features

def _log_player_feature_retrieval(player_id, feature_names, entity, features):
    debug_dir = Path(
        Path(__file__).resolve().parent.parent
        / "output"
        / "player-feature-retrieval-debug"
    )
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    debug_path = debug_dir / f"player_features_{entity}_{stamp}.log"
    with debug_path.open("a", encoding="utf-8") as handle:
        handle.write(f"player_ID={player_id}\n")
        handle.write(f"requested_features={feature_names}\n")
        handle.write(f"retrieved_features={features}\n")
        handle.write("---\n")
