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
            features[feature] = row[0] if row else None # gets the only thing that should be in the tuple because we are requesting for one column from the table
        # Debug: write out retrieved feature values to a dated log file.
        _log_player_feature_retrieval(player_id, feature_names, entity, features)
        return features

def _log_player_feature_retrieval(player_id, feature_names, entity, features):
    debug_dir = Path("player-feature-retrieval-debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    debug_path = debug_dir / f"player_features_{entity}_{stamp}.log"
    with debug_path.open("a", encoding="utf-8") as handle:
        handle.write(f"player_ID={player_id}\n")
        handle.write(f"requested_features={feature_names}\n")
        handle.write(f"retrieved_features={features}\n")
        handle.write("---\n")