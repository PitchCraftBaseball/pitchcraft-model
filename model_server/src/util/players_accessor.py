from __future__ import annotations

from typing import Optional

from model_shared.db import get_read_cursor


# Literally our only SQL query, This can be moved to a parquet call. 
def fetch_handedness(player_id: str, *, is_batter: bool) -> Optional[str]:
    column = "batting_side" if is_batter else "throwing_arm"
    with get_read_cursor() as cursor:
        cursor.execute(
            f"SELECT {column} FROM players WHERE id = %s LIMIT 1",
            (player_id,),
        )
        row = cursor.fetchone()
    return row[0] if row else None
