from __future__ import annotations

from pydantic import BaseModel, create_model
from typing import Any, Dict, List, Optional

class StaticPitchState(BaseModel):
    pitcher: int
    batter: int
    # stand: Optional[str] = None
    # p_throws: Optional[str] = None  
    # inning_topbot: Optional[str] = None
    # count_state: Optional[str] = None
    # prev_pitch_type: Optional[str] = None

    # balls: Optional[float] = 0
    # strikes: Optional[float] = 0
    # outs_when_up: Optional[float] = 0
    # inning: Optional[float] = 0
    # score_diff_bat: Optional[float] = 0
    # on_1b: Optional[float] = 0
    # on_2b: Optional[float] = 0
    # on_3b: Optional[float] = 0

def _infer_field_type(value: Any) -> Any:
    if value is None:
        return Any
    return type(value)
    
def build_pitch_state_from_features(
    pitcher_id: str,
    batter_id: str,
    state_features: Dict[str, Any],
    batter_features: Dict[str, Optional[str]],
    pitcher_features: Dict[str, Optional[str]],
) -> BaseModel:

    merged: Dict[str, Any] = {}
    merged.update(state_features)
    merged.update(batter_features)
    merged.update(pitcher_features)
    # Ensure player IDs are always set from the request.
    merged["pitcher"] = pitcher_id
    merged["batter"] = batter_id
    field_definitions = {
        key: (_infer_field_type(value), ...)
        for key, value in merged.items()
    }
    DynamicPitchState = create_model("DynamicPitchState", **field_definitions)
    return DynamicPitchState(**merged)