from __future__ import annotations

from pydantic import BaseModel, create_model
from typing import Any, Dict, List, Optional

class StaticPitchState(BaseModel):
    pitcher: int
    batter: int

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