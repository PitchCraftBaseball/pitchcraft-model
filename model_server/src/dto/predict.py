""" 
Collection of server data transfer objects 
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..util.inference_engine import Strategy


BATCH_MAX_ITEMS = 50


class PredictRequest(BaseModel):
    pitcher: str = Field(..., min_length=1)
    batter: str = Field(..., min_length=1)
    year: int
    state_features: Dict[str, Any] = Field(default_factory=dict)
    strategy: Strategy = "argmax"
    max_pitches: int = Field(default=12, ge=1, le=30)
    preferred_out_type: Optional[str] = None


class PredictedPitch(BaseModel):
    pitch_index: int
    pitch_type: str
    rnn_pitch_probs: Dict[str, float]
    p_strike: float
    p_ball: float
    out_type_probs: Dict[str, float]
    transition_event: str
    out_type_event: str
    balls_after: int
    strikes_after: int
    terminal: bool
    outcome: Optional[str] = None
    target_location: Optional[str] = None


class PredictResponse(BaseModel):
    outcome: Literal["walk", "strikeout", "groundout", "flyout", "hard_hit_flyball", "in_progress"]
    pitch_count: int
    sequence: List[PredictedPitch]


class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest] = Field(..., min_length=1, max_length=BATCH_MAX_ITEMS)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
