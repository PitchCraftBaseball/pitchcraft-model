from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from ..dto import (
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    PredictedPitch,
)
from .inference_engine import (
    InferenceEngine,
    PitchStep,
    PlayerNotFoundError,
    SimulationResult,
    Strategy,
    VALID_OUT_TYPES,
)
from .pitch_state_builder import MissingFeaturesError


REQUIRED_STATE_KEYS = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "bat_score_diff",
    "on_1b",
    "on_2b",
    "on_3b",
]

"""
Class that maintains an InferenceEngine lifetime and the helper methods that used 
to exist in `api.py`.  
"""
class RequestHandler:
    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine

    @staticmethod
    def validate_state(state_features: Dict[str, Any]) -> List[str]:
        return [k for k in REQUIRED_STATE_KEYS if k not in state_features]

    @staticmethod
    def resolve_strategy(
        req: PredictRequest, *, failed_index: Optional[int] = None
    ) -> Tuple[Strategy, Optional[str]]:
        pref = (req.preferred_out_type or "").strip() or None
        if pref is not None:
            if pref not in VALID_OUT_TYPES:
                detail: Dict[str, Any] = {
                    "message": "preferred_out_type must be one of the allowed values.",
                    "allowed": list(VALID_OUT_TYPES),
                    "got": req.preferred_out_type,
                }
                if failed_index is not None:
                    detail["failed_index"] = failed_index
                raise HTTPException(status_code=422, detail=detail)
            return "preferred", pref
        return req.strategy, None

    def run_simulation(
        self, req: PredictRequest, strategy: Strategy, pref: Optional[str]
    ) -> SimulationResult:
        return self._engine.simulate_plate_appearance(
            pitcher=req.pitcher,
            batter=req.batter,
            year=req.year,
            state_features=req.state_features,
            strategy=strategy,
            max_pitches=req.max_pitches,
            preferred_out_type=pref,
        )

    def run_batch(
        self, items: List[Tuple[PredictRequest, Strategy, Optional[str]]]
    ) -> List[SimulationResult]:
        results: List[SimulationResult] = []
        for i, (item, strategy, pref) in enumerate(items):
            try:
                results.append(self.run_simulation(item, strategy, pref))
            except (PlayerNotFoundError, MissingFeaturesError) as exc:
                exc.failed_index = i  # type: ignore[attr-defined]
                raise
        return results

    @staticmethod
    def step_to_payload(step: PitchStep) -> PredictedPitch:
        return PredictedPitch(
            pitch_index=step.pitch_index,
            pitch_type=step.pitch_type,
            rnn_pitch_probs=step.rnn_pitch_probs,
            p_strike=step.p_strike,
            p_ball=step.p_ball,
            out_type_probs=step.out_type_probs,
            transition_event=step.transition_event,
            out_type_event=step.out_type_event,
            balls_after=step.balls_after,
            strikes_after=step.strikes_after,
            terminal=step.terminal,
            outcome=step.outcome,
            target_location=step.target_location,
        )

    def result_to_response(self, result: SimulationResult) -> PredictResponse:
        return PredictResponse(
            outcome=result.outcome,
            pitch_count=result.pitch_count,
            sequence=[self.step_to_payload(step) for step in result.sequence],
        )

    def results_to_batch_response(
        self, results: List[SimulationResult]
    ) -> BatchPredictResponse:
        return BatchPredictResponse(
            results=[self.result_to_response(r) for r in results]
        )
