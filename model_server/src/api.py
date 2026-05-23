from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_shared.parameter_loaders import (
    latest_parameters_for_specific_model,
    latest_vocab_json_for_specific_model,
    load_vocabs_from_json,
)
from model_shared.rnn_definition import PitchRNN

from .util.feature_store import FeatureStore, ParquetHistoricalPitchesFeatureStore
from .util.inference_engine import (
    InferenceEngine,
    PitchStep,
    PlayerNotFoundError,
    SimulationResult,
    Strategy,
    VALID_OUT_TYPES,
)
from .util.pitch_state_builder import MissingFeaturesError


logger = logging.getLogger(__name__)


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


class Artifacts:
    def __init__(self, config_path: Path, vocab_path: Path) -> None:
        data = json.loads(config_path.read_text())
        self.max_len = int(data.get("max_len", 8))
        self.pad_id = int(data.get("pad_id", 0))
        self.emb_dims: Dict[str, int] = dict(data["emb_dims"])
        self.hidden = int(data.get("hidden", 128))
        self.num_layers = int(data.get("num_layers", 1))

        self.cat_vocabs, self.y_vocab, feature_spec = load_vocabs_from_json(vocab_path)
        self.cat_cols = list(feature_spec["cat_cols"])
        self.num_cols = list(feature_spec["num_cols"])
        self.bool_cols = list(feature_spec.get("bool_cols", []))

        self.id_to_pitch = {int(v): k for k, v in self.y_vocab.items()}

    def cat_vocab_sizes(self) -> Dict[str, int]:
        sizes = {}
        for col, vocab in self.cat_vocabs.items():
            max_id = max([0] + [int(v) for v in vocab.values()])
            sizes[col] = max_id + 1
        return sizes

    def num_classes(self) -> int:
        return max([0] + [int(v) for v in self.y_vocab.values()]) + 1


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


def _step_to_payload(step: PitchStep) -> PredictedPitch:
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


def _load_model_bundle(config_path: Path, model_type: str) -> Tuple[Artifacts, PitchRNN]:
    vocab_path = latest_vocab_json_for_specific_model(model_type)
    param_path = latest_parameters_for_specific_model(model_type)
    arts = Artifacts(config_path, vocab_path)
    model = PitchRNN(
        cat_vocab_sizes=arts.cat_vocab_sizes(),
        num_features=len(arts.num_cols),
        emb_dims=arts.emb_dims,
        hidden=arts.hidden,
        num_classes=arts.num_classes(),
        num_layers=arts.num_layers,
        pad_id=arts.pad_id,
    )
    model.load_state_dict(torch.load(param_path, map_location="cpu"))
    model.eval()
    return arts, model


def create_app(feature_store: Optional[FeatureStore] = None) -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = FastAPI(title="Pitch RNN Inference API")

    config_path = Path(__file__).resolve().parent / "model_config.json"
    if not config_path.exists():
        raise RuntimeError(
            "Missing model_config.json. Provide feature spec + vocabs before starting the API."
        )

    group_artifacts, group_rnn = _load_model_bundle(config_path, "group")
    sub_bundles: Dict[str, Tuple[Artifacts, PitchRNN]] = {
        "fastball": _load_model_bundle(config_path, "fastball"),
        "offspeed": _load_model_bundle(config_path, "offspeed"),
        "breaking": _load_model_bundle(config_path, "breaking"),
    }

    store: FeatureStore = feature_store or ParquetHistoricalPitchesFeatureStore()
    engine = InferenceEngine(
        group_artifacts=group_artifacts,
        group_rnn=group_rnn,
        sub_bundles=sub_bundles,
        feature_store=store,
    )

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        logger.info(
            "predict request: pitcher=%s batter=%s year=%s strategy=%s max_pitches=%s",
            req.pitcher, req.batter, req.year, req.strategy, req.max_pitches,
        )

        missing = [k for k in REQUIRED_STATE_KEYS if k not in req.state_features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": missing,
                    "message": "state_features is missing required keys.",
                },
            )

        pref = (req.preferred_out_type or "").strip() or None
        if pref is not None:
            if pref not in VALID_OUT_TYPES:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "preferred_out_type must be one of the allowed values.",
                        "allowed": list(VALID_OUT_TYPES),
                        "got": req.preferred_out_type,
                    },
                )
            strategy: Strategy = "preferred"
        else:
            strategy = req.strategy

        try:
            result: SimulationResult = engine.simulate_plate_appearance(
                pitcher=req.pitcher,
                batter=req.batter,
                year=req.year,
                state_features=req.state_features,
                strategy=strategy,
                max_pitches=req.max_pitches,
                preferred_out_type=pref,
            )
        except PlayerNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except MissingFeaturesError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": exc.missing,
                    "message": "Request is missing features required by the loaded model.",
                },
            )

        response = PredictResponse(
            outcome=result.outcome,
            pitch_count=result.pitch_count,
            sequence=[_step_to_payload(step) for step in result.sequence],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "predict response: pitcher=%s batter=%s outcome=%s pitch_count=%d elapsed_ms=%.2f",
            req.pitcher, req.batter, result.outcome, result.pitch_count, elapsed_ms,
        )
        return response

    return app


app = create_app()
