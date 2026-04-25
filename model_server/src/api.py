from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_shared.feature_engineering.feature_calculator import count_situation
from model_shared.parameter_loaders import (
    latest_parameters,
    latest_vocab_json,
    load_vocabs_from_json,
)
from model_shared.rnn_definition import PitchRNN

from .util.feature_store import FeatureStore, SqlHistoricalPitchesFeatureStore
from .util.pitch_state_builder import (
    MissingFeaturesError,
    build_pitch_state_from_features,
)
from .util.pitchcraft_inference_helper import build_pitch_probabilities, build_tensors
from .util.players_accessor import fetch_handedness


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
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.max_len = int(data.get("max_len", 8))
        self.pad_id = int(data.get("pad_id", 0))
        self.emb_dims: Dict[str, int] = dict(data["emb_dims"])
        self.hidden = int(data.get("hidden", 128))
        self.num_layers = int(data.get("num_layers", 1))

        vocab_path = latest_vocab_json()
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
    state_features: Dict[str, Any] = Field(default_factory=dict)


PredictResponse = Dict[str, Dict[str, float]]


def create_app(feature_store: Optional[FeatureStore] = None) -> FastAPI:
    app = FastAPI(title="Pitch RNN Inference API")

    artifacts_path = Path(__file__).resolve().parent / "model_config.json"
    model_path = latest_parameters()

    if not artifacts_path.exists():
        raise RuntimeError(
            "Missing model_config.json. Provide feature spec + vocabs before starting the API."
        )
    if not model_path.exists():
        raise RuntimeError(
            "Missing pitch_rnn_*.pt. Train the model before starting the API."
        )

    artifacts = Artifacts(artifacts_path)
    model = PitchRNN(
        cat_vocab_sizes=artifacts.cat_vocab_sizes(),
        num_features=len(artifacts.num_cols),
        emb_dims=artifacts.emb_dims,
        hidden=artifacts.hidden,
        num_classes=artifacts.num_classes(),
        num_layers=artifacts.num_layers,
        pad_id=artifacts.pad_id,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    store: FeatureStore = feature_store or SqlHistoricalPitchesFeatureStore()

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        logger.info("predict request: pitcher=%s batter=%s", req.pitcher, req.batter)

        missing = [k for k in REQUIRED_STATE_KEYS if k not in req.state_features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": missing,
                    "message": "state_features is missing required keys.",
                },
            )

        batter_side = fetch_handedness(req.batter, is_batter=True)
        if batter_side is None:
            raise HTTPException(
                status_code=404, detail=f"batter {req.batter} not found"
            )
        pitcher_arm = fetch_handedness(req.pitcher, is_batter=False)
        if pitcher_arm is None:
            raise HTTPException(
                status_code=404, detail=f"pitcher {req.pitcher} not found"
            )

        balls = int(req.state_features["balls"])
        strikes = int(req.state_features["strikes"])
        situation = count_situation(balls, strikes)

        pitcher_splits = store.get_pitcher_situation_splits(req.pitcher, situation)
        batter_splits = store.get_batter_situation_splits(req.batter, situation)

        enriched_state = {
            **req.state_features,
            "stand": batter_side,
            "p_throws": pitcher_arm,
            # TODO: Need to integrate transition model. Until then every
            # request is treated as the start of a new sequence.
            "prev_pitch_type": "START",
        }

        required_cols = list(artifacts.cat_cols) + list(artifacts.num_cols)
        try:
            state = build_pitch_state_from_features(
                req.pitcher,
                req.batter,
                enriched_state,
                batter_splits,
                pitcher_splits,
                required_cols=required_cols,
            )
        except MissingFeaturesError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": exc.missing,
                    "message": "Request is missing features required by the loaded model.",
                },
            )

        x_cat, x_num, seq_len = build_tensors([state], artifacts)
        with torch.no_grad():
            logits_pitch = model(x_cat, x_num)
            probs = torch.softmax(logits_pitch, dim=-1)[0]

        response = build_pitch_probabilities(
            probs, artifacts, seq_len, pitch_keys=["pitch_one"]
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "predict response: pitcher=%s batter=%s elapsed_ms=%.2f",
            req.pitcher,
            req.batter,
            elapsed_ms,
        )
        return response

    return app


app = create_app()
