from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .util.pitch_state_builder import build_pitch_state_from_features
from .util.pitchcraft_inference_helper import build_pitch_probabilities, build_tensors
from model_shared.parameter_loaders import (
    latest_parameters,
    latest_vocab_json,
    load_vocabs_from_json,
)
from model_shared.rnn_definition import PitchRNN
from .util.feature_db_accessor import fetch_player_features


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
    batter_features: List[str] = Field(default_factory=list)
    pitcher_features: List[str] = Field(default_factory=list)


PredictResponse = Dict[str, Dict[str, float]]


def create_app() -> FastAPI:
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

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        batter_features = fetch_player_features(
            req.batter,
            req.batter_features,
            entity="batter",
            is_batter=True,
        )
        pitcher_features = fetch_player_features(
            req.pitcher,
            req.pitcher_features,
            entity="pitcher",
            is_batter=False,
        )

        states = [
            build_pitch_state_from_features(
                req.pitcher,
                req.batter,
                req.state_features,
                batter_features,
                pitcher_features,
            )
        ]

        x_cat, x_num, seq_len = build_tensors(states, artifacts)
        with torch.no_grad():
            logits_pitch, _logits_horiz, _logits_vert = model(x_cat, x_num)
            # TODO: expose location predictions (horiz/vert) in a future API version
            probs = torch.softmax(logits_pitch, dim=-1)[0]

        return build_pitch_probabilities(probs, artifacts, seq_len)

    return app


app = create_app()
