from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .util.pitch_state_builder import build_pitch_state_from_features
from .util.pitchcraft_inference_helper import build_pitch_probabilities, build_tensors
from model_shared.parameter_loaders import latest_parameters, latest_vocab_csv, load_vocabs_from_csv
from .util.feature_db_accessor import fetch_player_features

# TODO: add controller error messaging that simply errors if the player ID is invalid

class SimplePitchRNN(nn.Module):
    def __init__(
        self,
        cat_vocab_sizes: Dict[str, int],
        num_features: int,
        emb_dim: int,
        hidden: int,
        num_classes: int,
        pad_id: int = 0,
    ) -> None:
        # Build embeddings + RNN + classifier head from artifact metadata.
        super().__init__()
        self.cat_cols = list(cat_vocab_sizes.keys())
        self.embs = nn.ModuleDict(
            {
                col: nn.Embedding(cat_vocab_sizes[col], emb_dim, padding_idx=pad_id)
                for col in self.cat_cols
            }
        )
        in_dim = len(self.cat_cols) * emb_dim + num_features
        self.rnn = nn.RNN(in_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        # Embed categorical features, concat with numeric, then run RNN + linear head.
        embs = []
        for j, col in enumerate(self.cat_cols):
            embs.append(self.embs[col](x_cat[:, :, j]))
        x = torch.cat(embs + [x_num], dim=-1)
        h, _ = self.rnn(x)
        return self.fc(h)

class Artifacts:
    def __init__(self, path: Path) -> None:
        # Load feature spec and hyperparameters from model_config.json.
        data = json.loads(path.read_text())
        self.feature_spec = data["feature_spec"]
        self.bool_cols = list(data.get("bool_cols", []))
        self.max_len = int(data.get("max_len", 8))
        self.pad_id = int(data.get("pad_id", 0))
        self.emb_dim = int(data.get("emb_dim", 16))
        self.hidden = int(data.get("hidden", 128))
        self.cat_cols = list(self.feature_spec["cat_cols"])
        self.num_cols = list(self.feature_spec["num_cols"])

        # Load vocabularies from the most recent vocab export.
        vocab_path = latest_vocab_csv()
        self.cat_vocabs, self.y_vocab = load_vocabs_from_csv(vocab_path)

        self.id_to_pitch = {int(v): k for k, v in self.y_vocab.items()}

    def cat_vocab_sizes(self) -> Dict[str, int]:
        # Compute embedding sizes per categorical column (including padding id).
        sizes = {}
        for col, vocab in self.cat_vocabs.items():
            max_id = max([0] + [int(v) for v in vocab.values()])
            sizes[col] = max_id + 1
        return sizes

    def num_classes(self) -> int:
        # Compute total number of output classes (including padding id).
        return max([0] + [int(v) for v in self.y_vocab.values()]) + 1

class PredictRequest(BaseModel):
    # Player IDs to use when retrieving batter/pitcher features.
    pitcher: str = Field(..., min_length=1)
    batter: str = Field(..., min_length=1)
    # State features are passed directly and already in embed-ready form.
    state_features: Dict[str, Any] = Field(default_factory=dict)
    # Feature name lists for the two retrieval buckets.
    batter_features: List[str] = Field(default_factory=list)
    pitcher_features: List[str] = Field(default_factory=list)

PredictResponse = Dict[str, Dict[str, float]]


def create_app() -> FastAPI:
    # Initialize the FastAPI app, load artifacts/model, and register routes.
    app = FastAPI(title="Pitch RNN Inference API")

    artifacts_path = Path(__file__).resolve().parent / "model_config.json"
    model_path = latest_parameters()

    if not artifacts_path.exists():
        raise RuntimeError("Missing model_config.json. Provide feature spec + vocabs before starting the API.")
    if not model_path.exists():
        raise RuntimeError("Missing simple_pitch_rnn_best.pt. Train the model before starting the API.")

    artifacts = Artifacts(artifacts_path)
    model = SimplePitchRNN(
        cat_vocab_sizes=artifacts.cat_vocab_sizes(),
        num_features=len(artifacts.num_cols),
        emb_dim=artifacts.emb_dim,
        hidden=artifacts.hidden,
        num_classes=artifacts.num_classes(),
        pad_id=artifacts.pad_id,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    @app.get("/health")
    def health() -> Dict[str, str]:
        # Liveness check endpoint.
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        # Retrieve features via the DB layer, then run the model and format results.
        # State features are passed in directly; only fetch player-centric features.
        batter_features = fetch_player_features(
            req.batter,
            req.batter_features,
            entity="batter",
            is_batter=True,
        )
        # Fetch pitcher-centric features using the pitcher ID.
        pitcher_features = fetch_player_features(
            req.pitcher,
            req.pitcher_features,
            entity="pitcher",
            is_batter=False,
        )

        # Build a single-step sequence for now; expand once feature mapping is implemented.
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
            logits = model(x_cat, x_num)
            probs = torch.softmax(logits, dim=-1)[0]

        return build_pitch_probabilities(probs, artifacts, seq_len)

    return app

app = create_app()