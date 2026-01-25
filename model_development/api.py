from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


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
        embs = []
        for j, col in enumerate(self.cat_cols):
            embs.append(self.embs[col](x_cat[:, :, j]))
        x = torch.cat(embs + [x_num], dim=-1)
        h, _ = self.rnn(x)
        return self.fc(h)


class PitchState(BaseModel):
    pitcher: Optional[str] = None
    batter: Optional[str] = None
    stand: Optional[str] = None
    p_throws: Optional[str] = None
    inning_topbot: Optional[str] = None
    count_state: Optional[str] = None
    prev_pitch_type: Optional[str] = None

    balls: Optional[float] = 0
    strikes: Optional[float] = 0
    outs_when_up: Optional[float] = 0
    inning: Optional[float] = 0
    score_diff_bat: Optional[float] = 0
    on_1b: Optional[float] = 0
    on_2b: Optional[float] = 0
    on_3b: Optional[float] = 0


class PredictRequest(BaseModel):
    sequence: List[PitchState] = Field(..., min_items=1)


class PredictResponse(BaseModel):
    sequence_length: int
    max_len: int
    pitch_types: List[str]
    probabilities: List[Dict[str, float]]


class Artifacts:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.feature_spec = data["feature_spec"]
        self.cat_vocabs = data["cat_vocabs"]
        self.y_vocab = data["y_vocab"]
        self.max_len = int(data.get("max_len", 8))
        self.pad_id = int(data.get("pad_id", 0))
        self.emb_dim = int(data.get("emb_dim", 16))
        self.hidden = int(data.get("hidden", 128))
        self.cat_cols = list(self.feature_spec["cat_cols"])
        self.num_cols = list(self.feature_spec["num_cols"])

        self.id_to_pitch = {int(v): k for k, v in self.y_vocab.items()}

    def cat_vocab_sizes(self) -> Dict[str, int]:
        sizes = {}
        for col, vocab in self.cat_vocabs.items():
            max_id = max([0] + [int(v) for v in vocab.values()])
            sizes[col] = max_id + 1
        return sizes

    def num_classes(self) -> int:
        return max([0] + [int(v) for v in self.y_vocab.values()]) + 1


def _encode_cat(value: Optional[str], vocab: Dict[str, int], pad_id: int) -> int:
    if value is None:
        return pad_id
    return int(vocab.get(str(value), pad_id))


def _encode_num(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_tensors(states: List[PitchState], artifacts: Artifacts) -> tuple[torch.Tensor, torch.Tensor, int]:
    max_len = artifacts.max_len
    seq_len = min(len(states), max_len)

    x_cat = np.full((max_len, len(artifacts.cat_cols)), artifacts.pad_id, dtype=np.int64)
    x_num = np.zeros((max_len, len(artifacts.num_cols)), dtype=np.float32)

    for i in range(seq_len):
        s = states[i]
        for j, col in enumerate(artifacts.cat_cols):
            vocab = artifacts.cat_vocabs[col]
            x_cat[i, j] = _encode_cat(getattr(s, col), vocab, artifacts.pad_id)
        for j, col in enumerate(artifacts.num_cols):
            x_num[i, j] = _encode_num(getattr(s, col))

    x_cat_t = torch.tensor(x_cat, dtype=torch.long).unsqueeze(0)
    x_num_t = torch.tensor(x_num, dtype=torch.float32).unsqueeze(0)
    return x_cat_t, x_num_t, seq_len


def create_app() -> FastAPI:
    app = FastAPI(title="Pitch RNN Inference API")

    artifacts_path = Path("artifacts.json")
    model_path = Path("simple_pitch_rnn_best.pt")

    if not artifacts_path.exists():
        raise RuntimeError("Missing artifacts.json. Export vocabs + feature_spec before starting the API.")
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
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if len(req.sequence) == 0:
            raise HTTPException(status_code=400, detail="sequence must contain at least one pitch state")

        x_cat, x_num, seq_len = build_tensors(req.sequence, artifacts)
        with torch.no_grad():
            logits = model(x_cat, x_num)
            probs = torch.softmax(logits, dim=-1)[0]

        pitch_types = [artifacts.id_to_pitch[i] for i in sorted(artifacts.id_to_pitch.keys()) if i != artifacts.pad_id]
        out_probs: List[Dict[str, float]] = []

        for t in range(seq_len):
            row = {}
            for pid in sorted(artifacts.id_to_pitch.keys()):
                if pid == artifacts.pad_id:
                    continue
                row[artifacts.id_to_pitch[pid]] = float(probs[t, pid].item())
            out_probs.append(row)

        return PredictResponse(
            sequence_length=seq_len,
            max_len=artifacts.max_len,
            pitch_types=pitch_types,
            probabilities=out_probs,
        )

    return app


app = create_app()
