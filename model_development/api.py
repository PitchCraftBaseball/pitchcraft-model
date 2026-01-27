from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# Ensure repo root is on sys.path so "src" imports resolve when running from model_development.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.db import find_table_for_column, get_read_cursor

# TODO: MOVE THESE GLOBALS TO A CONFIG FILE? 
RNN_VERSION = "v0_1"

# TODO: add controller error messaging that simply errors if the player ID is invalid
# TODO: break these out into individual modules? 
# code doesnt seem that bad to look at right now

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

# TODO: either get this code generated and move it to a different file for validation, or discuss another approach
class PitchState(BaseModel):
    pitcher: Optional[str] = None
    batter: Optional[str] = None
    stand: Optional[str] = None
    p_throws: Optional[str] = None # todo: discuss this field as a feature in general. the field comes from historical pitches, which is one to many rows 
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
    # Player IDs to use when retrieving batter/pitcher features.
    pitcher: str = Field(..., min_length=1)
    batter: str = Field(..., min_length=1)
    # State features are passed directly and already in embed-ready form.
    state_features: Dict[str, Any] = Field(default_factory=dict)
    # Feature name lists for the two retrieval buckets.
    batter_features: List[str] = Field(default_factory=list)
    pitcher_features: List[str] = Field(default_factory=list)


PredictResponse = Dict[str, Dict[str, float]]


def _latest_vocab_csv(vocab_dir: Path) -> Path:
    # Pick the most recent vocab file based on the YYYYMMDD suffix in the filename.
    vocab_files = sorted(vocab_dir.glob("rnn_vocab_*.csv"))
    if not vocab_files:
        raise RuntimeError("No vocab files found in vocab/. Run the notebook to generate rnn_vocab_YYYYMMDD.csv.")
    return vocab_files[-1]

def _latest_parameters(trained_parameters_dir: Path) -> Path: 
    parameters_files = sorted(trained_parameters_dir.glob(f"simple_rnn_{RNN_VERSION}_*.pt"))
    if not parameters_files:
        raise RuntimeError("No trained parameters found. Run training once and commit parameters file.")
    return parameters_files[-1]

def _load_vocabs_from_csv(vocab_path: Path) -> tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    # Load categorical and target vocabularies from the exported CSV.
    cat_vocabs: Dict[str, Dict[str, int]] = {}
    y_vocab: Dict[str, int] = {}
    with vocab_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            kind = row.get("kind", "")
            feature = row.get("feature", "")
            value = row.get("value", "")
            idx = int(row["id"]) if row.get("id") else 0
            if kind == "categorical":
                cat_vocabs.setdefault(feature, {})[str(value)] = idx
            elif kind == "target":
                y_vocab[str(value)] = idx
    return cat_vocabs, y_vocab


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
        vocab_path = _latest_vocab_csv(Path("vocab"))
        self.cat_vocabs, self.y_vocab = _load_vocabs_from_csv(vocab_path)

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


def _encode_cat(value: Optional[str], vocab: Dict[str, int], pad_id: int) -> int:
    # Map a categorical value to its vocab id, defaulting to pad_id.
    if value is None:
        return pad_id
    return int(vocab.get(str(value), pad_id))


def _encode_num(value: Optional[float]) -> float:
    # Coerce numeric values to float; return 0.0 for missing/invalid input.
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_tensors(states: List[PitchState], artifacts: Artifacts) -> tuple[torch.Tensor, torch.Tensor, int]:
    # Convert a list of pitch states into padded categorical/numeric tensors.
    max_len = artifacts.max_len
    seq_len = min(len(states), max_len, 4)

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


def _fetch_player_features(
    cursor,
    player_id: str,
    feature_names: List[str],
    *,
    entity: str,
    is_batter: bool,
) -> Dict[str, Optional[str]]:
    # Query across tables to find each feature for the given player_ID.
    # TODO: Decide how to aggregate values when a feature lives in historical_pitches.
    features: Dict[str, Optional[str]] = {}
    for feature in feature_names:
        table = find_table_for_column("public", feature)
        if table is None:
            features[feature] = None
            continue
        id_column = "player_id"
        if table == "players":
            # TODO: Consider renaming players.id to player_id for consistency.
            id_column = "id"
        if table == "historical_pitches":
            # Use batter/pitcher IDs for historical pitch rows.
            id_column = "batter" if is_batter else "pitcher"
        cursor.execute(
            f"SELECT {feature} FROM {table} WHERE {id_column} = %s LIMIT 1",
            (player_id,),
        )
        row = cursor.fetchone()
        features[feature] = row[0] if row else None
    # Debug: write out retrieved feature values to a dated log file.
    debug_dir = Path("player-feature-retrieval-debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    debug_path = debug_dir / f"player_features_{entity}_{stamp}.log"
    with debug_path.open("a", encoding="utf-8") as handle:
        handle.write(f"player_ID={player_id}\n")
        handle.write(f"requested_features={feature_names}\n")
        handle.write(f"retrieved_features={features}\n")
        handle.write("---\n")
    return features


def _build_pitch_state_from_features(
    pitcher_id: str,
    batter_id: str,
    state_features: Dict[str, Any],
    batter_features: Dict[str, Optional[str]],
    pitcher_features: Dict[str, Optional[str]],
) -> PitchState:
    # Boilerplate mapper from retrieved features into the PitchState schema.
    # TODO: Map feature names to the PitchState fields and numeric inputs.
    merged: Dict[str, Any] = {}
    merged.update(state_features)
    merged.update(batter_features)
    merged.update(pitcher_features)
    # Ensure player IDs are always set from the request.
    merged["pitcher"] = pitcher_id
    merged["batter"] = batter_id
    return PitchState(**merged)


def create_app() -> FastAPI:
    # Initialize the FastAPI app, load artifacts/model, and register routes.
    app = FastAPI(title="Pitch RNN Inference API")

    artifacts_path = Path("model_config.json")
    model_dir = Path(Path.cwd() / "model-training-notebooks" / "trained-parameters")
    model_path = _latest_parameters(model_dir)

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
        with get_read_cursor() as cursor:
            # State features are passed in directly; only fetch player-centric features.
            batter_features = _fetch_player_features(
                cursor,
                req.batter,
                req.batter_features,
                entity="batter",
                is_batter=True,
            )
            # Fetch pitcher-centric features using the pitcher ID.
            pitcher_features = _fetch_player_features(
                cursor,
                req.pitcher,
                req.pitcher_features,
                entity="pitcher",
                is_batter=False,
            )

        # Build a single-step sequence for now; expand once feature mapping is implemented.
        states = [
            _build_pitch_state_from_features(
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

        out_probs: List[Dict[str, float]] = []

        for t in range(seq_len):
            row = {}
            for pid in sorted(artifacts.id_to_pitch.keys()):
                if pid == artifacts.pad_id:
                    continue
                row[artifacts.id_to_pitch[pid]] = float(probs[t, pid].item())
            out_probs.append(row)

        pitch_keys = [
            "pitch_one",
            "pitch_two",
            "pitch_three",
            "pitch_four",
        ]
        pitches: Dict[str, Dict[str, float]] = {}
        for t, probs_map in enumerate(out_probs):
            if t >= len(pitch_keys):
                break
            pitches[pitch_keys[t]] = probs_map

        return pitches

    return app


app = create_app()
