from __future__ import annotations

from typing import Dict, List, Optional, Protocol

import numpy as np
from pydantic import BaseModel
import torch


class TensorArtifacts(Protocol):
    max_len: int
    pad_id: int
    cat_cols: List[str]
    num_cols: List[str]
    cat_vocabs: dict
    id_to_pitch: Dict[int, str]


def _encode_cat(value: Optional[str], vocab: dict, pad_id: int) -> int:
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


# The `states` array is dynamically generated. We will need to QA this to ensure that the fields are accurately being applied
def build_tensors(
    states: List[BaseModel], artifacts: TensorArtifacts
) -> tuple[torch.Tensor, torch.Tensor, int]:
    # Convert a list of pitch states into padded categorical/numeric tensors.
    max_len = artifacts.max_len
    seq_len = min(len(states), max_len, 4)

    x_cat = np.full(
        (max_len, len(artifacts.cat_cols)), artifacts.pad_id, dtype=np.int64
    )
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


def build_pitch_probabilities(
    probs: torch.Tensor,
    artifacts: TensorArtifacts,
    seq_len: int,
    pitch_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    # Convert model probabilities into a pitch-keyed response payload.
    if pitch_keys is None:
        pitch_keys = [
            "pitch_one",
            "pitch_two",
            "pitch_three",
            "pitch_four",
        ]

    out_probs: List[Dict[str, float]] = []
    pitch_ids = sorted(artifacts.id_to_pitch.keys())

    for t in range(seq_len):
        row: Dict[str, float] = {}
        for pid in pitch_ids:
            if pid == artifacts.pad_id:
                continue
            row[artifacts.id_to_pitch[pid]] = float(probs[t, pid].item())
        out_probs.append(row)

    pitches: Dict[str, Dict[str, float]] = {}
    for t, probs_map in enumerate(out_probs):
        if t >= len(pitch_keys):
            break
        pitches[pitch_keys[t]] = probs_map

    return pitches

# TODO: add build_location_probabilities for horiz/vert heads
