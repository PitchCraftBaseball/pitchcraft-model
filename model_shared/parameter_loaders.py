from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

TRAINED_PARAMETERS_DIR = Path(__file__).resolve().parent / "trained-parameters"
VOCAB_DIR = Path(__file__).resolve().parent / "vocab"


def latest_vocab_csv() -> Path:
    # Pick the most recent vocab file based on the YYYYMMDD suffix in the filename.
    vocab_files = sorted(VOCAB_DIR.glob("rnn_vocab_*.csv"))
    if not vocab_files:
        raise RuntimeError("No vocab files found in vocab/. Run the notebook to generate rnn_vocab_YYYYMMDD.csv.")
    return vocab_files[-1]


def load_vocabs_from_json(
    vocab_path: Path,
) -> tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict]:
    data = json.loads(vocab_path.read_text())
    cat_vocabs: Dict[str, Dict[str, int]] = {
        col: {str(k): int(v) for k, v in mapping.items()}
        for col, mapping in data["cat_vocabs"].items()
    }
    y_vocab: Dict[str, int] = {str(k): int(v) for k, v in data["y_vocab"].items()}
    feature_spec: Dict = data.get("feature_spec", {})
    return cat_vocabs, y_vocab, feature_spec

def latest_vocab_json_for_specific_model(model_type: str) -> Path:
    vocab_files = sorted(VOCAB_DIR.glob(f"rnn_vocab_{model_type}_*.json"))
    if not vocab_files:
        raise RuntimeError(f"No JSON vocab files found for model type '{model_type}'.")
    return vocab_files[-1]

def latest_parameters_for_specific_model(model_type: str) -> Path:
    param_files = sorted(TRAINED_PARAMETERS_DIR.glob(f"pitch_rnn_{model_type}_*.pt"))
    if not param_files:
        raise RuntimeError(f"No trained parameters found for model type '{model_type}'.")
    return param_files[-1]