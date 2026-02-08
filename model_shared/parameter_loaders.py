from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

# TODO: MOVE THESE GLOBALS TO A CONFIG FILE?
RNN_VERSION = "v0_1"
VOCAB_DIR = ""
TRAINED_PARAMETERS_DIR = Path(__file__).resolve().parent / "trained-parameters"


# all loader functions
def latest_vocab_csv(vocab_dir: Path) -> Path:
    # Pick the most recent vocab file based on the YYYYMMDD suffix in the filename.
    vocab_files = sorted(vocab_dir.glob("rnn_vocab_*.csv"))
    if not vocab_files:
        raise RuntimeError("No vocab files found in vocab/. Run the notebook to generate rnn_vocab_YYYYMMDD.csv.")
    return vocab_files[-1]


def latest_parameters() -> Path:
    parameters_files = sorted(TRAINED_PARAMETERS_DIR.glob(f"simple_rnn_{RNN_VERSION}_*.pt"))
    if not parameters_files:
        raise RuntimeError("No trained parameters found. Run training once and commit parameters file.")
    return parameters_files[-1]


def load_vocabs_from_csv(vocab_path: Path) -> tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
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
