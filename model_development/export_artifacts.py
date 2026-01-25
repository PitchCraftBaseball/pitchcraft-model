from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


FEATURE_SPEC = {
    "target": "y_next_pitch_type",
    "cat_cols": [
        "pitcher",
        "batter",
        "stand",
        "p_throws",
        "inning_topbot",
        "count_state",
        "prev_pitch_type",
    ],
    "num_cols": [
        "balls",
        "strikes",
        "outs_when_up",
        "inning",
        "score_diff_bat",
        "on_1b",
        "on_2b",
        "on_3b",
    ],
}

PAD_ID = 0
MAX_LEN = 8
EMB_DIM = 16
HIDDEN = 128


def split_by_pa_id(
    df: pd.DataFrame,
    pa_col: str = "pa_id",
    ratios: tuple[float, float] = (0.8, 0.2),
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r_train, r_test = ratios
    if abs((r_train + r_test) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1.0")

    pa_ids = df[pa_col].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(pa_ids)

    n = len(pa_ids)
    n_train = int(n * r_train)
    train_ids = set(pa_ids[:n_train])
    test_ids = set(pa_ids[n_train:])

    train_df = df[df[pa_col].isin(train_ids)].copy()
    test_df = df[df[pa_col].isin(test_ids)].copy()
    return train_df, test_df


def build_vocab(values: pd.Series) -> Dict[str, int]:
    uniq = pd.Series(values.dropna().unique())
    return {str(v): i for i, v in enumerate(uniq, start=1)}


def main() -> None:
    data_path = Path("rnn_data.csv")
    if not data_path.exists():
        raise FileNotFoundError("rnn_data.csv not found in current directory")

    data = pd.read_csv(data_path)
    data["is_real_pitch"] = data["pitch_type"].notna() & (data["pitch_type"] != "ABS")
    data["target_is_real_pitch"] = data.groupby("pa_id")["is_real_pitch"].shift(-1)
    data["y_next_pitch_type"] = data.groupby("pa_id")["pitch_type"].shift(-1)
    data_train = data[data["target_is_real_pitch"] == True].copy()

    train_df, _ = split_by_pa_id(data_train, pa_col="pa_id", ratios=(0.8, 0.2), seed=7)

    cat_vocabs = {c: build_vocab(train_df[c]) for c in FEATURE_SPEC["cat_cols"]}
    y_vocab = build_vocab(train_df[FEATURE_SPEC["target"]])

    artifacts = {
        "feature_spec": FEATURE_SPEC,
        "cat_vocabs": cat_vocabs,
        "y_vocab": y_vocab,
        "max_len": MAX_LEN,
        "pad_id": PAD_ID,
        "emb_dim": EMB_DIM,
        "hidden": HIDDEN,
    }

    Path("artifacts.json").write_text(json.dumps(artifacts, indent=2))
    print("Wrote artifacts.json")


if __name__ == "__main__":
    main()
