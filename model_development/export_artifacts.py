from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


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
    "bool_cols": [],
}

PAD_ID = 0
MAX_LEN = 8
EMB_DIM = 16
HIDDEN = 128


def _empty_vocab(fields: List[str]) -> Dict[str, Dict[str, int]]:
    # Produce empty vocabularies to be filled from your database or config workflow.
    return {field: {} for field in fields}


def main() -> None:
    artifacts = {
        "feature_spec": FEATURE_SPEC,
        "cat_vocabs": _empty_vocab(FEATURE_SPEC["cat_cols"]),
        "y_vocab": {},
        "max_len": MAX_LEN,
        "pad_id": PAD_ID,
        "emb_dim": EMB_DIM,
        "hidden": HIDDEN,
    }

    Path("model_config.json").write_text(json.dumps(artifacts, indent=2))
    print("Wrote model_config.json (fill in vocabularies before running the API)")

 
if __name__ == "__main__":
    main()
