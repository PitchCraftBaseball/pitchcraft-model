from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from model_shared.feature_spec import FEATURE_SPEC

"""
This file is responsible for generating model_config.json in the format expected by the API when it loads the model parameters. 
- `cat_cols` are categorical features 
- `num_cols` are numerical features
- `bool_cols` are boolean features. 
    - ! NOTE we aren't actually taking advantage of boolean columns at this point because the booleans come out as 1s and 0s from the database. We can refactor it out
"""


# ! If you update the training notebooks, you need to update the hyperparameters from here.
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

    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "model_server" / "src" / "model_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifacts, indent=2))
    print("Wrote model_config.json (fill in vocabularies before running the API)")


if __name__ == "__main__":
    main()
