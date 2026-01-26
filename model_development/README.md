# Pitch RNN Inference API

This folder contains a FastAPI service that loads a trained RNN model and
returns pitch-type probabilities for each pitch in an input sequence.

## Files

- `api.py`: FastAPI app for inference.
- `export_artifacts.py`: Writes a template `model_config.json` (feature spec + hyperparams).
- `model_config.json`: Feature spec + model hyperparams (vocabs are loaded from `vocab/`).
- `vocab/`: Date-stamped vocab exports from the training notebook (`rnn_vocab_YYYYMMDD.csv`).
- `feature_list/`: Date-stamped feature lists from the training notebook (`rnn_vocab_YYYYMMDD.csv`).
- `simple_pitch_rnn_best.pt`: Trained model weights (produced by the notebook).

## Setup

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Create config (feature spec + hyperparams):
```bash
python export_artifacts.py
```

3) Generate vocab exports from the training notebook:
- `vocab/rnn_vocab_YYYYMMDD.csv` (categorical + target vocabs)
- `feature_list/rnn_vocab_YYYYMMDD.csv` (feature list)

4) Ensure model weights exist:
`simple_pitch_rnn_best.pt`

## Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Test Run

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "pitcher": "642121",
    "batter": "595777",
    "state_features": {
      "inning_topbot": "Top",
      "count_state": "1-1",
      "prev_pitch_type": "FF",
      "balls": 1,
      "strikes": 1,
      "outs_when_up": 1,
      "inning": 3,
      "score_diff_bat": 0,
      "on_1b": 0,
      "on_2b": 1,
      "on_3b": 0
    },
    "batter_features": ["stand"],
    "pitcher_features": ["p_throws"]
  }'
```

The response includes `pitch_one` ... `pitch_four`, each with pitch-type
probabilities.
