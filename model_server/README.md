# Pitch RNN Inference API

This folder contains a FastAPI service that loads a trained RNN model and
returns pitch-type probabilities for each pitch in an input sequence.

## Files

- `src/api.py`: FastAPI app for inference.
- `src/model_config.json`: Model architecture hyperparameters (emb_dims, hidden, num_layers, etc.). Feature spec is loaded automatically from the latest `model_shared/vocab/rnn_vocab_*.json` at startup.

## Setup

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Ensure a trained checkpoint and vocab export exist:
- `model_shared/trained-parameters/pitch_rnn_YYYYMMDD.pt`
- `model_shared/vocab/rnn_vocab_YYYYMMDD.json`

The API picks up the most recent file of each at startup automatically.

## Run the API

```bash
uvicorn model_server.src.api:app --host 0.0.0.0 --port 8000
```

Or from the project root:
```bash
./start_server
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
      "count_situation": "even",
      "prev_horiz_bucket": "1",
      "prev_vert_bucket": "1",
      "balls": 1,
      "strikes": 1,
      "outs_when_up": 1,
      "inning": 3,
      "bat_score_diff": 0,
      "on_1b": 0,
      "on_2b": 1,
      "on_3b": 0,
      "prev_in_zone": 1,
      "pitcher_sit_fb_rate": 0.52,
      "pitcher_sit_br_rate": 0.31,
      "pitcher_sit_os_rate": 0.17,
      "pitcher_sit_whiff_rate": 0.28,
      "batter_sit_swing_rate": 0.45,
      "batter_sit_whiff_rate": 0.22
    },
    "batter_features": ["stand"],
    "pitcher_features": ["p_throws"]
  }'
```

The response includes `pitch_one` through `pitch_four`, each a map of pitch type
to probability.
