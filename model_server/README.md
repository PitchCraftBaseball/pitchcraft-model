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
curl -s -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
      "pitcher": "668933",
      "batter":  "695657",
      "state_features": {
        "balls": 1,
        "strikes": 2,
        "outs_when_up": 1,
        "inning": 3,
        "inning_topbot": "Top",
        "bat_score_diff": -2,
        "on_1b": false,
        "on_2b": false,
        "on_3b": true
      }
    }'
```

The response includes `pitch_one` through `pitch_four`, each a map of pitch type
to probability.
