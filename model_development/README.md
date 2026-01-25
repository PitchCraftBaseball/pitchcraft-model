# Pitch RNN Inference API

This folder contains a small FastAPI service that loads a trained RNN model and
returns pitch-type probabilities for each pitch in an input sequence.

## Files

- `api.py`: FastAPI app for inference.
- `export_artifacts.py`: Builds `artifacts.json` from `rnn_data.csv`.
- `simple_pitch_rnn_best.pt`: Trained model weights (produced by the notebook).
- `artifacts.json`: Saved feature spec + vocabularies + model hyperparams.

## Setup

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Create artifacts (requires `rnn_data.csv`):
```bash
python export_artifacts.py
```

3) Ensure model weights exist:
`simple_pitch_rnn_best.pt`

## Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Test Run

```bash
curl -s -X POST http://localhost:8000/predict \\
  -H 'Content-Type: application/json' \\
  -d '{
    "sequence": [
      {
        "pitcher": "12345",
        "batter": "98765",
        "stand": "L",
        "p_throws": "R",
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
      }
    ]
  }'
```

The response will include a `probabilities` array with a pitch-type probability
map for each pitch in the input sequence.
