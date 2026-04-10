# Pitch RNN Inference API

This folder contains a FastAPI service that loads a trained RNN model and
returns pitch-type probabilities for each pitch in an input sequence.

## Files

- `src/api.py`: FastAPI app for inference.
- `config_generators/build_model_config.py`: Writes a template `src/model_config.json` (feature spec + hyperparams).
- `src/model_config.json`: Feature spec + model hyperparams.

## Updating Model Training 
If you are updating the model training hyperparameters, please update `config_generators/build_model_config.py`. This was slight oversight when coming up with this design; in the future, I would like to refactor part of the training notebooks to just handle exporting the model config on its own.

## Setup
1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Create config (feature spec + hyperparams) from the repo root:
```bash
python -m model_server.config_generators.build_model_config
```

3) Generate vocab exports from the training notebook:
- `model_shared/vocab/rnn_vocab_YYYYMMDD.csv` (categorical + target vocabs)

## Run the API

```bash
uvicorn model_server.src.api:app --host 0.0.0.0 --port 8000
```

If you run from the project root, you can also run 
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
