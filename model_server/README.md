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

`/predict` now simulates a full plate appearance: it loops through RNN →
transition model → out-type model, advancing the count and updating
`prev_pitch_type` until the PA ends or `max_pitches` is reached.

Top-level fields:
- `year` (required) — used by the transition and out-type models to fetch
  prior-season stats from the DB.
- `strategy` (optional, default `"argmax"`) — `"argmax"` picks the most
  likely pitch / event each step; `"sample"` draws from the model
  distributions for a stochastic rollout.
- `max_pitches` (optional, default `12`, range `1`–`30`) — safety cap.
  If the loop hits the cap before terminating, the response's `outcome`
  is `"in_progress"`.

### Mid-PA example (1 ball, 2 strikes)

```bash
curl -s -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
      "pitcher": "668933",
      "batter":  "695657",
      "year": 2025,
      "strategy": "argmax",
      "max_pitches": 12,
      "state_features": {
        "balls": 1,
        "strikes": 2,
        "outs_when_up": 1,
        "inning": 3,
        "inning_topbot": "Top",
        "bat_score_diff": -2,
        "on_1b": false,
        "on_2b": false,
        "on_3b": true,
        "prev_pitch_type": "FF"
      }
    }'
```

### Initial plate appearance (fresh PA, 0-0 count)

For the first pitch of a PA, omit `prev_pitch_type` (or set it to
`"START"`) and start with `balls: 0, strikes: 0`:

```bash
curl -s -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
      "pitcher": "668933",
      "batter":  "695657",
      "year": 2025,
      "strategy": "argmax",
      "max_pitches": 12,
      "state_features": {
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "inning": 1,
        "inning_topbot": "Top",
        "bat_score_diff": 0,
        "on_1b": false,
        "on_2b": false,
        "on_3b": false
      }
    }'
```
**Note** we currently have `"year": 2025` specifically for pulling back historical data. 

### Response shape

```json
{
  "outcome": "walk | strikeout | groundout | flyout | hard_hit_flyball | in_progress",
  "pitch_count": 5,
  "sequence": [
    {
      "pitch_index": 1,
      "pitch_type": "FF",
      "rnn_pitch_probs": { "FF": 0.42, "SL": 0.31, "...": 0.0 },
      "p_strike": 0.61,
      "p_ball": 0.39,
      "out_type_probs": { "p_none": 0.78, "p_so": 0.05, "p_go": 0.09, "p_fo": 0.06, "p_hhfb": 0.02 },
      "transition_event": "strike",
      "out_type_event": "none",
      "balls_after": 0,
      "strikes_after": 1,
      "terminal": false,
      "outcome": null
    }
  ]
}
```

Only the final entry in `sequence` will have `terminal: true` and a
non-null `outcome` (matching the top-level `outcome`).
