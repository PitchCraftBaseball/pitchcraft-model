## JSON data structures

This project uses a few generated JSON files for run expectancy.

### `re288_pitch.json`

Structure: `outs -> base_state -> count -> pitch_type -> {occurrences, run_expectancy}`

Example:

```json
{
    "0": {
        "XXX": {
            "0-0": {
                "FF": {
                    "occurrences": 197919,
                    "run_expectancy": 0.497
                },
                "SL": {
                    "occurrences": 56277,
                    "run_expectancy": 0.5
                }
            }
        }
    }
}
```

Notes:
- `outs` is stored as a string: `"0"`, `"1"`, or `"2"`
- `base_state` is stored as three letters indicating in order of, first, second, and third base, with X indicating no runners on base, and O indicating runner on base.
- `count` is formatted as `ball-strike` such as `0-0`, `1-2`, or `3-2`
- `pitch_type` uses Statcast pitch codes such as `FF`, `SL`, `CH`, `CU`, etc.
- `run_expectancy` is the average runs to score for that situation and outcome type by end of the inning

Generator script: `re288_pitch_json.py`

- Path: `rnn_support_models/run_expectancy/re288_pitch_json.py`
- Dependencies: reads from the `historical_pitches` table via `model_shared.db.get_read_cursor()` (connects to the project's AWS RDS/Postgres). Requires `pandas` and a working DB connection (see `model_shared/db.py` and your `.env`).
- How to run (CLI):

```sh
python3 rnn_support_models/run_expectancy/re288_pitch_json.py --start-year 2015 --end-year 2025 --output-dir ./
```

- Parameters:
    - `--start-year` (int): first year to include. Default: `2015`.
    - `--end-year` (int): last year to include. Default: `2025`.
    - `--output-dir` (str): optional directory to save the JSON file. Default: script directory.
- Output filename: `{start_year}-{end_year}_re288_pitch.json` (written to `--output-dir` or the script directory).
### `re24.json`

Structure: `base_state -> outs -> batted_ball_type -> {occurrences, run_expectancy}`

Example:

```json
{
    "XXX": {
        "0": {
            "strikeout": {
                "occurrences": 104839,
                "run_expectancy": 0.25861
            },
            "groundball": {
                "occurrences": 143270,
                "run_expectancy": 0.44205
            },
            "line_drive": {
                "occurrences": 82231,
                "run_expectancy": 0.72907
            },
            "flyball": {
                "occurrences": 104496,
                "run_expectancy": 0.49698
            }
        }
    }
}
```

Notes:
- `base_state` is stored as three letters indicating in order of, first, second, and third base, with X indicating no runners on base, and O indicating runner on base.
- `outs` is stored as a string: `"0"`, `"1"`, or `"2"`
- `run_expectancy` is the average runs to score for that situation and outcome type by end of the inning

Generator script: `re24.py`

- Path: `rnn_support_models/run_expectancy/re24.py`
- Dependencies: reads from the `historical_pitches` table via `model_shared.db.get_read_cursor()` (connects to the project's AWS RDS/Postgres). Requires `pandas` and a working DB connection.
- How to run (CLI):

```sh
python3 rnn_support_models/run_expectancy/re24.py
```

- Parameters: none (the script uses internal `START_YEAR` / `END_YEAR` constants).
    - `START_YEAR` default: `2015` (edit the file to change)
    - `END_YEAR` default: `2025` (edit the file to change)
- Output filename: `{START_YEAR}-{END_YEAR}_re24.json` (written to the script directory).
