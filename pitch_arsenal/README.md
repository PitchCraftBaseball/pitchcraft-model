## JSON data structures

This project uses a few generated JSON files for pitch arsenal lookups.

### `pitch_arsenal.json`

Structure: `pitcher_id -> {player_name, year -> yearly_pitch_stats}`

Example:

```json
{
    "112526": {
        "player_name": "Colon, Bartolo",
        "2015": {
            "pitches": 2922,
            "pitch_type": {
                "SI": 1550,
                "FF": 843,
                "SL": 278
            },
            "pitch_type_percentage": {
                "SI": 0.530459,
                "FF": 0.288501,
                "SL": 0.09514
            },
            "pitch_group": {
                "fastball": 2393,
                "breaking": 287,
                "offspeed": 209,
                "other": 0
            },
            "pitch_group_percentage": {
                "fastball": 0.81896,
                "breaking": 0.09822,
                "offspeed": 0.071526,
                "other": 0.0
            }
        }
    }
}
```

Notes:
- `pitcher_id` is stored as a string key
- `pitches` is the total pitch count for that pitcher in that year
- `pitch_type` contains raw counts for each pitch code used by that pitcher
- `pitch_group` combines pitch types into `fastball`, `breaking`, `offspeed`, and `other`
- Percentage fields are normalized to the yearly pitch total

Generator script: `pitch_arsenal_json.py`

- Path: `pitch_arsenal/pitch_arsenal_json.py`
- Dependencies: reads from the `historical_pitches` table via `model_shared.db.get_read_cursor()` (connects to the project's AWS RDS/Postgres). Uses `model_shared.logger` and requires `pandas`-compatible DB access for large queries.
- How to run (CLI):

```sh
python3 pitch_arsenal/pitch_arsenal_json.py --start-year 2015 --end-year 2025 --output-dir ./pitch_arsenal
```

- Parameters:
    - `--start-year` (int): first year to include. Default: `2015`.
    - `--end-year` (int): last year to include. Default: `2025`.
    - `--output-dir` (str): optional directory to save the JSON file. Default: script directory (`pitch_arsenal/` when run from project root).
- Output filename: `{start_year}-{end_year}_pitch_arsenal.json` (written to `--output-dir` or the script directory).

