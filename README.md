# pitchcraft-model

Placeholder README for the Pitchcraft model

## Setup

To get started with development, follow these steps:

**1. Clone the repository using Git:**

```bash
git clone <url>
```

You may need to generate a personal access token through GitHub to clone via HTTPS.

**2. Navigate to the root of the project and install the required dependencies:**


```bash
make install
```

If you add dependencies, add them to `requirements.txt` using `pipreqs` or `pip freeze`. I recommend `pipreqs` since it scans the repository for imports so no unnecessary dependencies are added accidentally:

```bash
pip install pipreqs
pipreqs --force .
```

**3. Configure `.env` file:**

Copy the `.env.sample` file into a new file called `.env` and configure the variables as needed. 

The variable `DB_RDS_CERT_PATH` is the path to the SSL certificate bundle for Amazon RDS. To connect to our database with SSL, we need a certificate bundle for Amazon RDS (read more [here](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.SSL.html#UsingWithRDS.SSL.CertificatesDownload)). `aws-rds-cert.pem` is using the bundle for `us-east-1`.

After these environement variables are set, you should be able to connect to AWS RDS database. Here is an example of how to read from the database with the `get_read_cursor` context manager:
```python
from src.data.db import get_read_cursor

query = """
    SELECT * FROM players WHERE id = %s
"""
params = []
params.append(434378)

with get_read_cursor() as cursor:
    cursor.execute(query, tuple(params))
    result = cursor.fetchone()
    print(result)
```

# Notes
1) `psycog2` WILL NOT BUILD WITH ANY VERSION OF PYTHON>=3.11 

## JSON data structures

This project uses a few generated JSON files for run expectancy and pitch arsenal lookups.

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