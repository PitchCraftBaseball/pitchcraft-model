from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def save_training_data(df: pd.DataFrame, filename: str = "historical_pitches.parquet") -> None:
    path = DATA_DIR / filename
    df.to_parquet(path, index=False, compression="snappy")
    print(f"Saved {len(df):,} rows to {path}")

def load_training_data(filename: str = "historical_pitches.parquet") -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No cached data found at {path}. Run the DB query first.")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")
    return df