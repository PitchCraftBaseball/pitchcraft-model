from pathlib import Path
from .db import query_historical_pitches_by_year
from .out_type_features import enrich_with_out_type_features, with_out_type_historical_pitch_columns
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

def get_training_data(features: list[str], force_refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "historical_pitches.parquet"
    
    if cache_path.exists() and not force_refresh:
        return load_training_data()
    
    print("No cache found, querying database...")
    features = with_out_type_historical_pitch_columns(features)
    df = query_historical_pitches_by_year("historical_pitches", features, start_year=2021, end_year=2025)
    df = enrich_with_out_type_features(df)
    save_training_data(df)
    return df
