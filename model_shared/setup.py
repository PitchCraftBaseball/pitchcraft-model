"""
Setup script. Might call this in an entrypoint script to set up a historical cache for the inference model. TBD. - Dylan  
"""
import logging

from .db import query_historical_pitches_by_year
from .out_type_features import (
    enrich_with_out_type_features,
    with_out_type_historical_pitch_columns,
)
from .parquet import save_training_data
from pitch_rnn.setup_data import validate_feature_list_file

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Running setup script")
    feature_list_dict = validate_feature_list_file("/home/shakotan/git-linux/pitchcraft-repos/pitchcraft-model/feature_list")
    if not feature_list_dict:
        raise ValueError("Feature list validation failed")

    features = feature_list_dict.get("historical_pitches")
    if not features:
        raise ValueError("No features found for historical_pitches")

    features = with_out_type_historical_pitch_columns(features)
    df = query_historical_pitches_by_year("historical_pitches", features, start_year=2021, end_year=2025)
    df = enrich_with_out_type_features(df)

    # Ensure stable parquet typing for date-like values returned as Python objects.
    if "game_date" in df.columns:
        df["game_date"] = df["game_date"].astype("string")

    save_training_data(df)

    
