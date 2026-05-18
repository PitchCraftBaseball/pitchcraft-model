import pandas as pd
import sys
from model_shared.feature_engineering.pitch_constants import *
from model_shared.feature_engineering.data_preprocessor import clean_data
from model_shared.feature_engineering.feature_repository import get_rnn_features
from model_shared.feature_engineering.location import *
from model_shared.parquet import get_training_data
from pitch_rnn.pitch_rnn_trainer import rnn_training_handler
from evaluations.pitch_rnn.evaluate_rnn import evaluate_rnn
from model_shared.db import query_historical_pitches_by_year 
from model_shared.parquet import *
from model_shared.feature_list import validate_feature_list_file

MODEL_TYPE = "group" #group, fastball, breaking or offspeed

_BASE_CAT_COLS = [
    "pitcher", "batter", "stand", "p_throws", "inning_topbot",
    "count_state", "prev_pitch_type", "base_state", "prev_pitch_group",
]

_BASE_NUM_COLS = [
    "outs_when_up", "inning", "bat_score_diff",
    "pitcher_sit_fb_rate", "pitcher_sit_br_rate",
    "pitcher_sit_os_rate", "pitcher_sit_whiff_rate",
    "batter_sit_swing_rate", "batter_sit_whiff_rate",
]

_FAMILY_NUM_COLS = [
    "pitcher_family_fb_rate",
    "pitcher_family_br_rate",
    "pitcher_family_os_rate",
]

FEATURE_SPECS = {
    "group": {
        "target": "y_next_pitch_group",
        "cat_cols": _BASE_CAT_COLS,
        "num_cols": _BASE_NUM_COLS + _FAMILY_NUM_COLS,
    },
    "fastball": {
        "target": "y_next_pitch_fastball",
        "cat_cols": _BASE_CAT_COLS,
        "num_cols": _BASE_NUM_COLS,
    },
    "breaking": {
        "target": "y_next_pitch_breaking",
        "cat_cols": _BASE_CAT_COLS,
        "num_cols": _BASE_NUM_COLS,
    },
    "offspeed": {
        "target": "y_next_pitch_offspeed",
        "cat_cols": _BASE_CAT_COLS,
        "num_cols": _BASE_NUM_COLS,
    },
}

FEATURE_SPEC = FEATURE_SPECS[MODEL_TYPE]

EMB_DIMS = {
    "pitcher": 32,
    "batter":  32,
    "stand": 4,
    "p_throws": 4,
    "inning_topbot": 4,
    "count_state": 8,
    "prev_pitch_type": 8,
    "base_state": 8,
    "prev_pitch_group": 3
}

MODEL_HYPERPARAMETERS = {
    'smoothing_weights': 0.33,
    'epochs': 20,
    'model_layers': 2,
    'optimizer_lr': 0.001,
    'stopping_patience': 3,
    'stopping_delta': 0.001,
    'batch_size': 64,
    'dropout': 0.35,
    'hidden_size': 128
}

def main():
    # Step 1: Load teh Data 
    # data = pd.read_csv('./model-training-notebooks/historical_pitches_rnn_data.csv')
    feature_list_file = sys.argv[1]
    feature_name_list = validate_feature_list_file(feature_list_file)
    
    if not feature_name_list:
        print("Feature validation failed. Exiting.")
        return

    # Extract the flat feature list from the table map
    features = feature_name_list.get("historical_pitches")
    if not features:
        print("No features found for historical_pitches table. Exiting.")
        return

    data = get_training_data(features)
    print("Collected Data")

    # Step 2: Cleaning the Data 
    #Send the data to the preprocessor to get rid of features and pitches we do not care about 
    data = clean_data(data)
    # print("Cleaned Data")

    # # # Step 3: Add Features from Feature Repo
    data = get_rnn_features(data)
    print("Completed Feature Engineering")

    data = data[~data['game_year'].isin([2021, 2022])]

    # # # Step 4: Send to RNN to be trained 
    rnn_training_handler(data, FEATURE_SPEC, EMB_DIMS, MODEL_HYPERPARAMETERS, MODEL_TYPE)

    # # Step 5: Send to be trained
    evaluate_rnn(emb_dims=EMB_DIMS, num_layers=MODEL_HYPERPARAMETERS["model_layers"], use_arsenal_mask=False, hidden=MODEL_HYPERPARAMETERS['hidden_size'])

if __name__ == "__main__":
    #main()
    evaluate_rnn(emb_dims=EMB_DIMS, num_layers=MODEL_HYPERPARAMETERS["model_layers"], use_arsenal_mask=False, hidden=MODEL_HYPERPARAMETERS['hidden_size'])
