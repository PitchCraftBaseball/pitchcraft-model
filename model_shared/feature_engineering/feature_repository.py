from model_shared.feature_engineering.feature_calculator import *

def get_rnn_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out = calculate_pitch_features(out)
    out = calculate_game_state_features(out)

    return out