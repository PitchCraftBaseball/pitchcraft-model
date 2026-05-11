import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from tests.rnn_support_models.shared.constants import OUT_TYPE_FEATURES, TRANSITION_FEATURES

def generate_mock_value(feature):
    if 'angle' in feature:
        return np.random.uniform(-15.0, 45.0)
    
    if 'velocity' in feature:
        return np.random.uniform(70.0, 110.0)
    
    if 'average_mph' in feature:
        return np.random.uniform(70.0, 100.0) 
    
    if any(x in feature for x in ['per_pa', 'average']):
        return np.random.uniform(0.0, 0.350)
    
    if 'percentage' in feature:
        return np.random.uniform(0.0, 100.0)
    
    if feature in ['pitch_type', 'prev_pitch_type']:
        return np.random.choice(['FF', 'SL', 'CH', 'SI', 'CU'])
    
    if feature in ['on_1b', 'on_2b', 'on_3b', 'balls', 'inning', 'location', 'bat_score_diff', 'strikes', 'outs_when_up']:
        return np.random.randint(0, 1)
    
    if feature in ['p_throws', 'stand']:
        return np.random.choice(['R', 'L'])
    
    return 0.0


@pytest.fixture
def mock_model_df():
    def _build(model_type='out_type'):
        if model_type == 'out_type':
            features = OUT_TYPE_FEATURES
        else:
            features = TRANSITION_FEATURES

        data = {}
        for feature in features:
            data[feature] = [generate_mock_value(feature)]
        return pd.DataFrame(data)

    return _build

@pytest.fixture
def mock_parquet_df():
    return pd.DataFrame([
        {
            'player_id': 660271,
            'position': 'P',
            'year': 2025,
            'metric': 'whiff_percentage',
            'loc1': 22.4, 'loc2': 28.3, 'loc3': 19.6, 'loc4': 20.5,
            'loc5': 52, 'loc6': 13.6, 'loc7': 41.2, 'loc8': 63
        },
        {
            'player_id': 660271,
            'position': 'B',
            'year': 2025,
            'metric': 'whiff_percentage',
            'loc1': 25.1, 'loc2': 24.9, 'loc3': 20.1, 'loc4': 21.4,
            'loc5': 34.7, 'loc6': 40.9, 'loc7': 57.8, 'loc8': 53.3
        }
    ])


@pytest.fixture(autouse=True)
def mock_parquet_reads():
    with( 
        patch('model_shared.feature_tables.fetch_player_out_type_historical_features', return_value={}),
        patch('model_shared.feature_tables.fetch_player_transition_historical_features', return_value={}),
        patch('model_shared.feature_tables.fetch_player_location_features', return_value={})
    ):
        yield
