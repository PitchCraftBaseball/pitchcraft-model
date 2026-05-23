import pandas as pd

from model_shared.inference_utils import prepare_inference_data
from tests.rnn_support_models.shared.constants import OUT_TYPE_FEATURES


def test_shared_platoon_feature_computation(mock_model_df):
    mock_df = mock_model_df()
    features = ['is_platoon']

    mock_df['stand'] = pd.Categorical(['L'], categories=['L', 'R'])
    mock_df['p_throws'] = pd.Categorical(['R'], categories=['L', 'R'])
    
    result_true = prepare_inference_data(mock_df, features)
    assert result_true.loc[0, 'is_platoon'] == 1

    mock_df['stand'] = pd.Categorical(['R'], categories=['L', 'R'])
    mock_df['p_throws'] = pd.Categorical(['R'], categories=['L', 'R'])
    
    result_false = prepare_inference_data(mock_df, features)
    assert result_false.loc[0, 'is_platoon'] == 0


def test_shared_count_feature_computation(mock_model_df):
    mock_df = mock_model_df()
    features = ['two_strikes', 'full_count']

    mock_df['strikes'] = 2
    mock_df['balls'] = 3
    
    result_full_count = prepare_inference_data(mock_df, features)
    assert result_full_count.loc[0, 'two_strikes'] == 1
    assert result_full_count.loc[0, 'full_count'] == 1

    mock_df['strikes'] = 2
    mock_df['balls'] = 0
    
    result_two_strikes = prepare_inference_data(mock_df, features)
    assert result_two_strikes.loc[0, 'two_strikes'] == 1
    assert result_two_strikes.loc[0, 'full_count'] == 0

    mock_df['strikes'] = 0
    mock_df['balls'] = 0
    
    result_neither = prepare_inference_data(mock_df, features)
    assert result_neither.loc[0, 'two_strikes'] == 0
    assert result_neither.loc[0, 'full_count'] == 0


def test_shared_one_hot_encoding(mock_model_df):
    mock_df = mock_model_df()
    features = [
        'prev_pitch_type_FF', 'pitch_type_SI' , 'stand_R', 'p_throws_R', 'inning_topbot_Top',
        'prev_pitch_type_SI', 'pitch_type_FF'
    ]

    mock_df['prev_pitch_type'] = pd.Categorical(['FF'], categories=['CH', 'CU', 'FF', 'SI', 'SL'])
    mock_df['pitch_type'] = pd.Categorical(['SI'], categories=['CH', 'CU', 'FF', 'SI', 'SL'])
    mock_df['stand'] = pd.Categorical(['R'], categories=['L', 'R'])
    mock_df['p_throws'] = pd.Categorical(['L'], categories=['L', 'R'])
    mock_df['inning_topbot'] = pd.Categorical(['Top'], categories=['Bot', 'Top'])
    
    result = prepare_inference_data(mock_df, features)
    
    assert any(col.startswith('stand_') for col in result.columns)
    assert any(col.startswith('p_throws_') for col in result.columns)
    assert any(col.startswith('pitch_type_') for col in result.columns)
    assert any(col.startswith('prev_pitch_type_') for col in result.columns)
    assert any(col.startswith('inning_topbot_') for col in result.columns)

    assert result.loc[0, 'stand_R'] == 1
    assert result.loc[0, 'p_throws_R'] == 0
    assert result.loc[0, 'prev_pitch_type_FF'] == 1
    assert result.loc[0, 'prev_pitch_type_SI'] == 0
    assert result.loc[0, 'pitch_type_FF'] == 0
    assert result.loc[0, 'pitch_type_SI'] == 1
    assert result.loc[0, 'inning_topbot_Top'] == 1


def test_shared_extra_features_ignored(mock_model_df):
    mock_df = mock_model_df()
    mock_df['extra_feature'] = 10
    mock_df['second_extra_feature'] = 'Test'
    
    result = prepare_inference_data(mock_df, OUT_TYPE_FEATURES)
    
    assert 'extra_feature' not in result.columns
    assert 'second_extra_feature' not in result.columns


def test_shared_missing_features_added(mock_model_df):
    mock_df = mock_model_df()
    mock_df = mock_df.drop('batter_prev_gb_percentage', axis=1)
    
    result = prepare_inference_data(mock_df, OUT_TYPE_FEATURES)
    
    assert 'batter_prev_gb_percentage' in result.columns
    assert result.loc[0, 'batter_prev_gb_percentage'] == 0
