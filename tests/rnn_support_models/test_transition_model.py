import numpy as np
import pytest
from model_shared.transition_inference import build_pitch_result_probabilities, build_transition_features_from_parquet, predict_pitch_transition_outcome
from unittest.mock import patch

from tests.rnn_support_models.shared.constants import VALID_GAME_CONTEXT


def test_transition_model_probability_math(mock_model_df):
    mock_df = mock_model_df(model_type='transition')
    result = build_pitch_result_probabilities(mock_df)

    prob_keys = ['p_ball', 'p_strike']

    for key in prob_keys:
        assert key in result

    probs = np.array([result[key] for key in prob_keys])

    assert np.all(probs >= 0)
    assert np.all(probs <= 1)

    total = np.sum(probs)
    assert np.isclose(total, 1.0, atol=1e-6)


def test_transition_invalid_feature_type_raises_exception():
    invalid_features = {
        'balls': 'two',
        'strikes': 'one',
        'stand': 2,
        'p_throws': 1,
        'inning': 'Sixth',
        'inning_topbot': 1,
        'bat_score_diff': [1],
        'on_1b': 'Yes',
        'on_2b': 'No',
        'on_3b': 'Yes',
        'outs_when_up': 'Two',
        'prev_pitch_type': 1,
    }

    with pytest.raises(ValueError):
        predict_pitch_transition_outcome(
            batter_id='660271',
            pitcher_id='660271',
            pitch_type='FF',
            year=2025,
            game_context=invalid_features,
            location=1,
        )


def test_transition_missing_required_inputs_raises_exception():
    with pytest.raises(ValueError):
        predict_pitch_transition_outcome(
            batter_id=None,
            pitcher_id=None,
            pitch_type=None,
            year=None,
            game_context=VALID_GAME_CONTEXT,
            location=1,
        )


def test_transition_missing_game_context_features_raises_exception():
    missing_features = {
        k: v for k, v in VALID_GAME_CONTEXT.items() if k not in ['p_throws', 'balls', 'on_3b', 'prev_pitch_type', 'inning']
    }

    with pytest.raises(KeyError):
        predict_pitch_transition_outcome(
            batter_id='660271',
            pitcher_id='660271',
            pitch_type='FF',
            year=2025,
            game_context=missing_features,
            location=1,
        )


def test_transition_correct_location_metric_mapping(mock_parquet_df):
    target_location = 6

    def fake_location(player_id, year, *, is_batter, metrics):
        row_idx = 1 if is_batter else 0
        row = mock_parquet_df.iloc[row_idx]
        return {f"{row['metric']}_loc{i}": row[f'loc{i}'] for i in range(1, 9)}

    with (
        patch('model_shared.feature_tables.fetch_player_location_features', side_effect=fake_location),
    ):
        result_features = build_transition_features_from_parquet(
            batter_id='660271',
            pitcher_id='660271',
            pitch_type='FF',
            year=2025,
            location=target_location,
        )

    assert result_features['pitcher_loc_whiff_percentage'] == 13.6
    assert result_features['batter_loc_whiff_percentage'] == 40.9


def test_transition_model_full_pipeline():
    result = predict_pitch_transition_outcome(
        batter_id='660271',
        pitcher_id='660271',
        pitch_type='FF',
        year=2025,
        game_context=VALID_GAME_CONTEXT,
        location=1,
    )

    prob_keys = {'p_ball', 'p_strike'}

    assert isinstance(result, dict)
    assert prob_keys.issubset(result.keys())


def test_transition_handles_parquet_load_failure():
    with patch('model_shared.feature_tables.fetch_player_transition_historical_features', side_effect=FileNotFoundError('Parquet file not found on disk')):
        with pytest.raises((FileNotFoundError, RuntimeError)) as excinfo:
            predict_pitch_transition_outcome(
                batter_id='660271',
                pitcher_id='660271',
                pitch_type="FF",
                year=2025,
                game_context=VALID_GAME_CONTEXT,
                location=1
            )

        assert 'Parquet file' in str(excinfo.value)

