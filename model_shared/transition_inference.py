"""Inference helper for the transition model (relocated from
``rnn_support_models.transition_model.transition_inference_helper``).

The trained pickles live with the training code at
``rnn_support_models/transition_model/models/`` and are loaded from there
at module import time. Features are pulled from the per-table parquet
files bootstrapped by ``model_shared.setup`` via
``model_shared.feature_tables``, mirroring the approach used by
``rnn_support_models.out_type_model.out_type_inference_helper``.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd
from . inference_utils import prepare_inference_data
from . import feature_tables

LOC_METRICS = (
    'strikeout_percentage', 'whiff_percentage', 'walk_percentage', 'swing_percentage', 'foul_percentage'
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _REPO_ROOT / "rnn_support_models" / "transition_model" / "models"

stage1_data = joblib.load(_MODELS_DIR / 'swing_model_v2.pkl')
stage2_data = joblib.load(_MODELS_DIR / 'called_strike_model_v2.pkl')

swing_model = stage1_data['model']
swing_features = stage1_data['features']

called_strike_model = stage2_data['model']
called_strike_features = stage2_data['features']


def build_pitch_result_probabilities(df: pd.DataFrame) -> Dict[str, float]:
    df_stage1 = prepare_inference_data(df, swing_features)

    p_swing = swing_model.predict_proba(df_stage1)[:, 1]
    p_take = 1 - p_swing

    df_stage2 = prepare_inference_data(df, called_strike_features)
    p_strike_given_take = called_strike_model.predict_proba(df_stage2)[:, 1] # P(Called Strike | Take)
    p_ball_given_take = 1 - p_strike_given_take # P(Ball | Take)

    p_called_strike = p_take * p_strike_given_take # P(Called Strike) = P(Take) * P(Called Strike | Take)
    p_ball = p_take * p_ball_given_take # P(Ball) = P(Take) * P(Ball | Take)

    p_strike = p_swing + p_called_strike # P(Strike) = P(Swing) + P(Called Strike)

    total = p_strike + p_ball
    p_strike /= total
    p_ball /= total

    return {
        'p_strike': p_strike,
        'p_ball': p_ball,
    }


def build_transition_features_from_parquet(
    batter_id: str,
    pitcher_id: str,
    pitch_type: str,
    year: int,
    location: int = None,
) -> Dict[str, Any]:
    batter_prev_raw = feature_tables.fetch_player_transition_historical_features(
        batter_id, year, pitch_type, is_batter=True,
    )
    pitcher_prev_raw = feature_tables.fetch_player_transition_historical_features(
        pitcher_id, year, pitch_type, is_batter=False,
    )
    batter_loc_raw = feature_tables.fetch_player_location_features(
        batter_id, year, is_batter=True, metrics=LOC_METRICS,
    )
    pitcher_loc_raw = feature_tables.fetch_player_location_features(
        pitcher_id, year, is_batter=False, metrics=LOC_METRICS,
    )

    raw: Dict[str, Any] = {}

    rename_cols = {
        'whiff_percentage': 'whiff_percentage',
        'chase_percentage': 'chase_percentage',
        'zone_contact_percentage': 'zone_contact_percentage',
        'first_pitch_swing_percentage': 'first_pitch_swing_percentage',
        'meatball_swing_percentage': 'meatball_swing_percentage',
    }

    pitch_rename_cols = {
        'putaway_percentage': 'putaway_percentage',
        'pitch_whiff_percentage': 'whiff_percentage',
    }

    def _remap_stats(raw_dict, prefix):
        for col_name, rename in rename_cols.items():
            remapped = f'{prefix}_prev_{rename}'
            raw[remapped] = raw_dict.get(col_name)

        for col_name, rename in pitch_rename_cols.items():
            remapped = f'{prefix}_pitch_{rename}'
            raw[remapped] = raw_dict.get(col_name)

        raw['sz_top'] = raw_dict.get('sz_top')
        raw['sz_bot'] = raw_dict.get('sz_bot')

    _remap_stats(batter_prev_raw, 'batter')
    _remap_stats(pitcher_prev_raw, 'pitcher')

    def _map_loc(loc_dict, prefix, target_loc):
        for metric in LOC_METRICS:
            remapped = f'{prefix}_loc_{metric}'
            raw[remapped] = loc_dict.get(f'{metric}_loc{target_loc}')

    _map_loc(batter_loc_raw, 'batter', location)
    _map_loc(pitcher_loc_raw, 'pitcher', location)

    batter_zone_pct = float(batter_prev_raw.get('zone_percentage') or 0)
    batter_swing_pct = float(batter_prev_raw.get('zone_swing_percentage') or 0)
    raw['batter_prev_looking_strike_percentage'] = batter_zone_pct * (1 - batter_swing_pct / 100.0)

    return {k: (float(v) if v is not None else 0.0) for k, v in raw.items()}


def predict_pitch_transition_outcome(
    batter_id: str,
    pitcher_id: str,
    pitch_type: str,
    year: int,
    game_context: Dict[str, Any],
    location: int = None,
) -> Dict[str, float]:
    if None in (batter_id, pitcher_id, pitch_type, year):
        raise ValueError(
            'batter_id, pitcher_id, pitch_type, and year are all required'
        )
    
    parquet_features = build_transition_features_from_parquet(
        batter_id,
        pitcher_id,
        pitch_type,
        year,
        location=location,
    )

    full_features = {**parquet_features, **game_context}

    full_features['pitch_type'] = pitch_type
    full_features['location'] = location

    df = pd.DataFrame([full_features])
    probs = build_pitch_result_probabilities(df)
    return probs
