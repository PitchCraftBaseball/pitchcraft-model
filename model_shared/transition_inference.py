"""Inference helper for the transition model (relocated from
``rnn_support_models.transition_model.transition_inference_helper``).

The trained pickles live with the training code at
``rnn_support_models/transition_model/models/`` and are loaded from there
at module import time. Features are pulled from
``data/historical_pitches.parquet`` rather than the feature DB, mirroring
the approach used by
``rnn_support_models.out_type_model.out_type_inference_helper``.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd

ZONE_METRICS = [
    'strikeout_percentage', 'whiff_percentage', 'walk_percentage', 'swing_percentage'
]

SWING_DESCRIPTIONS = {
    'foul_bunt', 'foul', 'hit_into_play', 'swinging_strike', 'foul_tip',
    'swinging_strike_blocked', 'missed_bunt', 'bunt_foul_tip'
}
WHIFF_DESCRIPTIONS = {'swinging_strike', 'foul_tip', 'swinging_strike_blocked'}
IN_ZONE_VALUES = set(range(1, 10))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _REPO_ROOT / "rnn_support_models" / "transition_model" / "models"
_HISTORICAL_PITCHES_PATH = _REPO_ROOT / 'data' / 'historical_pitches.parquet'

stage1_data = joblib.load(_MODELS_DIR / 'swing_model_v1.pkl')
stage2_data = joblib.load(_MODELS_DIR / 'called_strike_model_v1.pkl')

swing_model = stage1_data['model']
swing_features = stage1_data['features']

called_strike_model = stage2_data['model']
called_strike_features = stage2_data['features']

@lru_cache(maxsize=1)
def _load_historical_pitches() -> pd.DataFrame:
    if not _HISTORICAL_PITCHES_PATH.exists():
        raise FileNotFoundError(
            f"No cached historical pitches parquet found at {_HISTORICAL_PITCHES_PATH}"
        )

    df = pd.read_parquet(_HISTORICAL_PITCHES_PATH)
    df = df.copy()
    df['_batter_id'] = df['batter'].astype('Int64').astype(str)
    df['_pitcher_id'] = df['pitcher'].astype('Int64').astype(str)
    df['_zone_int'] = pd.to_numeric(df.get('zone'), errors='coerce').astype('Int64')
    df['_description'] = df['description'].fillna('')
    return df

def _rate(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    if denominator is None or denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)

def _player_history(player_id: str, year: int, is_batter: bool) -> pd.DataFrame:
    df = _load_historical_pitches()
    id_col = '_batter_id' if is_batter else '_pitcher_id'
    return df[(df[id_col] == str(player_id)) & (df['game_year'] == year)]

def _median_float(series: pd.Series, default: float = 0.0) -> float:
    value = pd.to_numeric(series, errors='coerce').median()
    if value is None or np.isnan(value):
        return default
    return float(value)

def _precomputed_transition_features(
    batter_id: str,
    pitcher_id: str,
    pitch_type: str,
    year: int,
    zone: Optional[int],
) -> Dict[str, float]:
    df = _load_historical_pitches()
    feature_cols = sorted(
        {
            col
            for model_features in (swing_features, called_strike_features)
            for col in model_features
            if col in df.columns
            and (
                col.startswith('batter_prev_')
                or col.startswith('batter_pitch_')
                or col.startswith('batter_zone_')
                or col.startswith('pitcher_prev_')
                or col.startswith('pitcher_pitch_')
                or col.startswith('pitcher_zone_')
                or col in {'sz_top', 'sz_bot'}
            )
        }
    )
    if not feature_cols:
        return {}

    mask = (
        (df['_batter_id'] == str(batter_id))
        & (df['_pitcher_id'] == str(pitcher_id))
        & (df['game_year'] == year)
        & (df['pitch_type'] == pitch_type)
    )
    if zone is not None:
        zoned = df[mask & (df['_zone_int'] == int(zone))]
        matched = zoned if not zoned.empty else df[mask]
    else:
        matched = df[mask]

    if matched.empty:
        return {}

    features: Dict[str, float] = {}
    for col in feature_cols:
        value = pd.to_numeric(matched[col], errors='coerce').median()
        if value is not None and not np.isnan(value):
            features[col] = float(value)
    return features

def _parquet_player_features(
    player_id: str,
    year: int,
    pitch_type: str,
    *,
    is_batter: bool,
    prefix: str,
    zone: Optional[int],
) -> Dict[str, float]:
    history = _player_history(player_id, year, is_batter)
    pitch_history = history[history['pitch_type'] == pitch_type]

    features: Dict[str, float] = {}
    if history.empty:
        return features

    descriptions = history['_description']
    swings = descriptions.isin(SWING_DESCRIPTIONS)
    whiffs = descriptions.isin(WHIFF_DESCRIPTIONS)
    in_zone = history['_zone_int'].isin(IN_ZONE_VALUES)
    out_zone = ~in_zone
    contacts = swings & ~whiffs

    features[f'{prefix}_prev_whiff_rate'] = _rate(whiffs.sum(), swings.sum())
    features[f'{prefix}_prev_chase_rate'] = _rate((swings & out_zone).sum(), out_zone.sum())
    features[f'{prefix}_prev_zone_contact_rate'] = _rate((contacts & in_zone).sum(), (swings & in_zone).sum())
    features[f'{prefix}_prev_looking_strike_rate'] = _rate((~swings & in_zone).sum(), len(history))

    first_pitch = history[(history['balls'] == 0) & (history['strikes'] == 0)]
    if not first_pitch.empty:
        first_pitch_swings = first_pitch['_description'].isin(SWING_DESCRIPTIONS)
        features[f'{prefix}_prev_first_pitch_swing_rate'] = _rate(first_pitch_swings.sum(), len(first_pitch))
    else:
        features[f'{prefix}_prev_first_pitch_swing_rate'] = 0.0

    # "Meatball" requires zone+velocity bounds that aren't recoverable from
    # raw pitch rows. The precomputed-rollup path supplies it; this fallback
    # (no matchup history) leaves it at 0.
    features[f'{prefix}_prev_meatball_swing_rate'] = 0.0

    if not pitch_history.empty:
        pitch_desc = pitch_history['_description']
        pitch_swings = pitch_desc.isin(SWING_DESCRIPTIONS)
        pitch_whiffs = pitch_desc.isin(WHIFF_DESCRIPTIONS)
        two_strike_pitches = pitch_history['strikes'] == 2
        features[f'{prefix}_pitch_whiff_rate'] = _rate(pitch_whiffs.sum(), pitch_swings.sum())
        features[f'{prefix}_pitch_putaway_rate'] = _rate((pitch_whiffs & two_strike_pitches).sum(), two_strike_pitches.sum())
    else:
        features[f'{prefix}_pitch_whiff_rate'] = 0.0
        features[f'{prefix}_pitch_putaway_rate'] = 0.0

    if zone is not None:
        zone_history = history[history['_zone_int'] == int(zone)]
        zone_desc = zone_history['_description'] if not zone_history.empty else pd.Series(dtype=object)
        zone_swings = zone_desc.isin(SWING_DESCRIPTIONS)
        zone_whiffs = zone_desc.isin(WHIFF_DESCRIPTIONS)

        features[f'{prefix}_zone_whiff_percentage'] = 100.0 * _rate(zone_whiffs.sum(), zone_swings.sum())
        features[f'{prefix}_zone_swing_percentage'] = 100.0 * _rate(zone_swings.sum(), len(zone_history))

        # PA-level outcome rates aren't reconstructable from raw pitch
        # descriptions alone. The precomputed-rollup path supplies them
        # from zone_metrics; this fallback leaves them at 0.
        features[f'{prefix}_zone_strikeout_percentage'] = 0.0
        features[f'{prefix}_zone_walk_percentage'] = 0.0

    return features

def prepare_inference_data(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cat_cols = ['prev_pitch_type', 'pitch_type', 'stand', 'p_throws', 'inning_topbot']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df['two_strikes'] = (df['strikes'] == 2).astype(int)
    df['full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)

    if 'p_throws_R' in df.columns and 'stand_R' in df.columns:
        df['is_platoon'] = (df['p_throws_R'] != df['stand_R']).astype(int)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

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
    zone: int = None,
) -> Dict[str, Any]:
    precomputed = _precomputed_transition_features(
        batter_id,
        pitcher_id,
        pitch_type,
        year,
        zone,
    )
    if precomputed:
        return precomputed

    batter_history = _player_history(batter_id, year, True)
    pitcher_history = _player_history(pitcher_id, year, False)

    raw: Dict[str, Any] = {}
    raw.update(
        _parquet_player_features(
            batter_id,
            year,
            pitch_type,
            is_batter=True,
            prefix='batter',
            zone=zone,
        )
    )
    raw.update(
        _parquet_player_features(
            pitcher_id,
            year,
            pitch_type,
            is_batter=False,
            prefix='pitcher',
            zone=zone,
        )
    )

    strike_zone_source = batter_history if not batter_history.empty else pitcher_history
    raw['sz_top'] = _median_float(strike_zone_source['sz_top']) if not strike_zone_source.empty else 0.0
    raw['sz_bot'] = _median_float(strike_zone_source['sz_bot']) if not strike_zone_source.empty else 0.0

    return {k: (float(v) if v is not None else 0.0) for k, v in raw.items()}

def predict_pitch_transition_outcome(
    batter_id: str,
    pitcher_id: str,
    pitch_type: str,
    year: int,
    game_context: Dict[str, Any],
    zone: int = None,
) -> Dict[str, float]:
    parquet_features = build_transition_features_from_parquet(
        batter_id,
        pitcher_id,
        pitch_type,
        year,
        zone=zone,
    )

    full_features = {**parquet_features, **game_context}

    full_features['pitch_type'] = pitch_type
    full_features['zone'] = zone

    df = pd.DataFrame([full_features])
    probs = build_pitch_result_probabilities(df)
    return probs
