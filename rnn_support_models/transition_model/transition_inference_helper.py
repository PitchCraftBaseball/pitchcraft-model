from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd
from model_server.src.util import feature_db_accessor

# Columns that need to be divided by 100 after fetch from DB
SQL_PCT_COLS = {
    'batter_prev_whiff_rate', 'batter_prev_chase_rate', 'batter_prev_looking_strike_rate',
    'batter_prev_zone_contact_rate', 'batter_pitch_putaway_rate',
    'batter_pitch_whiff_rate', 'pitcher_prev_whiff_rate', 'pitcher_prev_chase_rate',
    'pitcher_pitch_putaway_rate', 'pitcher_pitch_whiff_rate',
    'batter_prev_first_pitch_swing_rate', 'pitcher_prev_first_pitch_swing_rate',
    'batter_prev_meatball_swing_rate', 'pitcher_prev_meatball_swing_rate'
}

ZONE_METRICS = [
    'strikeout_percentage', 'whiff_percentage', 'walk_percentage', 'swing_percentage'
]

_MODELS_DIR = Path(__file__).parent / 'models'

stage1_data = joblib.load(_MODELS_DIR / 'swing_model_v1.pkl')
stage2_data = joblib.load(_MODELS_DIR / 'called_strike_model_v1.pkl')

swing_model = stage1_data['model']
swing_features = stage1_data['features']

called_strike_model = stage2_data['model']
called_strike_features = stage2_data['features']

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

def build_transition_features_from_db(
    batter_id: str,
    pitcher_id: str,
    pitch_type: str,
    year: int,
    zone: int = None,
) -> Dict[str, Any]:
    # Batter previous season general stats
    batter_prev_raw = feature_db_accessor.fetch_player_transition_historical_features(
        batter_id,
        year,
        pitch_type,
        entity='batter_transition_prev',
        is_batter=True,
    )

    # Pitcher previous season general stats
    pitcher_prev_raw = feature_db_accessor.fetch_player_transition_historical_features(
        pitcher_id,
        year,
        pitch_type,
        entity='pitcher_transition_prev',
        is_batter=False,
    )

    # Batter previous season per-zone stats
    batter_zone_raw = feature_db_accessor.fetch_player_zone_features(
        batter_id,
        year,
        entity='batter_transition_zone',
        is_batter=True,
        metrics=ZONE_METRICS,
    )

    # Pitcher previous season per-zone stats
    pitcher_zone_raw = feature_db_accessor.fetch_player_zone_features(
        pitcher_id,
        year,
        entity='pitcher_transition_zone',
        is_batter=False,
        metrics=ZONE_METRICS,
    )

    raw: Dict[str, Any] = {}

    rename_cols = {
        'whiff_percentage': 'whiff_rate',
        'chase_percentage': 'chase_rate',
        'zone_contact_percentage': 'zone_contact_rate',
        'first_pitch_swing_percentage': 'first_pitch_swing_rate',
        'meatball_swing_percentage': 'meatball_swing_rate',
    }

    pitch_rename_cols = {
        'putaway_percentage': 'putaway_rate',
        'pitch_whiff_percentage': 'whiff_rate',
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

    def _map_zone(zone_dict, prefix, target_zone):
        if target_zone is None or zone_dict is None:
            return

        for metric in ZONE_METRICS:
            remapped = f'{prefix}_zone_{metric}'
            raw[remapped] = zone_dict.get(f'{metric}_zone{target_zone}')

    _map_zone(batter_zone_raw, 'batter', zone)
    _map_zone(pitcher_zone_raw, 'pitcher', zone)

    batter_zone_pct = float(batter_prev_raw.get('zone_percentage') or 0)
    batter_swing_pct = float(batter_prev_raw.get('zone_swing_percentage') or 0)
    raw['batter_prev_looking_strike_rate'] = batter_zone_pct * (1 - batter_swing_pct / 100.0)

    # Divide necessary columns by 100 to get a 0 to 1 value
    for col in SQL_PCT_COLS:
        if col in raw and raw[col] is not None:
            raw[col] = float(raw[col]) / 100.0
        else:
            raw[col] = 0.0

    return {k: (float(v) if v is not None else 0.0) for k, v in raw.items()}

def predict_pitch_transition_outcome(
    batter_id: str,
    pitcher_id: str,
    pitch_type: str,
    year: int,
    game_context: Dict[str, Any],
    zone: int = None,
) -> Dict[str, float]:
    db_features = build_transition_features_from_db(
        batter_id,
        pitcher_id,
        pitch_type,
        year,
        zone=zone,
    )

    full_features = {**db_features, **game_context}

    full_features['pitch_type'] = pitch_type
    full_features['zone'] = zone

    df = pd.DataFrame([full_features])
    probs = build_pitch_result_probabilities(df)
    return probs

# test
# context = {
#     'balls': 0,
#     'strikes': 1,
#     'stand': 'L',
#     'p_throws': 'R',
#     'inning': 4,
#     'inning_topbot': 'Top',
#     'bat_score': 2,
#     'fld_score': 1,
#     'runner_on_1b': 1,
#     'runner_on_2b': 0,
#     'runner_on_3b': 0,
#     'outs_when_up': 1,
#     'prev_pitch_type': 'NONE',
#     'prev_zone': 0
# }

# prediction = predict_pitch_transition_outcome(
#     batter_id='575929', # Willson Contreras, decently high swing rates outside strikezone (15.3 |   21.9 |   32.7 |   38.4)
#     pitcher_id='554430', # Zack Wheeler
#     pitch_type='FF',
#     zone=14,
#     year=2025,
#     game_context=context
# )

# print(prediction)
