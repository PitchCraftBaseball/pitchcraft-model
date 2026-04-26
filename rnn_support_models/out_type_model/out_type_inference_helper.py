from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd

from model_shared import feature_tables

# Columns that need to be divided by 100 after fetch (DB stores them on a
# 0-100 scale; the models were trained on 0-1).
SQL_PCT_COLS = {
    'batter_prev_fb_rate', 'batter_prev_gb_rate', 'batter_prev_whiff_rate',
    'batter_prev_chase_rate', 'batter_prev_weak_rate', 'batter_prev_under_rate',
    'batter_prev_topped_rate', 'batter_prev_flareburner_rate', 'batter_prev_solid_rate', 'batter_prev_barrel_rate',
    'batter_prev_barrels_per_pa', 'batter_prev_looking_strike_rate',
    'batter_prev_zone_contact_rate', 'batter_pitch_putaway_rate',
    'batter_pitch_whiff_rate', 'pitcher_prev_fb_rate', 'pitcher_prev_gb_rate',
    'pitcher_prev_whiff_rate', 'pitcher_prev_chase_rate', 'pitcher_prev_weak_rate',
    'pitcher_prev_under_rate', 'pitcher_prev_topped_rate', 'pitcher_prev_flareburner_rate', 'pitcher_prev_solid_rate', 'pitcher_prev_barrel_rate',
    'pitcher_prev_barrels_per_pa', 'pitcher_pitch_putaway_rate', 'pitcher_pitch_whiff_rate',
}

ZONE_METRICS = [
    'batting_average', 'average_exit_velocity',
    'average_launch_angle', 'contact_batting_average',
    'hard_hit_bip_percentage', 'expected_batting_average',
    'strikeout_percentage', 'whiff_percentage', 'walk_percentage', 'ground_ball_percentage',
    'line_drive_percentage', 'fly_ball_percentage', 'popup_percentage', 'swing_percentage'
]

_MODELS_DIR = Path(__file__).parent / 'models'

stage1_data = joblib.load(_MODELS_DIR / 'pa_end_model_v1.pkl')
stage2_data = joblib.load(_MODELS_DIR / 'so_model_v1.pkl')
stage3_data = joblib.load(_MODELS_DIR / 'bip_model_v1.pkl')
stage4_data = joblib.load(_MODELS_DIR / 'fb_model_v1.pkl')

pa_end_model = stage1_data['model']
pa_end_features = stage1_data['features']

so_model = stage2_data['model']
so_features = stage2_data['features']

bip_model = stage3_data['model']
bip_features = stage3_data['features']

fb_model = stage4_data['model']
fb_features = stage4_data['features']


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


def build_out_type_probabilities(df: pd.DataFrame) -> Dict[str, float]:
    df_stage1 = prepare_inference_data(df, pa_end_features)

    p_pa_end = pa_end_model.predict_proba(df_stage1)[:, 1]
    p_no_end = 1 - p_pa_end

    df_stage2 = prepare_inference_data(df, so_features)
    p_so_given_end = so_model.predict_proba(df_stage2)[:, 1] # P(SO | PA End)
    p_bip_given_end = 1 - p_so_given_end # P(BIP | PA End)

    df_stage3 = prepare_inference_data(df, bip_features)
    p_gb_given_bip = bip_model.predict_proba(df_stage3)[:, 1] # P(GB | BIP, PA End)
    p_fb_given_bip = 1 - p_gb_given_bip # P(FB | BIP, PA End)

    p_so = p_pa_end * p_so_given_end # P(SO) = P(PA End) * P(SO | PA End)
    p_bip_end = p_pa_end * p_bip_given_end # P(BIP) = P(PA End) * P(BIP | PA End)

    p_go = p_bip_end * p_gb_given_bip # P(GB) = P(BIP) * P(GB | BIP)
    p_fb = p_bip_end * p_fb_given_bip # P(FB) = P(BIP) * P(FB | BIP)

    df_stage4 = prepare_inference_data(df, fb_features)
    p_hhfb_given_fb = fb_model.predict_proba(df_stage4)[:, 1]

    p_hhfb = p_fb * p_hhfb_given_fb # P(HHFB) = P(FB) * P(HHFB | FB)
    p_fo = p_fb * (1.0 - p_hhfb_given_fb) # P(FO) = 1 - P(HHFB | FB), regular fly balls that aren't hard-hit

    return {
        'p_none': p_no_end,
        'p_so': p_so,
        'p_go': p_go,
        'p_fo': p_fo,
        'p_hhfb': p_hhfb
    }


def build_out_type_features_from_parquet(
    batter_id: str,
    pitcher_id: str,
    pitch_type: str,
    year: int,
    zone: int = None,
) -> Dict[str, Any]:
    batter_prev_raw = feature_tables.fetch_player_out_type_historical_features(
        batter_id, year, pitch_type, is_batter=True,
    )
    pitcher_prev_raw = feature_tables.fetch_player_out_type_historical_features(
        pitcher_id, year, pitch_type, is_batter=False,
    )
    batter_zone_raw = feature_tables.fetch_player_zone_features(
        batter_id, year, is_batter=True, metrics=ZONE_METRICS,
    )
    pitcher_zone_raw = feature_tables.fetch_player_zone_features(
        pitcher_id, year, is_batter=False, metrics=ZONE_METRICS,
    )

    raw: Dict[str, Any] = {}

    rename_cols = {
        'ground_ball_percentage': 'gb_rate',
        'air_ball_percentage': 'fb_rate',
        'whiff_percentage': 'whiff_rate',
        'chase_percentage': 'chase_rate',
        'weak_percentage': 'weak_rate',
        'under_percentage': 'under_rate',
        'topped_percentage': 'topped_rate',
        'flareburner_percentage': 'flareburner_rate',
        'solid_percentage': 'solid_rate',
        'barrel_percentage': 'barrel_rate',
        'barrels_per_pa': 'barrels_per_pa',
        'zone_contact_percentage': 'zone_contact_rate',
    }

    pitch_rename_cols = {
        'batting_average': 'batting_average',
        'putaway_percentage': 'putaway_rate',
        'pitch_whiff_percentage': 'whiff_rate',
        'average_launch_angle': 'average_launch_angle',
        'average_exit_velocity': 'average_exit_velocity',
        'expected_batting_average': 'expected_batting_average',
        'average_mph': 'average_mph',
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

    raw['batter_zone_fly_ball_rate'] = (
        float(raw.get('batter_zone_line_drive_percentage') or 0) +
        float(raw.get('batter_zone_fly_ball_percentage') or 0) +
        float(raw.get('batter_zone_popup_percentage') or 0)
    )
    raw['pitcher_zone_fly_ball_rate'] = (
        float(raw.get('pitcher_zone_line_drive_percentage') or 0) +
        float(raw.get('pitcher_zone_fly_ball_percentage') or 0) +
        float(raw.get('pitcher_zone_popup_percentage') or 0)
    )

    for col in SQL_PCT_COLS:
        if col in raw and raw[col] is not None:
            raw[col] = float(raw[col]) / 100.0
        else:
            raw[col] = 0.0

    return {k: (float(v) if v is not None else 0.0) for k, v in raw.items()}


def predict_pitch_out_type_outcome(
    batter_id: str,
    pitcher_id: str,
    pitch_type: str,
    year: int,
    game_context: Dict[str, any],
    zone: int = None,
) -> Dict[str, float]:
    parquet_features = build_out_type_features_from_parquet(
        batter_id,
        pitcher_id,
        pitch_type,
        year,
        zone,
    )

    full_features = {**parquet_features, **game_context}

    full_features['pitch_type'] = pitch_type
    full_features['zone'] = zone

    df = pd.DataFrame([full_features])
    probs = build_out_type_probabilities(df)
    return probs
