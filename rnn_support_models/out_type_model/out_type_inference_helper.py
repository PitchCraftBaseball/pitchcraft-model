from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd
from model_shared.inference_utils import prepare_inference_data
from model_shared import feature_tables


LOC_METRICS = (
    'batting_average', 'average_exit_velocity',
    'average_launch_angle', 'contact_batting_average',
    'hard_hit_bip_percentage', 'expected_batting_average',
    'strikeout_percentage', 'whiff_percentage', 'walk_percentage', 'ground_ball_percentage',
    'line_drive_percentage', 'fly_ball_percentage', 'popup_percentage', 'swing_percentage', 'foul_percentage'
)

_MODELS_DIR = Path(__file__).parent / 'models'

stage1_data = joblib.load(_MODELS_DIR / 'pa_end_model_v2.pkl')
stage2_data = joblib.load(_MODELS_DIR / 'so_model_v2.pkl')
stage3_data = joblib.load(_MODELS_DIR / 'bip_model_v2.pkl')
stage4_data = joblib.load(_MODELS_DIR / 'fb_model_v2.pkl')

pa_end_model = stage1_data['model']
pa_end_features = stage1_data['features']

so_model = stage2_data['model']
so_features = stage2_data['features']

bip_model = stage3_data['model']
bip_features = stage3_data['features']

fb_model = stage4_data['model']
fb_features = stage4_data['features']


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
    location: int = None,
) -> Dict[str, Any]:
    batter_prev_raw = feature_tables.fetch_player_out_type_historical_features(
        batter_id, year, pitch_type, is_batter=True,
    )
    pitcher_prev_raw = feature_tables.fetch_player_out_type_historical_features(
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
        'ground_ball_percentage': 'gb_percentage',
        'air_ball_percentage': 'fb_percentage',
        'whiff_percentage': 'whiff_percentage',
        'chase_percentage': 'chase_percentage',
        'weak_percentage': 'weak_percentage',
        'under_percentage': 'under_percentage',
        'topped_percentage': 'topped_percentage',
        'flareburner_percentage': 'flareburner_percentage',
        'solid_percentage': 'solid_percentage',
        'barrel_percentage': 'barrel_percentage',
        'barrels_per_pa': 'barrels_per_pa',
        'zone_contact_percentage': 'zone_contact_rate',
    }

    pitch_rename_cols = {
        'batting_average': 'batting_average',
        'putaway_percentage': 'putaway_rate',
        'pitch_whiff_percentage': 'whiff_percentage',
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

    def _map_loc(loc_dict, prefix, target_loc):
        for metric in LOC_METRICS:
            remapped = f'{prefix}_loc_{metric}'
            raw[remapped] = loc_dict.get(f'{metric}_loc{target_loc}')

    _map_loc(batter_loc_raw, 'batter', location)
    _map_loc(pitcher_loc_raw, 'pitcher', location)

    batter_zone_pct = float(batter_prev_raw.get('zone_percentage') or 0)
    batter_swing_pct = float(batter_prev_raw.get('zone_swing_percentage') or 0)
    raw['batter_prev_looking_strike_percentage'] = batter_zone_pct * (1 - batter_swing_pct / 100.0)

    raw['batter_loc_fly_ball_percentage'] = (
        float(raw.get('batter_loc_line_drive_percentage') or 0) +
        float(raw.get('batter_loc_fly_ball_percentage') or 0) +
        float(raw.get('batter_loc_popup_percentage') or 0)
    )
    raw['pitcher_loc_fly_ball_percentage'] = (
        float(raw.get('pitcher_loc_line_drive_percentage') or 0) +
        float(raw.get('pitcher_loc_fly_ball_percentage') or 0) +
        float(raw.get('pitcher_loc_popup_percentage') or 0)
    )

    return {k: (float(v) if v is not None else 0.0) for k, v in raw.items()}


def predict_pitch_out_type_outcome(
    batter_id: str,
    pitcher_id: str,
    pitch_type: str,
    year: int,
    game_context: Dict[str, any],
    location: int = None,
) -> Dict[str, float]:

    if None in (batter_id, pitcher_id, pitch_type, year):
        raise ValueError(
            'batter_id, pitcher_id, pitch_type, and year are all required'
        )

    parquet_features = build_out_type_features_from_parquet(
        batter_id,
        pitcher_id,
        pitch_type,
        year,
        location,
    )

    full_features = {**parquet_features, **game_context}

    full_features['pitch_type'] = pitch_type
    full_features['location'] = location

    df = pd.DataFrame([full_features])
    probs = build_out_type_probabilities(df)
    return probs
