import pandas as pd

def sort_statcast(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["game_date", "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"], ascending=[True, True, True, False, True, True]).reset_index(drop=True)

def universal_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out = out[out["game_type"] == 'R']
    out = out[out["pitch_type"] != 'UN']
    
    out["pa_id"] = out["game_pk"].astype(str) + "_" + out["at_bat_number"].astype(str)

    # Handling if previous pitch in sequence was ABS 
    abs_label = ["ABS"]

    is_abs = out["pitch_type"].isin(abs_label)
    out["pitch_type_for_prev"] = out["pitch_type"].mask(is_abs)

    # Track last real pitch type within each PA, will be NaN until first pitch is actually thrown
    out["last_real_pitch_type"] = (out.groupby("pa_id")["pitch_type_for_prev"].ffill())

    # Previous pitch type is the last real pitch but shifted 1. NaNs are now filled
    out["prev_pitch_type"] = (out.groupby("pa_id")["last_real_pitch_type"].shift(1).fillna("START"))

    # don't care about thse columns
    out = out.drop(columns=["pitch_type_for_prev", "last_real_pitch_type"])

    out["seq_len"] = out.groupby("pa_id")["pitch_type"].transform("size")

    return out

def data_remapping(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    pitch_remapping = {
        'SC': 'CU',   # screwball -> curveball family
        'CS': 'CU',   # slow curve -> curveball
        'FO': 'FS'
    }

    out['pitch_type'] = out['pitch_type'].replace(pitch_remapping)

    abs_remapping = {
        'automatic_ball': 'ABS',
        'automatic_strike': 'ABS',
    }
    out['pitch_type'] = (out['description'].map(abs_remapping).fillna(out['pitch_type']))

    return out


def drop_unused_cols(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    drop_cols = [
        "player_name", "events", "sv_id", "umpire", "des",
        "spin_dir", "spin_rate_deprecated", "break_angle_deprecated", "break_length_deprecated", "game_type", "home_team", "away_team", "type", "hit_location", "bb_type", 
        "hc_x", "hc_y", "tfs_deprecated", "tfs_zulu_deprecated", "hit_distance_sc", "launch_speed", "launch_angle", "effective_speed", 
        "fielder_2", "fielder_3", "fielder_4",	"fielder_5", "fielder_6", "fielder_7", "fielder_8",	"fielder_9",
        "estimated_ba_using_speedangle", "estimated_woba_using_speedangle", "estimated_slg_using_speedangle", "woba_value", "woba_denom", "babip_value", "iso_value", "launch_speed_angle",
        "if_fielding_alignment", "of_fielding_alignment", "delta_home_win_exp", "hyper_speed", "bat_speed", "swing_length", "home_win_exp", "bat_win_exp", 
        "age_pit_legacy", "age_bat_legacy", "age_pit", "age_bat", "attack_angle", "attack_direction", "swing_path_tilt", 
        "intercept_ball_minus_batter_pos_x_inches", "intercept_ball_minus_batter_pos_y_inches", "Unnamed: 0", 
        "delta_run_exp", "delta_pitcher_run_exp", "batter_days_until_next_game", "api_break_z_with_gravity", "api_break_x_arm", "api_break_x_batter_in", "batter_days_until_next_game",
        "pitcher_days_until_next_game", "batter_days_since_prev_game", "pitcher_days_since_prev_game", "n_priorpa_thisgame_player_at_bat", "n_thruorder_pitcher", 
        "vx0", "vy0", "vz0", "ax", "ay", "az", "release_spin_rate", "spin_axis", "arm_angle", 'release_pos_x', 'release_pos_z', 'release_extension', 'release_pos_y',
        'post_away_score','post_home_score', 'post_bat_score', 'post_fld_score',
    ]
    
    clean_data = out.drop(columns=[c for c in drop_cols], errors="ignore")

    return clean_data

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out = sort_statcast(out)
    out = universal_features(out)
    out = data_remapping(out)
    out = drop_unused_cols(out)
    return out 
