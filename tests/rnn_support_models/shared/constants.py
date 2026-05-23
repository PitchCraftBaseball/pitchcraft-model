GAME_CONTEXT_FEATURES = [
    'balls',
    'strikes',
    'outs_when_up',
    'inning',
    'inning_topbot',
    'bat_score_diff',
    'on_1b',
    'on_2b',
    'on_3b',
    'pitch_type',
    'prev_pitch_type',
    'location',
]

PLAYER_FEATURES = [
    'p_throws',
    'stand',
    'sz_top',
    'sz_bot',
]

OUT_TYPE_HISTORICAL_FEATURES = [
    'batter_prev_whiff_percentage', 'batter_prev_gb_percentage', 'batter_prev_fb_percentage', 'batter_prev_chase_percentage',
    'batter_prev_weak_percentage', 'batter_prev_under_percentage', 'batter_prev_topped_percentage', 'batter_prev_flareburner_percentage', 'batter_prev_solid_percentage',
    'batter_prev_barrel_percentage', 'batter_prev_barrels_per_pa', 'batter_prev_looking_strike_percentage', 'batter_prev_zone_contact_percentage',
    'pitcher_prev_fb_percentage', 'pitcher_prev_gb_percentage', 'pitcher_prev_whiff_percentage', 'pitcher_prev_chase_percentage',
    'pitcher_prev_weak_percentage', 'pitcher_prev_under_percentage', 'pitcher_prev_topped_percentage', 'pitcher_prev_flareburner_percentage',
    'pitcher_prev_solid_percentage', 'pitcher_prev_barrel_percentage', 'pitcher_prev_barrels_per_pa',
]

OUT_TYPE_PITCH_FEATURES = [
    'pitcher_pitch_putaway_percentage', 'batter_pitch_putaway_percentage', 'pitcher_pitch_whiff_percentage', 'batter_pitch_whiff_percentage',
    'pitcher_pitch_average_launch_angle', 'pitcher_pitch_average_exit_velocity', 'pitcher_pitch_expected_batting_average',
    'batter_pitch_average_launch_angle', 'batter_pitch_average_exit_velocity', 'batter_pitch_expected_batting_average',
    'pitcher_pitch_batting_average', 'batter_pitch_batting_average', 'pitcher_pitch_average_mph', 'batter_pitch_average_mph',
]

OUT_TYPE_LOC_FEATURES = [
    'pitcher_loc_batting_average', 'batter_loc_batting_average', 'pitcher_loc_average_exit_velocity',
    'batter_loc_average_exit_velocity', 'pitcher_loc_average_launch_angle', 'batter_loc_average_launch_angle',
    'pitcher_loc_contact_batting_average', 'batter_loc_contact_batting_average', 'pitcher_loc_hard_hit_bip_percentage',
    'batter_loc_hard_hit_bip_percentage', 'batter_loc_strikeout_percentage', 'pitcher_loc_strikeout_percentage', 'batter_loc_whiff_percentage',
    'pitcher_loc_whiff_percentage', 'batter_loc_fly_ball_percentage', 'pitcher_loc_fly_ball_percentage', 'batter_loc_walk_percentage',
    'pitcher_loc_walk_percentage', 'batter_loc_ground_ball_percentage', 'pitcher_loc_ground_ball_percentage',
    'batter_loc_swing_percentage', 'pitcher_loc_swing_percentage', 'batter_loc_foul_percentage', 'pitcher_loc_foul_percentage'
]

TRANSITION_HISTORICAL_FEATURES = [
    'batter_prev_whiff_percentage', 'batter_prev_chase_percentage', 'batter_prev_looking_strike_percentage',
    'batter_prev_zone_contact_percentage', 'pitcher_prev_whiff_percentage', 'pitcher_prev_chase_percentage',
    'pitcher_prev_looking_strike_percentage', 'pitcher_prev_zone_contact_percentage', 'batter_prev_first_pitch_swing_percentage',
    'pitcher_prev_first_pitch_swing_percentage', 'pitcher_prev_meatball_swing_percentage', 'batter_prev_meatball_swing_percentage'
]

TRANSITION_PITCH_FEATURES = [
    'pitcher_pitch_putaway_percentage', 'batter_pitch_putaway_percentage', 
    'pitcher_pitch_whiff_percentage', 'batter_pitch_whiff_percentage',
    
]

TRANSITION_LOC_FEATURES = [
    'pitcher_loc_strikeout_percentage', 
    'batter_loc_strikeout_percentage',
    'batter_loc_whiff_percentage',
    'pitcher_loc_whiff_percentage',
    'batter_loc_walk_percentage',
    'pitcher_loc_walk_percentage',
    'batter_loc_swing_percentage',
    'pitcher_loc_swing_percentage',
    'batter_loc_foul_percentage',
    'pitcher_loc_foul_percentage',
]


VALID_GAME_CONTEXT = {
    'balls': 2,
    'strikes': 1,
    'stand': 'R',
    'p_throws': 'R',
    'inning': 6,
    'inning_topbot': 'Top',
    'bat_score_diff': 1,
    'on_1b': 1,
    'on_2b': 0,
    'on_3b': 1,
    'outs_when_up': 2,
    'prev_pitch_type': 'FF'
}

OUT_TYPE_FEATURES = (
    GAME_CONTEXT_FEATURES +
    PLAYER_FEATURES +
    OUT_TYPE_HISTORICAL_FEATURES + 
    OUT_TYPE_PITCH_FEATURES +
    OUT_TYPE_LOC_FEATURES
)

TRANSITION_FEATURES = (
    GAME_CONTEXT_FEATURES +
    PLAYER_FEATURES +
    TRANSITION_HISTORICAL_FEATURES + 
    TRANSITION_PITCH_FEATURES +
    TRANSITION_LOC_FEATURES
)