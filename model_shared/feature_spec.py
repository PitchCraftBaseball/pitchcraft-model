# ! If you update the training notebooks, you need to update the features here.
FEATURE_SPEC = {
    "target": "y_next_pitch_type",
    "cat_cols": [
        "pitcher",
        "batter",
        "stand",
        "p_throws",
        "inning_topbot",
        "count_state",
        "prev_pitch_type",
    ],
    "num_cols": [
        "balls",
        "strikes",
        "outs_when_up",
        "inning",
        "score_diff_bat",
        "on_1b",
        "on_2b",
        "on_3b",
    ],
    "bool_cols": [],
}