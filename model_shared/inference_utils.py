from __future__ import annotations
import pandas as pd

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
