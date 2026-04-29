from model_shared.feature_engineering.pitch_constants import *
import pandas as pd

def pitch_to_family(pitch):
    if pd.isna(pitch):
        return None
    if pitch in FASTBALL:
        return 'fastball'
    if pitch in BREAKING:
        return 'breaking'
    if pitch in OFFSPEED:
        return 'offspeed'
    return None  # excludes rare/other pitch types from 


def count_situation(balls: int, strikes: int) -> str:
    """Return 'ahead', 'behind', or 'even' from the *pitcher's* perspective."""
    if strikes > balls:
        return "ahead"
    if balls > strikes:
        return "behind"
    return "even"


def calculate_pitch_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['is_swing'] = out['description'].isin(SWING_CODE)
    out['is_whiff'] = out['description'].isin(WHIFF_CODE)

    out["is_called_strike"] = (out["description"] == "called_strike")
    out["is_ball"] = out["description"].isin(["ball", "blocked_ball"])

    out['in_zone'] = (out['zone'] < 10)
    out['out_zone'] = (out['zone'] > 10)

    out["pitch_group"] = out["pitch_type"].map(
        lambda x: pitch_to_family(x)
    )
    return out

def calculate_game_state_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Make base occupancy into bools
    for base_col in ["on_1b", "on_2b", "on_3b"]:
        out[base_col] = out[base_col].notna()

    out['count_state'] = (out['balls'].astype(int).astype(str) + "-" + out['strikes'].astype(int).astype(str))

    out["count_situation"] = out.apply(
        lambda r: count_situation(int(r["balls"]), int(r["strikes"])), axis=1
    )

    return out

def pitcher_situation_lookup(df: pd.DataFrame) -> pd.DataFrame:
    hist = df[["pitcher", "balls", "strikes", "pitch_type", "description"]].copy()
    hist["count_situation"] = hist.apply(
        lambda r: count_situation(int(r["balls"]), int(r["strikes"])), axis=1
    )
    hist["is_fastball"] = hist["pitch_type"].isin(FASTBALL)
    hist["is_breaking"] = hist["pitch_type"].isin(BREAKING)
    hist["is_offspeed"] = hist["pitch_type"].isin(OFFSPEED)
    hist["is_whiff"]    = hist["description"].isin(WHIFF_CODE)

    return (
        hist.groupby(["pitcher", "count_situation"])
        .agg(
            pitcher_sit_n          = ("pitch_type", "size"),
            pitcher_sit_fb_rate    = ("is_fastball", "mean"),
            pitcher_sit_br_rate    = ("is_breaking", "mean"),
            pitcher_sit_os_rate    = ("is_offspeed", "mean"),
            pitcher_sit_whiff_rate = ("is_whiff", "mean"),
        )
        .reset_index()
    )


def batter_situation_lookup(df: pd.DataFrame) -> pd.DataFrame:
    hist = df[["batter", "balls", "strikes", "pitch_type", "description"]].copy()
    hist["count_situation"] = hist.apply(
        lambda r: count_situation(int(r["balls"]), int(r["strikes"])), axis=1
    )
    hist["is_swing"] = hist["description"].isin(SWING_CODE)
    hist["is_whiff"] = hist["description"].isin(WHIFF_CODE)

    return (
        hist.groupby(["batter", "count_situation"])
        .agg(
            batter_sit_n           = ("pitch_type", "size"),
            batter_sit_swing_rate  = ("is_swing", "mean"),
            batter_sit_whiff_rate  = ("is_whiff", "mean"),
        )
        .reset_index()
    )


def add_pitcher_count_split_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds
    pitcher_sit_fb_rate    Fastball rate when pitcher is in this situation
    pitcher_sit_br_rate    Breaking rate
    pitcher_sit_os_rate    Offspeed rate
    pitcher_sit_whiff_rate Whiff rate
    pitcher_sit_n          Number of historical pitches in this situation
    """
    out = df.copy()

    if "count_situation" not in out.columns:
        out = calculate_game_state_features(out)

    # Drop any pre-existing output columns so re-running this cell is safe
    pitcher_sit_cols = ["pitcher_sit_n", "pitcher_sit_fb_rate", "pitcher_sit_br_rate",
                        "pitcher_sit_os_rate", "pitcher_sit_whiff_rate"]
    out = out.drop(columns=[c for c in pitcher_sit_cols if c in out.columns])

    lookup = pitcher_situation_lookup(df)
    out = out.merge(lookup, on=["pitcher", "count_situation"], how="left")
    return out


def add_batter_count_split_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds
    batter_sit_swing_rate   How often batter swings in this situation
    batter_sit_whiff_rate   Whiff rate
    batter_sit_n            Historical pitch count in this situation
    """
    out = df.copy()

    if "count_situation" not in out.columns:
        out = calculate_game_state_features(out)

    # Drop any pre-existing output columns so re-running this cell is safe
    batter_sit_cols = ["batter_sit_n", "batter_sit_swing_rate", "batter_sit_whiff_rate"]
    out = out.drop(columns=[c for c in batter_sit_cols if c in out.columns])

    lookup = batter_situation_lookup(df)
    out = out.merge(lookup, on=["batter", "count_situation"], how="left")
    return out

def get_vs_pitcher_stats(batter_id: int, pitcher_id: int, df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["batter"] == batter_id) & (df["pitcher"] == pitcher_id)

    history = df[mask]
    if history.empty:
        return pd.DataFrame()
    
    return pd.DataFrame([{
        "n_pitches": len(history),
        "swing_rate": history["description"].isin(SWING_CODE).mean(),
        "whiff_rate": history["description"].isin(WHIFF_CODE).mean(),
        "called_k_rate": (history["description"] == "called_strike").mean(),
    }])

def situational_split(pitcher_id: int, df: pd.DataFrame) -> pd.DataFrame:
    mask = df["pitcher"] == pitcher_id
    history = df[mask].copy()

    if history.empty:
        return pd.DataFrame()
    
    history["count_situation"] = history.apply(
        lambda r: count_situation(int(r["balls"]), int(r["strikes"])), axis=1
    )
    history["is_fastball"] = history["pitch_type"].isin(FASTBALL)
    history["is_breaking"] = history["pitch_type"].isin(BREAKING)
    history["is_offspeed"] = history["pitch_type"].isin(OFFSPEED)
    history["is_swing"]    = history["description"].isin(SWING_CODE)
    history["is_whiff"]    = history["description"].isin(WHIFF_CODE)
    history["is_called_k"] = history["description"] == "called_strike"

    splits = (
            history.groupby("count_situation")
            .agg(
                n_pitches      = ("pitch_type", "size"),
                fastball_rate  = ("is_fastball", "mean"),
                breaking_rate  = ("is_breaking", "mean"),
                offspeed_rate  = ("is_offspeed", "mean"),
                swing_rate     = ("is_swing", "mean"),
                whiff_rate     = ("is_whiff", "mean"),
                called_k_rate  = ("is_called_k", "mean"),
            )
            .reset_index()
        )
    splits.insert(0, "pitcher", pitcher_id)
    return splits