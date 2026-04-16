from __future__ import annotations

import pandas as pd
import numpy as np
from model_shared.feature_engineering.pitch_constants import *

def compute_bucket_boundaries(train_df):
    """
    Only needed for horizontal now — vertical uses sz_top/sz_bot directly.
    """
    loc_data = train_df[train_df["plate_x"].notna()].copy()
    
    return {
        "x_low":  loc_data["plate_x"].quantile(0.33),
        "x_high": loc_data["plate_x"].quantile(0.67),
    }

def get_pitch_location_buckets(
    plate_x: float,
    plate_z: float,
    sz_top:  float,
    sz_bot:  float,
    stand:   str,
    boundaries: dict,
) -> dict | None:
    """
    Returns the horizontal bucket, vertical bucket, and
    zone classification for a single pitch.

    Args:
        plate_x:    horizontal plate position (catcher's perspective)
        plate_z:    vertical plate position in feet
        sz_top:     top of batter's strike zone
        sz_bot:     bottom of batter's strike zone
        stand:      batter handedness, "L" or "R"
        boundaries: dict from compute_bucket_boundaries()
                    containing "x_low" and "x_high"

    Returns:
        {
            "horiz_bucket": 0 (away) | 1 (middle) | 2 (in),
            "vert_bucket":  0 (low)  | 1 (middle) | 2 (up),
            "in_zone":      True | False,
        }
        or None if any required value is missing.
    """
    if any(pd.isna(v) for v in [plate_x, plate_z, sz_top, sz_bot]):
        return None

    return {
        "horiz_bucket": _get_horiz_bucket(plate_x, stand, boundaries),
        "vert_bucket":  _get_vert_bucket(plate_z, sz_top, sz_bot),
        "in_zone":      _get_zone(plate_x, plate_z, sz_top, sz_bot),
    }


def _get_horiz_bucket(
    plate_x:    float,
    stand:      str,
    boundaries: dict,
) -> int:
    """
    Returns horizontal bucket from the catcher's perspective.
        0 = away
        1 = middle
        2 = in
    """
    x = plate_x if stand == "R" else -plate_x

    if x < boundaries["x_low"]:
        return 0  # away
    elif x > boundaries["x_high"]:
        return 2  # in
    else:
        return 1  # middle


def _get_vert_bucket(
    plate_z: float,
    sz_top:  float,
    sz_bot:  float,
) -> int:
    """
    Returns vertical bucket relative to batter's strike zone.
        0 = low
        1 = middle
        2 = up
    """
    zone_height = sz_top - sz_bot
    low_thresh  = sz_bot + zone_height * (1 / 3)
    high_thresh = sz_bot + zone_height * (2 / 3)

    if plate_z < low_thresh:
        return 0  # low
    elif plate_z > high_thresh:
        return 2  # up
    else:
        return 1  # middle


def _get_zone(
    plate_x: float,
    plate_z: float,
    sz_top:  float,
    sz_bot:  float,
) -> bool:
    """
    Returns True if pitch is within the strike zone.
    Uses standard Statcast zone width of 0.83 feet
    on each side of center.
    """
    ZONE_HALF_WIDTH = 0.83

    in_horizontal = abs(plate_x) <= ZONE_HALF_WIDTH
    in_vertical   = sz_bot <= plate_z <= sz_top

    return in_horizontal and in_vertical

def add_location_targets(df: pd.DataFrame, boundaries: dict) -> pd.DataFrame:
    out = df.copy()

    missing = (
        out["plate_x"].isna() |
        out["plate_z"].isna() |
        out["sz_top"].isna()  |
        out["sz_bot"].isna()
    )

    # horiz bucket
    x = np.where(out["stand"] == "R", out["plate_x"], -out["plate_x"])
    out["horiz_bucket"] = np.select(
        [x < boundaries["x_low"], x > boundaries["x_high"]],
        [0, 2],
        default=1
    ).astype(float)  # float allows NaN

    # vert bucket
    zone_h     = out["sz_top"] - out["sz_bot"]
    low_thresh = out["sz_bot"] + zone_h * (1/3)
    hi_thresh  = out["sz_bot"] + zone_h * (2/3)
    out["vert_bucket"] = np.select(
        [out["plate_z"] < low_thresh, out["plate_z"] > hi_thresh],
        [0, 2],
        default=1
    ).astype(float)  # float allows NaN

    # in_zone — cast to float before assignment so NaN is valid
    out["in_zone"] = (
        (out["plate_x"].abs() <= 0.83) &
        (out["plate_z"] >= out["sz_bot"]) &
        (out["plate_z"] <= out["sz_top"])
    ).astype(float)  # float allows NaN

    # null out rows with missing location data
    out.loc[missing, ["horiz_bucket", "vert_bucket", "in_zone"]] = np.nan

    return out

def add_prev_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds previous pitch location features within each PA.
    Must be called after add_location_targets() so that
    horiz_bucket, vert_bucket, and in_zone already exist.
    """
    out = df.copy()

    out["prev_horiz_bucket"] = (
        out.groupby("pa_id")["horiz_bucket"]
        .shift(1)
        .fillna(PAD_ID)
        .astype(int)
    )
    out["prev_vert_bucket"] = (
        out.groupby("pa_id")["vert_bucket"]
        .shift(1)
        .fillna(PAD_ID)
        .astype(int)
    )
    out["prev_in_zone"] = (
        out.groupby("pa_id")["in_zone"]
        .shift(1)
        .astype(float)
        .fillna(0.0)
    )

    return out