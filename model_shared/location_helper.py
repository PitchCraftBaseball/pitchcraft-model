import pandas as pd
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.preprocessing import MinMaxScaler

from model_shared.feature_engineering.pitch_constants import BASE_LABELS

_BASE_STATE_DECODE = {v: k for k, v in BASE_LABELS.items()}

ZONE_HALF_WIDTH = 17 / 12 / 2
BUCKETS = [
    'high_in_zone', 'high_away_zone', 'low_in_zone', 'low_away_zone',
    'high_in_chase', 'high_away_chase', 'low_in_chase', 'low_away_chase',
]
BUCKET_TO_ZONE: dict = {bucket: i + 1 for i, bucket in enumerate(BUCKETS)}
BATTER_MIN = 20
PITCHER_MIN = 30
FALLBACK_LOCATION = 'low_away_chase'


@dataclass
class LocationContext:
    pitcher_row: pd.Series
    batter_row: pd.Series
    pitcher_id: int
    pt_bucket_stats: pd.DataFrame
    league_avg: dict
    re24: dict
    RE_MAX: float


@dataclass
class GlobalLocationContext:
    """Per-process state needed to score any (pitcher, batter) pair. Build
    once via build_global_location_context(); slice into a per-pair
    LocationContext via pair_location_context().
    """
    pitcher_scaled: pd.DataFrame
    batter_scaled: pd.DataFrame
    pt_bucket_stats: pd.DataFrame
    league_avg: dict
    re24: dict
    RE_MAX: float


def sort_statcast(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["game_type"] == 'R']
    df = df[df["pitch_type"] != 'UN']
    return df.sort_values(
        ["game_date", "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"],
        ascending=[True, True, True, False, True, True]
    ).reset_index(drop=True)


def assign_buckets_vec(df: pd.DataFrame) -> pd.Series:
    px  = df['plate_x']
    pz  = df['plate_z']
    top = df['sz_top']
    bot = df['sz_bot']
    std = df['stand']

    has_data = px.notna() & pz.notna() & top.notna() & bot.notna() & std.notna()

    mid_z     = (top + bot) / 2
    is_high   = pz >= mid_z
    is_inside = np.where(std == 'R', px < 0, px >= 0)
    in_zone   = (
        (px >= -ZONE_HALF_WIDTH) & (px <= ZONE_HALF_WIDTH) &
        (pz >= bot) & (pz <= top)
    )

    vert   = np.where(is_high,   'high', 'low')
    horiz  = np.where(is_inside, 'in',   'away')
    loc    = np.where(in_zone,   'zone', 'chase')

    bucket = pd.Series(vert + '_' + horiz + '_' + loc, index=df.index, dtype=object)
    bucket[~has_data] = np.nan
    return bucket


def _add_pitch_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['whiff'] = df['description'].isin([
        'swinging_strike', 'swinging_strike_blocked', 'foul_tip'
    ]).astype(int)
    df['swing'] = df['description'].isin([
        'foul_bunt', 'foul', 'hit_into_play', 'swinging_strike', 'foul_tip',
        'swinging_strike_blocked', 'missed_bunt', 'bunt_foul_tip'
    ]).astype(int)
    df['gb']  = (df['bb_type'] == 'ground_ball').astype(int)
    df['fly'] = (df['bb_type'].isin(['fly_ball', 'popup'])).astype(int)
    df['bip'] = df['bb_type'].notna().astype(int)
    return df


def _build_league_stats(pitch_df: pd.DataFrame) -> dict:
    lg = pitch_df.groupby('bucket').agg(
        pitches = ('pitch_type', 'count'),
        swings  = ('swing', 'sum'),
        whiffs  = ('whiff', 'sum'),
        bip     = ('bip', 'sum'),
        gb      = ('gb', 'sum'),
    ).reset_index()

    lg['lg_whiff_rate']   = lg['whiffs'] / lg['swings'].clip(lower=1)
    lg['lg_gb_rate']      = lg['gb']     / lg['bip'].clip(lower=1)
    lg['lg_contact_rate'] = 1 - lg['lg_whiff_rate']
    lg['lg_usage_rate']   = lg['pitches'] / lg['pitches'].sum()

    chase_lg = (
        pitch_df[pitch_df['bucket'].str.contains('chase')]
        .groupby('bucket')
        .agg(chases=('swing', 'sum'), ooz=('pitch_type', 'count'))
        .reset_index()
    )
    chase_lg['lg_chase_rate'] = chase_lg['chases'] / chase_lg['ooz'].clip(lower=1)

    lg = lg.merge(chase_lg[['bucket', 'lg_chase_rate']], on='bucket', how='left')
    lg['lg_chase_rate'] = lg['lg_chase_rate'].fillna(0)

    return lg.set_index('bucket').to_dict('index')


def _build_batter_stats(pitch_df: pd.DataFrame, league_avg: dict) -> pd.DataFrame:
    b_raw = pitch_df.groupby(['batter', 'bucket']).agg(
        pitches = ('pitch_type', 'count'),
        swings  = ('swing', 'sum'),
        whiffs  = ('whiff', 'sum'),
        bip     = ('bip', 'sum'),
        gb      = ('gb', 'sum'),
    ).reset_index()

    chase_b = (
        pitch_df[pitch_df['bucket'].str.contains('chase')]
        .groupby(['batter', 'bucket'])
        .agg(chases=('swing', 'sum'), ooz=('pitch_type', 'count'))
        .reset_index()
    )
    chase_b['raw_chase_rate'] = chase_b['chases'] / chase_b['ooz'].clip(lower=1)
    b_raw = b_raw.merge(chase_b[['batter', 'bucket', 'raw_chase_rate']], on=['batter', 'bucket'], how='left')

    b_raw['raw_whiff_rate']   = b_raw['whiffs'] / b_raw['swings'].clip(lower=1)
    b_raw['raw_contact_rate'] = 1 - b_raw['raw_whiff_rate']
    b_raw['raw_gb_rate']      = b_raw['gb']     / b_raw['bip'].clip(lower=1)

    def _fallback(row):
        lg = league_avg[row['bucket']]
        ok = row['pitches'] >= BATTER_MIN
        return pd.Series({
            'whiff_rate':   row['raw_whiff_rate']   if ok else lg['lg_whiff_rate'],
            'contact_rate': row['raw_contact_rate'] if ok else lg['lg_contact_rate'],
            'gb_rate':      row['raw_gb_rate']      if ok else lg['lg_gb_rate'],
            'chase_rate':   row['raw_chase_rate']   if (ok and pd.notna(row.get('raw_chase_rate'))) else lg['lg_chase_rate'],
        })

    fb_cols = b_raw.apply(_fallback, axis=1)
    return pd.concat([b_raw[['batter', 'bucket', 'pitches']], fb_cols], axis=1)


def _build_pitcher_stats(pitch_df: pd.DataFrame, league_avg: dict):
    p_raw = pitch_df.groupby(['pitcher', 'bucket']).agg(
        pitches = ('pitch_type', 'count'),
        swings  = ('swing', 'sum'),
        whiffs  = ('whiff', 'sum'),
        bip     = ('bip', 'sum'),
        gb      = ('gb', 'sum'),
    ).reset_index()

    pitcher_totals = pitch_df.groupby('pitcher')['pitch_type'].count().reset_index(name='total_pitches')
    p_raw = p_raw.merge(pitcher_totals, on='pitcher', how='left')

    p_raw['raw_whiff_rate'] = p_raw['whiffs'] / p_raw['swings'].clip(lower=1)
    p_raw['raw_gb_rate']    = p_raw['gb']     / p_raw['bip'].clip(lower=1)
    p_raw['raw_usage_rate'] = p_raw['pitches'] / p_raw['total_pitches'].clip(lower=1)

    def _fallback(row):
        lg = league_avg[row['bucket']]
        ok = row['pitches'] >= PITCHER_MIN
        return pd.Series({
            'whiff_rate': row['raw_whiff_rate'] if ok else lg['lg_whiff_rate'],
            'gb_rate':    row['raw_gb_rate']    if ok else lg['lg_gb_rate'],
            'usage_rate': row['raw_usage_rate'],
        })

    pf_cols = p_raw.apply(_fallback, axis=1)
    pitcher_bucket_stats = pd.concat([p_raw[['pitcher', 'bucket', 'pitches']], pf_cols], axis=1)

    pt_raw = pitch_df.groupby(['pitcher', 'pitch_type', 'bucket']).agg(
        pitches = ('description', 'count'),
        swings  = ('swing', 'sum'),
        whiffs  = ('whiff', 'sum'),
    ).reset_index()

    pt_raw['pt_whiff_rate'] = pt_raw['whiffs'] / pt_raw['swings'].clip(lower=1)
    pt_bucket_stats = pt_raw[pt_raw['pitches'] >= 10].copy()

    return pitcher_bucket_stats, pt_bucket_stats


def _pivot_wide(df: pd.DataFrame, id_col: str, feature_cols: list) -> pd.DataFrame:
    result = df[[id_col]].drop_duplicates()
    for feat in feature_cols:
        piv = df.pivot_table(index=id_col, columns='bucket', values=feat)
        piv.columns = [f'{col}_{feat}' for col in piv.columns]
        result = result.merge(piv.reset_index(), on=id_col, how='left')
    return result


def _scale_features(batter_bucket_stats: pd.DataFrame, pitcher_bucket_stats: pd.DataFrame):
    batter_loc  = _pivot_wide(batter_bucket_stats,  'batter',  ['whiff_rate', 'contact_rate', 'chase_rate', 'gb_rate'])
    pitcher_loc = _pivot_wide(pitcher_bucket_stats, 'pitcher', ['whiff_rate', 'gb_rate', 'usage_rate'])

    batter_feat_cols  = [c for c in batter_loc.columns  if c != 'batter']
    pitcher_feat_cols = [c for c in pitcher_loc.columns if c != 'pitcher']

    batter_scaled  = batter_loc.copy()
    pitcher_scaled = pitcher_loc.copy()

    batter_scaled[batter_feat_cols]   = MinMaxScaler().fit_transform(batter_loc[batter_feat_cols].fillna(0))
    pitcher_scaled[pitcher_feat_cols] = MinMaxScaler().fit_transform(pitcher_loc[pitcher_feat_cols].fillna(0))

    return batter_scaled, pitcher_scaled


# Signed pressure: positive = pitcher behind (needs strikes), negative = pitcher ahead.
# Magnitude drives how much we up-weight the count term.
_COUNT_PRESSURE: dict = {
    (3, 0):  1.00,
    (3, 1):  0.80,
    (2, 0):  0.65,
    (3, 2):  0.50,
    (2, 1):  0.35,
    (1, 0):  0.20,
    (0, 0):  0.00,
    (1, 1):  0.00,
    (2, 2):  0.00,
    (0, 1): -0.30,
    (1, 2): -0.55,
    (0, 2): -0.75,
}


def _count_adj(count: tuple, bucket: str) -> float:
    """Zone-favoring score [0, 1]. 1 = strongly prefer zone; 0 = strongly prefer chase."""
    pressure = _COUNT_PRESSURE.get(count, 0.0)
    is_chase = 'chase' in bucket
    if pressure > 0:
        # Behind: chase is disqualifying, zone value scales with pressure
        return 0.0 if is_chase else 0.5 + 0.5 * pressure
    elif pressure < 0:
        # Ahead: expand the zone
        abs_p = abs(pressure)
        return (0.5 + 0.5 * abs_p) if is_chase else (0.5 - 0.3 * abs_p)
    else:
        return 0.5


def _count_weight(count: tuple) -> float:
    """Count term weight in [0.05, 0.30]. Grows when pitcher is strongly behind or ahead."""
    return 0.05 + 0.25 * abs(_COUNT_PRESSURE.get(count, 0.0))


def _pt_affinity(
    pitcher_id: int,
    pitch_type: str,
    bucket: str,
    pt_bucket_stats: pd.DataFrame,
    league_avg: dict,
) -> Optional[float]:
    mask = (
        (pt_bucket_stats['pitcher'] == pitcher_id) &
        (pt_bucket_stats['pitch_type'] == pitch_type) &
        (pt_bucket_stats['bucket'] == bucket)
    )
    rows = pt_bucket_stats[mask]
    if rows.empty:
        return None
    lg_whiff = league_avg.get(bucket, {}).get('lg_whiff_rate', 0.25)
    raw = float(rows.iloc[0]['pt_whiff_rate'])
    return min(raw / max(lg_whiff, 0.01), 2.0) / 2.0


def calculate_location_scores(
    pitcher_row,
    batter_row,
    pitcher_id: int,
    pitch_type: str,
    re24: dict,
    base_state: str,
    outs: int,
    count: tuple,
    RE_MAX: float,
    pt_bucket_stats: pd.DataFrame,
    league_avg: dict,
    target_out_type: Optional[str] = None,
) -> dict:
    re_weight = 0.20
    scores = {}
    balls, strikes = count
    count_key = f'{balls}-{strikes}'

    count_w = _count_weight(count)
    base_scale = (1.0 - count_w) / 0.95  # keeps whiff/affinity weights proportional as count_w grows

    for bucket in BUCKETS:
        p_whiff = float(pitcher_row.get(f'{bucket}_whiff_rate', 0) or 0)
        b_whiff = float(batter_row.get(f'{bucket}_whiff_rate',  0) or 0)

        pt_aff   = _pt_affinity(pitcher_id, pitch_type, bucket, pt_bucket_stats, league_avg)
        affinity = pt_aff if pt_aff is not None else p_whiff

        re_node = re24.get(str(outs), {}).get(base_state, {}).get(count_key, {}).get(pitch_type)
        re_raw  = re_node['run_expectancy'] if isinstance(re_node, dict) else RE_MAX * 0.5
        re_score = 1 - re_raw / RE_MAX

        cadj = _count_adj(count, bucket)

        if target_out_type == "groundout":
            p_gb = float(pitcher_row.get(f'{bucket}_gb_rate', 0) or 0)
            b_gb = float(batter_row.get(f'{bucket}_gb_rate',  0) or 0)
            non_re = (
                base_scale * 0.35 * p_gb +
                base_scale * 0.35 * b_gb +
                base_scale * 0.25 * affinity +
                count_w * cadj
            )
        elif target_out_type == "flyout":
            p_fly = 1.0 - float(pitcher_row.get(f'{bucket}_gb_rate', 0.5) or 0.5)
            b_fly = 1.0 - float(batter_row.get(f'{bucket}_gb_rate',  0.5) or 0.5)
            non_re = (
                base_scale * 0.35 * p_fly +
                base_scale * 0.35 * b_fly +
                base_scale * 0.25 * affinity +
                count_w * cadj
            )
        else:
            non_re = (
                base_scale * 0.35 * p_whiff +
                base_scale * 0.35 * b_whiff +
                base_scale * 0.25 * affinity +
                count_w * cadj
            )

        scores[bucket] = round((1 - re_weight) * non_re + re_weight * re_score, 4)

    return scores


def build_global_location_context() -> GlobalLocationContext:
    """Build the per-process location context. Heavy: reads the parquet,
    runs league-wide aggregations, fits MinMaxScalers. Call once at app
    startup; per-request work then becomes a row lookup via
    pair_location_context().
    """
    parquet_path = Path(__file__).parent.parent / "data" / "historical_pitches.parquet"
    df = pd.read_parquet(parquet_path)
    df = df[df["game_year"] == 2025]
    pitch_df = sort_statcast(df)
    pitch_df = _add_pitch_metrics(pitch_df)
    pitch_df['bucket'] = assign_buckets_vec(pitch_df)
    pitch_df = pitch_df[pitch_df['bucket'].notna()].copy()

    re_path = Path(__file__).parent.parent / "model_server" / "src" / "util" / "re288.json"
    with open(re_path) as f:
        re24 = json.load(f)

    RE_MAX = max(
        re24[str(o)][bs][ck][pt]['run_expectancy']
        for o in range(3)
        for bs in re24.get(str(o), {})
        for ck in re24[str(o)][bs]
        for pt in re24[str(o)][bs][ck]
        if isinstance(re24[str(o)][bs][ck].get(pt), dict)
    )

    league_avg = _build_league_stats(pitch_df)
    batter_bucket_stats = _build_batter_stats(pitch_df, league_avg)
    pitcher_bucket_stats, pt_bucket_stats = _build_pitcher_stats(pitch_df, league_avg)
    batter_scaled, pitcher_scaled = _scale_features(batter_bucket_stats, pitcher_bucket_stats)

    return GlobalLocationContext(
        pitcher_scaled=pitcher_scaled,
        batter_scaled=batter_scaled,
        pt_bucket_stats=pt_bucket_stats,
        league_avg=league_avg,
        re24=re24,
        RE_MAX=RE_MAX,
    )


def pair_location_context(
    global_ctx: GlobalLocationContext, pitcher, batter
) -> Optional[LocationContext]:
    """Per-request slice of the global context. Returns None if either
    player has no data in the dataset (matching the old precompute_*
    behavior).
    """
    pitcher_id = int(pitcher)
    batter_id  = int(batter)

    p_rows = global_ctx.pitcher_scaled[global_ctx.pitcher_scaled['pitcher'] == pitcher_id]
    b_rows = global_ctx.batter_scaled[global_ctx.batter_scaled['batter']   == batter_id]

    if p_rows.empty or b_rows.empty:
        return None

    return LocationContext(
        pitcher_row=p_rows.iloc[0],
        batter_row=b_rows.iloc[0],
        pitcher_id=pitcher_id,
        pt_bucket_stats=global_ctx.pt_bucket_stats,
        league_avg=global_ctx.league_avg,
        re24=global_ctx.re24,
        RE_MAX=global_ctx.RE_MAX,
    )


_default_global_ctx: Optional[GlobalLocationContext] = None


def precompute_location_context(pitcher, batter) -> Optional[LocationContext]:
    """Backwards-compatible per-pair builder. Builds the global context on
    first use and caches it for the process. Server code should call
    build_global_location_context() at startup and pair_location_context()
    per request instead, to avoid the lazy-build cost on the first request.
    """
    global _default_global_ctx
    if _default_global_ctx is None:
        _default_global_ctx = build_global_location_context()
    return pair_location_context(_default_global_ctx, pitcher, batter)


def get_optimal_location_from_context(
    context: Optional[LocationContext],
    pitch_type: str,
    state_features: dict,
    target_out_type: Optional[str] = None,
) -> str:
    """Lightweight scoring call — no I/O. Expects a context from precompute_location_context."""
    if context is None:
        return FALLBACK_LOCATION

    outs    = int(state_features['outs_when_up'])
    balls   = int(state_features['balls'])
    strikes = int(state_features['strikes'])

    base_state_int = (
        int(bool(state_features.get("on_1b", 0)))     +
        int(bool(state_features.get("on_2b", 0))) * 2 +
        int(bool(state_features.get("on_3b", 0))) * 4
    )
    base_state = _BASE_STATE_DECODE[base_state_int]

    scores = calculate_location_scores(
        context.pitcher_row, context.batter_row, context.pitcher_id, pitch_type,
        context.re24, base_state, outs, (balls, strikes), context.RE_MAX,
        context.pt_bucket_stats, context.league_avg,
        target_out_type=target_out_type,
    )

    return max(scores, key=scores.get)


def get_optimal_location(pitcher, batter, pitch_type: str, state_features: dict) -> str:
    """Convenience wrapper — builds context and scores in one call."""
    context = precompute_location_context(pitcher, batter)
    return get_optimal_location_from_context(context, pitch_type, state_features)
