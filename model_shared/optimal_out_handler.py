import pandas as pd
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from model_shared.feature_engineering.feature_calculator import *
from model_shared.feature_engineering.pitch_constants import BASE_LABELS

_BASE_STATE_DECODE = {v: k for k, v in BASE_LABELS.items()}
from sklearn.preprocessing import MinMaxScaler


@dataclass
class OptimalOutContext:
    """Per-process state needed to score optimal-out for any (pitcher, batter, state)
    triple. Build once via build_optimal_out_context(); pass to
    get_optimal_out_from_context() per request.
    """
    pitcher_scaled: pd.DataFrame
    batter_scaled: pd.DataFrame
    re24: dict
    RE24_MAX: float


def sort_statcast(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["game_type"] == 'R']
    df = df[df["pitch_type"] != 'UN']
    return df.sort_values(["game_date", "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"], ascending=[True, True, True, False, True, True]).reset_index(drop=True)

def out_type_refined(row):
    if row['events'] in ['strikeout', 'strikeout_double_play']:
        return 'strikeout'
    elif row['bb_type'] == 'ground_ball':
        return 'groundout'
    elif row['bb_type'] in ['fly_ball', 'popup']:
        return 'flyout'
    elif row['bb_type'] == 'line_drive':
        return 'lineout'
    else:
        return None

def _latest_year_per(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """For each player in id_col, keep only their most recent game_year.
    A player active in 2025 uses 2025; one whose last appearance was 2024
    uses 2024; etc. Two slices are needed downstream because the same row's
    "latest year" differs depending on whether you key by batter or pitcher.
    """
    max_year = frame.groupby(id_col)['game_year'].transform('max')
    return frame[frame['game_year'] == max_year]


def full_dataframe(pitch_df: pd.DataFrame, woba_pivot: pd.DataFrame ) -> pd.DataFrame:

    pitch_df['whiff'] = pitch_df['description'].isin([
        'swinging_strike', 'swinging_strike_blocked', 'foul_tip'
    ]).astype(int)

    pitch_df['swing'] = pitch_df['description'].isin([
        'foul_bunt', 'foul', 'hit_into_play', 'swinging_strike', 'foul_tip',
        'swinging_strike_blocked', 'missed_bunt', 'bunt_foul_tip'
    ]).astype(int)

    pitch_df['gb']           = (pitch_df['bb_type'] == 'ground_ball').astype(int)
    pitch_df['fly']          = ((pitch_df['bb_type'] == 'fly_ball') | (pitch_df['bb_type'] == 'popup')).astype(int)
    pitch_df['bip']          = pitch_df['bb_type'].notna().astype(int)
    pitch_df['popup']        = (pitch_df['bb_type'] == 'popup').astype(int)
    pitch_df['fly_ball_only'] = (pitch_df['bb_type'] == 'fly_ball').astype(int)
    pitch_df['hard_hit_fly'] = ((pitch_df['launch_speed'] >= 95) & (pitch_df['bb_type'] == 'fly_ball')).astype(int)
    pitch_df['barrel']       = (pitch_df['launch_speed_angle'] == 6).astype(int)

    batter_pitch = pitch_df.groupby(['batter', 'pitch_group']).agg(
        swings        = ('swing', 'sum'),
        whiffs        = ('whiff', 'sum'),
        bip           = ('bip', 'sum'),
        gb            = ('gb', 'sum'),
        fly           = ('fly', 'sum'),
        popup         = ('popup', 'sum'),
        fly_ball_only = ('fly_ball_only', 'sum'),
        hard_hit_fly  = ('hard_hit_fly', 'sum'),
        barrels       = ('barrel', 'sum')
    ).reset_index()

    batter_pitch['whiff_rate']        = batter_pitch['whiffs'] / batter_pitch['swings']
    batter_pitch['gb_rate']           = batter_pitch['gb'] / batter_pitch['bip']
    batter_pitch['fly_rate']          = batter_pitch['fly'] / batter_pitch['bip']
    batter_pitch['popup_rate']        = batter_pitch['popup'] / batter_pitch['bip']
    batter_pitch['hard_hit_fly_rate'] = batter_pitch['hard_hit_fly'] / batter_pitch['fly_ball_only']
    batter_pitch['barrel_rate']       = batter_pitch['barrels'] / batter_pitch['bip']

    # batter_pitch: batter	pitch_group 	swings	 whiffs 	bip	gb	 fly	whiff_rate 	 gb_rate 	fly_rate

    whiff_pivot = batter_pitch.pivot(index='batter', columns='pitch_group', values='whiff_rate').reset_index()
    whiff_pivot.columns = ['batter', 'Breaking_whiff', 'Fastball_whiff', 'Offspeed_whiff']

    gb_pivot = batter_pitch.pivot(index='batter', columns='pitch_group', values='gb_rate').reset_index()
    gb_pivot.columns = ['batter', 'Breaking_gb', 'Fastball_gb', 'Offspeed_gb']

    fly_pivot = batter_pitch.pivot(index='batter', columns='pitch_group', values='fly_rate').reset_index()
    fly_pivot.columns = ['batter', 'Breaking_fly', 'Fastball_fly', 'Offspeed_fly']

    full_df = woba_pivot.merge(whiff_pivot, on='batter', how='left')
    full_df = full_df.merge(gb_pivot, on='batter', how='left')
    full_df = full_df.merge(fly_pivot, on='batter', how='left')

    return full_df

def calculate_scores(p, b, re24, base_state, outs, RE24_MAX):
    scores = {}

    woba_vals = {
        'woba_fb': 1 - b['Fastball_wOBA'],
        'woba_br': 1 - b['Breaking_wOBA'],
        'woba_os': 1 - b['Offspeed_wOBA'],
        'woba_avg': 1 - b['wOBA'],
    }

    # RE24 lookup, lower run expectancy is better for the pitcher
    re_state = re24[base_state][str(outs)]
    re_raw = {
        'strikeout': re_state['strikeout']['run_expectancy'],
        'groundout': re_state['groundball']['run_expectancy'],
        'flyout':    re_state['flyball']['run_expectancy'],
    }
    # high re_score = low RE = good for pitcher
    re_score = {ot: 1 - re_raw[ot] / RE24_MAX for ot in re_raw}

    empty_base_bonus = 0.1 if base_state == 'XXX' else 0.0
    re_weight = 0.10 if base_state == 'XXX' else 0.34
    scores['strikeout'] = (1 - re_weight) * (
        0.35 * (p['whiff_rate'] + b['whiff_rate']) / 2 +
        0.35 * (p['chase_rate'] + b['chase_rate']) / 2 +
        0.15 * woba_vals['woba_br'] +
        0.15 * woba_vals['woba_os'] -
        0.08 * b['contact_rate']
    ) + re_weight * re_score['strikeout']

    scores['groundout'] = (1 - re_weight) * (
        0.35 * p['gb_rate'] +
        0.35 * b['gb_rate'] +
        0.05 * p['contact_rate'] +
        0.05 * b['contact_rate'] +
        0.10 * woba_vals['woba_fb'] +
        0.10 * woba_vals['woba_avg']
    ) + re_weight * re_score['groundout']

    b_fly_specific = (
        0.4 * b.get('Fastball_fly', b['fly_rate']) +
        0.35 * b.get('Breaking_fly', b['fly_rate']) +
        0.25 * b.get('Offspeed_fly', b['fly_rate'])
    )

    scores['flyout'] = (1 - re_weight) * (
        0.22 * p['fly_rate'] +
        0.22 * b_fly_specific +
        0.22 * b['popup_rate'] +
        0.12 * (1 - b['barrel_rate']) +
        0.12 * (1 - b['hard_hit_fly_rate']) +
        0.10 * woba_vals['woba_avg'] 
    ) + re_weight * re_score['flyout'] + empty_base_bonus
    return scores

def calculate_optimal_out(pitcher, batter, outs, base_state, pitcher_scaled, batter_scaled, re24, RE24_MAX):
    """
    pitcher_name : str  e.g. 'Zack Wheeler'
    batter_name  : str  e.g. 'Aaron Judge'
    outs         : int  0, 1, or 2
    base_state   : str  8-state code where O=runner, X=empty, positions are 1B/2B/3B
                        'XXX' bases empty, 'OXX' runner on 1st, 'OOO' bases loaded, etc.
    """
    if base_state not in re24:
        valid = list(re24.keys())
        raise ValueError(f"Invalid base_state '{base_state}'. Valid values: {valid}")
    if outs not in (0, 1, 2):
        raise ValueError(f"Invalid outs: {outs}. Must be 0, 1, or 2.")

    pitcher_id = int(pitcher)
    batter_id  = int(batter)
    pitcher_row = pitcher_scaled[pitcher_scaled['pitcher'] == pitcher_id]
    batter_row  = batter_scaled[batter_scaled['batter'] == batter_id]

    if pitcher_row.empty:
        raise ValueError(f"No data for pitcher: {pitcher_id}")
    if batter_row.empty:
        raise ValueError(f"No data for batter: {batter_id}")

    pitcher = pitcher_row
    batter  = batter_row

    p = pitcher.iloc[0]
    b = batter.iloc[0]
    
    scores = calculate_scores(p, b, re24, base_state, outs, RE24_MAX)

    best = max(scores, key=scores.get)

    return scores


def build_optimal_out_context() -> OptimalOutContext:
    """Build the per-process context used by get_optimal_out_from_context.

    Reads the historical pitches parquet once and runs the league-wide
    aggregations that used to happen on every per-PA call. Only the small
    scaled per-player frames (and re24 tables) are retained — the raw
    parquet is dropped when this function returns.
    """
    file_path = Path(__file__).parent.parent / "rnn_support_models" / "run_expectancy" / "2025-2025_re24.json"
    with open(file_path) as f:
        re24 = json.load(f)

    RE24_MAX = max(
        re24[bs][str(o)][ot]['run_expectancy']
        for bs in re24
        for o in range(3)
        for ot in ['strikeout', 'groundball', 'flyball']
    )


    # Gather Data - is this right
    df = pd.read_parquet(
        Path(__file__).parent.parent / "data" / "historical_pitches.parquet")
    df = df[df["game_year"].isin([2023, 2024, 2025])]

    data = sort_statcast(df)

    pa_df = data[data["events"].notna() & (data["events"] != "truncated_pa")].copy()
    pa_df['pitch_group'] = pa_df['pitch_type'].apply(pitch_to_family)
    pa_df['out_type']    = pa_df.apply(out_type_refined, axis=1)

    # calculate_woba mutates its argument in place, adding the per-row wOBA
    # component columns (uBB, HBP, 1B..HR, AB, IBB, SF). Run it on the parent
    # before the split so BOTH the batter and pitcher slices carry those
    # columns (the pitcher walk-rate aggregation below reads 'uBB' too). The
    # return value here is discarded; the latest-year batter wOBA is computed
    # from pa_df_b further down.
    calculate_woba(pa_df)

    # Per-player latest-year slices. Done before any per-role aggregation so
    # that batter-side groupbys see only each batter's most recent season and
    # pitcher-side groupbys see only each pitcher's most recent season.
    pa_df_b = _latest_year_per(pa_df, 'batter')
    pa_df_p = _latest_year_per(pa_df, 'pitcher')

    # Do Feature Engineering — overall batter wOBA from their latest year only.
    woba_df = calculate_woba(pa_df_b)
    woba = woba_df[['wOBA']].reset_index()

    # Drop rows where pitch type doesn't fit a group
    pa_df_grouped = pa_df_b[pa_df_b['pitch_group'].notna()]

    # Aggregate by batter AND pitch group
    grouped = pa_df_grouped.groupby(['batter', 'pitch_group'])[
        ['uBB','HBP','1B','2B','3B','HR','AB','IBB','SF']
    ].sum()

    # Apply weights
    grouped['wOBA'] = (
        0.691 * grouped['uBB'] +
        0.722 * grouped['HBP'] +
        0.882 * grouped['1B'] +
        1.252 * grouped['2B'] +
        1.584 * grouped['3B'] +
        2.037 * grouped['HR']
    ) / (grouped['AB'] + grouped['uBB'] + grouped['SF'] + grouped['HBP'])

    woba_by_pitch = grouped[['wOBA']].reset_index()

    woba_pivot = woba_by_pitch.pivot(index='batter', columns='pitch_group', values='wOBA').reset_index()
    woba_pivot.columns = ['batter', 'Breaking_wOBA', 'Fastball_wOBA', 'Offspeed_wOBA']
    woba_pivot = woba_pivot[['batter', 'Breaking_wOBA', 'Fastball_wOBA', 'Offspeed_wOBA']]

    # For each batter, get how many plater appearances they have 
    pa_counts = grouped.groupby('batter').apply(
        lambda x: (x['AB'] + x['uBB'] + x['SF'] + x['HBP']).sum()
    ).reset_index(name='PA')

    # Filter to qualified batters
    min_pa = 50
    # A list of batter ids with only qualified batters
    qualified = pa_counts[pa_counts['PA'] >= min_pa]['batter']

    data['pitch_group'] = data['pitch_type'].apply(pitch_to_family)
    pitch_df = data[data['pitch_group'].notna()].copy()

    # Lift per-row derivations above the role split so both batter and
    # pitcher slices carry the columns the aggregations need. full_dataframe
    # also writes these columns; on pitch_df_b that becomes an idempotent
    # overwrite (its return value is unused downstream).
    pitch_df['whiff'] = pitch_df['description'].isin([
        'swinging_strike', 'swinging_strike_blocked', 'foul_tip'
    ]).astype(int)
    pitch_df['swing'] = pitch_df['description'].isin([
        'foul_bunt', 'foul', 'hit_into_play', 'swinging_strike', 'foul_tip',
        'swinging_strike_blocked', 'missed_bunt', 'bunt_foul_tip'
    ]).astype(int)
    pitch_df['gb']            = (pitch_df['bb_type'] == 'ground_ball').astype(int)
    pitch_df['fly']           = ((pitch_df['bb_type'] == 'fly_ball') | (pitch_df['bb_type'] == 'popup')).astype(int)
    pitch_df['bip']           = pitch_df['bb_type'].notna().astype(int)
    pitch_df['popup']         = (pitch_df['bb_type'] == 'popup').astype(int)
    pitch_df['fly_ball_only'] = (pitch_df['bb_type'] == 'fly_ball').astype(int)
    pitch_df['hard_hit_fly']  = ((pitch_df['launch_speed'] >= 95) & (pitch_df['bb_type'] == 'fly_ball')).astype(int)
    pitch_df['barrel']        = (pitch_df['launch_speed_angle'] == 6).astype(int)
    pitch_df['chase']         = ((pitch_df['zone'] > 9) & (pitch_df['swing'] == 1)).astype(int)
    pitch_df['out_of_zone']   = (pitch_df['zone'] > 9).astype(int)

    pitch_df_b = _latest_year_per(pitch_df, 'batter')
    pitch_df_p = _latest_year_per(pitch_df, 'pitcher')

    full_df = full_dataframe(pitch_df_b, woba_pivot)

    outs_df_b = pa_df_b[(pa_df_b['events'] == 'field_out') | (pa_df_b['events'] == 'strikeout')].copy()
    outs_df_p = pa_df_p[(pa_df_p['events'] == 'field_out') | (pa_df_p['events'] == 'strikeout')].copy()

    # Total PAs per batter
    pa_totals = pa_df_b.groupby('batter').size().reset_index(name='total_pa')

    # Out type counts per batter
    ### MIGHT NEED SOME VALIDATION ###
    out_counts = outs_df_b.groupby(['batter', 'out_type']).size().reset_index(name='count')

    # Merge and calculate rates
    # Out of all the ABs, what percentage does each out type occur?
    out_rates = out_counts.merge(pa_totals, on='batter', how='left')
    out_rates['rate'] = out_rates['count'] / out_rates['total_pa']

    # Pivot to wide
    batter_out_profile = out_rates.pivot_table(
        index='batter', columns='out_type', values='rate', fill_value=0
    ).reset_index()
    batter_out_profile.columns.name = None

    # # --- Pitcher out type tendencies ---
    # # Same logic but for pitchers
    pitcher_pa_totals = pa_df_p.groupby('pitcher').size().reset_index(name='total_pa')

    pitcher_out_counts = outs_df_p.groupby(['pitcher', 'out_type']).size().reset_index(name='count')

    pitcher_out_rates = pitcher_out_counts.merge(pitcher_pa_totals, on='pitcher', how='left')
    pitcher_out_rates['rate'] = pitcher_out_rates['count'] / pitcher_out_rates['total_pa']

    pitcher_out_profile = pitcher_out_rates.pivot_table(
        index='pitcher', columns='out_type', values='rate', fill_value=0
    ).reset_index()
    pitcher_out_profile.columns.name = None

    # --- Batter features ---
    batter_features = pitch_df_b.groupby(['batter']).agg(
        swings        = ('swing', 'sum'),
        whiffs        = ('whiff', 'sum'),
        bip           = ('bip', 'sum'),
        gb            = ('gb', 'sum'),
        fly           = ('fly', 'sum'),
        popup         = ('popup', 'sum'),
        fly_ball_only = ('fly_ball_only', 'sum'),
        hard_hit_fly  = ('hard_hit_fly', 'sum'),
        barrels       = ('barrel', 'sum')
    ).reset_index()

    # overall features, as opposed to by pitch group
    batter_features['whiff_rate']        = batter_features['whiffs'] / batter_features['swings']
    batter_features['gb_rate']           = batter_features['gb'] / batter_features['bip']
    batter_features['fly_rate']          = batter_features['fly'] / batter_features['bip']
    batter_features['popup_rate']        = batter_features['popup'] / batter_features['bip']
    batter_features['hard_hit_fly_rate'] = batter_features['hard_hit_fly'] / batter_features['fly_ball_only']
    batter_features['barrel_rate']       = batter_features['barrels'] / batter_features['bip']

    # Contact rate
    batter_features['contact_rate'] = 1 - batter_features['whiff_rate']  # already have whiffs/swings

    # Walk rate — needs PA-level data
    walk_rates = pa_df_b.groupby('batter').agg(
        walks = ('uBB', 'sum'),
        total_pa = ('events', 'count')
    ).reset_index()
    walk_rates['walk_rate'] = walk_rates['walks'] / walk_rates['total_pa']

    batter_features = batter_features.merge(walk_rates[['batter', 'walk_rate']], on='batter', how='left')

    # Add overall wOBA
    overall_woba = woba[['batter', 'wOBA']]
    batter_features = batter_features.merge(overall_woba, on='batter', how='left')

    # Add out type rates
    batter_features = batter_features.merge(batter_out_profile, on='batter', how='left')

    # Add chase rate (swings on pitches outside zone)
    chase_rates = pitch_df_b.groupby('batter').agg(
        chases       = ('chase', 'sum'),
        out_of_zone  = ('out_of_zone', 'sum')
    ).reset_index()
    chase_rates['chase_rate'] = chase_rates['chases'] / chase_rates['out_of_zone']

    batter_features = batter_features.merge(chase_rates[['batter', 'chase_rate']], on='batter', how='left')

    # --- Pitcher features ---
    pitcher_features = pitcher_out_profile.copy()

    # Add pitcher whiff, gb, chase rates
    pitcher_pitch = pitch_df_p.groupby('pitcher').agg(
        swings      = ('swing', 'sum'),
        whiffs      = ('whiff', 'sum'),
        bip         = ('bip', 'sum'),
        gb          = ('gb', 'sum'),
        fly         = ('fly', 'sum'),
        chases      = ('chase', 'sum'),
        out_of_zone = ('out_of_zone', 'sum')
    ).reset_index()

    pitcher_pitch['whiff_rate'] = pitcher_pitch['whiffs'] / pitcher_pitch['swings']
    pitcher_pitch['gb_rate']    = pitcher_pitch['gb'] / pitcher_pitch['bip']
    pitcher_pitch['fly_rate']    = pitcher_pitch['fly'] / pitcher_pitch['bip']
    pitcher_pitch['chase_rate'] = pitcher_pitch['chases'] / pitcher_pitch['out_of_zone']

    # Same for pitchers, after pitcher_pitch is built
    pitcher_features = pitcher_features.merge(
               pitcher_pitch[['pitcher', 'whiff_rate']].rename(
                   columns={'whiff_rate': 'raw_whiff_rate'}
               ),
               on='pitcher', how='left'
           )
    pitcher_features['contact_rate'] = 1 - pitcher_features['raw_whiff_rate']
    pitcher_features = pitcher_features.drop(columns=['raw_whiff_rate'])

    pitcher_walk_rates = pa_df_p.groupby('pitcher').agg(
        walks = ('uBB', 'sum'),
        total_pa = ('events', 'count')
    ).reset_index()
    pitcher_walk_rates['walk_rate'] = pitcher_walk_rates['walks'] / pitcher_walk_rates['total_pa']

    pitcher_features = pitcher_features.merge(
        pitcher_walk_rates[['pitcher', 'walk_rate']], on='pitcher', how='left'
    )

    pitcher_features = pitcher_features.merge(
        pitcher_pitch[['pitcher', 'whiff_rate', 'gb_rate', 'fly_rate', 'chase_rate']],
        on='pitcher', how='left'
    )

    # Do the woba_pivot merge first
    batter_features = batter_features.merge(woba_pivot, on='batter', how='left')

    # Define all columns upfront (including the pitch-type wOBAs)
    batter_scale_cols = [
        'wOBA', 'whiff_rate', 'gb_rate', 'fly_rate', 'chase_rate',
        'contact_rate', 'walk_rate',
        'popup_rate', 'hard_hit_fly_rate', 'barrel_rate',
        'Breaking_wOBA', 'Fastball_wOBA', 'Offspeed_wOBA'
    ]
    pitcher_scale_cols = [
        'whiff_rate', 'gb_rate', 'fly_rate', 'chase_rate',
        'contact_rate', 'walk_rate',
    ]

    batter_scaler  = MinMaxScaler()
    pitcher_scaler = MinMaxScaler()

    batter_scaled  = batter_features.copy().reset_index(drop=True)
    pitcher_scaled = pitcher_features.copy().reset_index(drop=True)

    batter_scaled[batter_scale_cols]   = batter_scaler.fit_transform(batter_features[batter_scale_cols].fillna(0))
    pitcher_scaled[pitcher_scale_cols] = pitcher_scaler.fit_transform(pitcher_features[pitcher_scale_cols].fillna(0))

    # _FALLBACK = {"strikeout": 1/3, "groundout": 1/3, "flyout": 1/3}

    return OptimalOutContext(
        pitcher_scaled=pitcher_scaled,
        batter_scaled=batter_scaled,
        re24=re24,
        RE24_MAX=RE24_MAX,
    )


def get_optimal_out_from_context(
    ctx: OptimalOutContext, pitcher, batter, state_features
):
    """Per-request optimal-out lookup. Cheap: derives (outs, base_state) and
    runs calculate_optimal_out against the prebuilt scaled frames in ctx.
    """
    outs = state_features['outs_when_up']
    base_state_int = (
        int(bool(state_features.get("on_1b", 0))) * 1
        + int(bool(state_features.get("on_2b", 0))) * 2
        + int(bool(state_features.get("on_3b", 0))) * 4
    )
    base_state = _BASE_STATE_DECODE[base_state_int]

    return calculate_optimal_out(
        pitcher, batter, outs, base_state,
        ctx.pitcher_scaled, ctx.batter_scaled, ctx.re24, ctx.RE24_MAX,
    )


_default_ctx: Optional[OptimalOutContext] = None


def get_optimal_out(pitcher, batter, state_features):
    """Backwards-compatible single-call API. Builds the heavy context on
    first use and caches it for subsequent calls in this process. Server
    code should call build_optimal_out_context() at startup and use
    get_optimal_out_from_context() instead to avoid the lazy-build cost
    landing on the first request.
    """
    global _default_ctx
    if _default_ctx is None:
        _default_ctx = build_optimal_out_context()
    return get_optimal_out_from_context(_default_ctx, pitcher, batter, state_features)


if __name__ == "__main__":
    pitcher = 808967
    batter = 664761
    state_features = {
        "outs_when_up": 0,
        "on_1b": 0,
        "on_2b": 0,
        "on_3b": 0,
    }
    optimal_out = get_optimal_out(pitcher, batter, state_features)
    print(optimal_out)
    optimal_out_type = max(optimal_out, key=optimal_out.get)  # "strikeout", "groundout", or "flyout"
    print(optimal_out_type)
