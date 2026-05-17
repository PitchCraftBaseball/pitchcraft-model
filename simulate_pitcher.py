"""
Simulate N plate appearances for a given pitcher against a random sample of
batters and game states, then report pitch usage split by batter handedness.

Requires the inference server to be running (docker compose up or equivalent).
Batter pool and handedness are sourced from the local historical_pitches.parquet
file — no AWS calls needed from this script.

Usage:
    python simulate_pitcher.py <pitcher_id> [--n-pa 200] [--year 2025] [--strategy argmax]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from run_inference import run_inference, DEFAULT_STATE_FEATURES

PARQUET_PATH = Path(__file__).parent / "data" / "historical_pitches.parquet"


def load_batter_pool() -> List[Tuple[str, str]]:
    """Return list of (batter_id, stand) from local parquet, filtered to 2025."""
    df = pd.read_parquet(PARQUET_PATH, columns=["batter", "stand", "game_year"])
    df = df[df["game_year"] == 2025]
    df["batter"] = df["batter"].astype(str)
    return list(df[["batter", "stand"]].drop_duplicates().itertuples(index=False, name=None))


def random_state() -> Dict[str, Any]:
    on_1b = random.randint(0, 1)
    on_2b = random.randint(0, 1)
    on_3b = random.randint(0, 1)
    return {
        "balls": 0,
        "strikes": 0,
        "outs_when_up": random.randint(0, 2),
        "inning": random.randint(1, 9),
        "inning_topbot": random.choice(["Top", "Bot"]),
        "bat_score_diff": random.randint(-5, 5),
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
    }


def simulate(
    pitcher: str,
    n_pa: int,
    year: int,
    strategy: str,
    base_url: str,
) -> None:
    batter_pool = load_batter_pool()
    if not batter_pool:
        sys.exit("No batters found in parquet file.")

    # pitch_counts[hand][pitch_type] = total times selected
    pitch_counts: Dict[str, Dict[str, int]] = {"L": defaultdict(int), "R": defaultdict(int)}
    outcome_counts: Dict[str, int] = defaultdict(int)
    errors = 0

    print(f"Simulating {n_pa} plate appearances for pitcher {pitcher} (strategy={strategy}, year={year})")
    print("-" * 60)

    for i in range(1, n_pa + 1):
        batter_id, stand = random.choice(batter_pool)
        state = random_state()

        try:
            result = run_inference(
                pitcher=pitcher,
                batter=batter_id,
                year=year,
                state_features=state,
                strategy=strategy,
                base_url=base_url,
            )
        except RuntimeError as exc:
            errors += 1
            print(f"  [PA {i}] ERROR: {exc}", file=sys.stderr)
            continue

        outcome_counts[result["outcome"]] += 1
        for step in result["sequence"]:
            pitch_counts[stand][step["pitch_type"]] += 1

        if i % 25 == 0:
            print(f"  Completed {i}/{n_pa} plate appearances ({errors} errors)")

    print()
    _print_summary(pitcher, n_pa, errors, pitch_counts, outcome_counts)


def _print_summary(
    pitcher: str,
    n_pa: int,
    errors: int,
    pitch_counts: Dict[str, Dict[str, int]],
    outcome_counts: Dict[str, int],
) -> None:
    completed = n_pa - errors
    print(f"{'=' * 60}")
    print(f"PITCHER: {pitcher}   PAs simulated: {completed}/{n_pa}")
    print(f"{'=' * 60}")

    print("\nOUTCOMES")
    print(f"  {'Outcome':<20} {'Count':>6}  {'%':>6}")
    print(f"  {'-' * 36}")
    for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / completed if completed else 0
        print(f"  {outcome:<20} {count:>6}  {pct:>5.1f}%")

    all_pitches = set(pitch_counts["L"]) | set(pitch_counts["R"])
    total_L = sum(pitch_counts["L"].values())
    total_R = sum(pitch_counts["R"].values())

    print("\nPITCH USAGE SPLITS")
    print(f"  {'Pitch':<8} {'vs L':>8} {'(%)':>7}  {'vs R':>8} {'(%)':>7}  {'Total':>8} {'(%)':>7}")
    print(f"  {'-' * 56}")

    grand_total = total_L + total_R
    for pitch in sorted(all_pitches):
        l_count = pitch_counts["L"].get(pitch, 0)
        r_count = pitch_counts["R"].get(pitch, 0)
        total = l_count + r_count
        l_pct = 100 * l_count / total_L if total_L else 0
        r_pct = 100 * r_count / total_R if total_R else 0
        t_pct = 100 * total / grand_total if grand_total else 0
        print(f"  {pitch:<8} {l_count:>8} {l_pct:>6.1f}%  {r_count:>8} {r_pct:>6.1f}%  {total:>8} {t_pct:>6.1f}%")

    print(f"  {'TOTAL':<8} {total_L:>8}          {total_R:>8}          {grand_total:>8}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate plate appearances for a pitcher.")
    parser.add_argument("pitcher", help="Pitcher player ID")
    parser.add_argument("--n-pa", type=int, default=100, help="Number of plate appearances to simulate")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--strategy", choices=["argmax", "sample", "optimal_out"], default="argmax")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the inference server")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    simulate(
        pitcher=args.pitcher,
        n_pa=args.n_pa,
        year=args.year,
        strategy=args.strategy,
        base_url=args.url,
    )
