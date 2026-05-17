"""
Programmatic wrapper around the running inference API server.
Start the server first (e.g. `docker compose up` or `uvicorn model_server.src.api:app`),
then call run_inference() or run this script directly.
"""

import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

BASE_URL = "http://localhost:3175"

DEFAULT_STATE_FEATURES: Dict[str, Any] = {
    "balls": 0,
    "strikes": 0,
    "outs_when_up": 0,
    "inning": 1,
    "inning_topbot": "Bot",
    "bat_score_diff": 0,
    "on_1b": 0,
    "on_2b": 0,
    "on_3b": 0,
}


def run_inference(
    pitcher: str,
    batter: str,
    year: int = 2025,
    state_features: Optional[Dict[str, Any]] = None,
    strategy: str = "argmax",
    max_pitches: int = 12,
    base_url: str = BASE_URL,
) -> Dict[str, Any]:
    payload = {
        "pitcher": pitcher,
        "batter": batter,
        "year": year,
        "state_features": {**DEFAULT_STATE_FEATURES, **(state_features or {})},
        "strategy": strategy,
        "max_pitches": max_pitches,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API error {e.code}: {body}") from e


def print_result(result: Dict[str, Any]) -> None:
    print(f"Outcome:     {result['outcome']}")
    print(f"Pitch count: {result['pitch_count']}")
    for step in result["sequence"]:
        loc = step.get("target_location", "?")
        print(
            f"  [{step['pitch_index']}] {step['pitch_type']:>4s} @ {loc:<10s}"
            f"  {step['transition_event']:>6s} / {step['out_type_event']:<6s}"
            f"  count: {step['balls_after']}-{step['strikes_after']}"
            + (f"  -> {step['outcome']}" if step["terminal"] else "")
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single plate-appearance simulation.")
    parser.add_argument("pitcher", help="Pitcher player ID")
    parser.add_argument("batter", help="Batter player ID")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--strategy", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--max-pitches", type=int, default=8)
    parser.add_argument("--url", default=BASE_URL, help="Base URL of the inference server")
    parser.add_argument("--state", default="{}", help="JSON string of state_features overrides")
    args = parser.parse_args()

    state_overrides = json.loads(args.state)
    result = run_inference(
        pitcher=args.pitcher,
        batter=args.batter,
        year=args.year,
        state_features=state_overrides,
        strategy=args.strategy,
        max_pitches=args.max_pitches,
        base_url=args.url,
    )
    print_result(result)
