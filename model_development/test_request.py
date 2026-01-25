import json
import urllib.request


def main() -> None:
    url = "http://localhost:8000/predict"
    payload = {
        "sequence": [
            {
                "pitcher": "12345",
                "batter": "98765",
                "stand": "L",
                "p_throws": "R",
                "inning_topbot": "Top",
                "count_state": "1-1",
                "prev_pitch_type": "FF",
                "balls": 1,
                "strikes": 1,
                "outs_when_up": 1,
                "inning": 3,
                "score_diff_bat": 0,
                "on_1b": 0,
                "on_2b": 1,
                "on_3b": 0,
            }
        ]
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
    print(body)


if __name__ == "__main__":
    main()
