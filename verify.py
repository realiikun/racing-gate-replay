"""Check coverage, per-flight aggregation, denominator bounds and call timing."""
import collections
import csv
import gzip
import json
from bisect import bisect_left
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
ACTIONS = ("wait", "aligned", "legacy")


def table(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    anchor = table("anchor_outages_ms.csv")
    grouped = collections.defaultdict(list)
    for row in anchor:
        key = row["flight"], row["tid"], row["anchor"], row["radius"]
        grouped[key].append(row)
    assert len(grouped) == 268 * 4
    for rows in grouped.values():
        assert [int(r["duration_ms"]) for r in rows] == [0, 25, 50, 75]
        for action in ACTIONS:
            values = [float(r[action]) for r in rows]
            assert 250.00001 >= values[0] >= values[1] >= values[2] >= values[3] >= 0
    summaries = table("anchor_summary_ms.csv")
    rank = collections.Counter()
    for row in summaries:
        order = "<".join(sorted(ACTIONS, key=lambda a: float(row[a])))
        rank[(row["flight"], order)] += 1
    print("Anchor ordering counts", dict(rank))
    for row in summaries:
        if row["radius"] == "0.05" and row["flight"] == "flight_equal":
            print("Anchor", row["duration_ms"], *(round(float(row[a]), 3) for a in ACTIONS))
    runtime = json.loads((ROOT / "continuous_runtime.json").read_text())
    continuous = table("continuous_scores.csv")
    for flight in {r["flight"] for r in continuous}:
        calls = table(f"calls_{flight}.csv")
        with gzip.open(ROOT / f"predictions_{flight}.jsonl.gz", "rt") as src:
            frames = [json.loads(line) for line in src]
        timestamps = [r["timestamp_us"] for r in frames]
        assert all(a < b for a, b in zip(timestamps, timestamps[1:]))
        assert len(timestamps) == runtime[flight]["frames"]
        assert len(calls) == runtime[flight]["calls"]
        assert len({r["capture_us"] for r in calls}) == len(calls)
        assert all((not r["return_us"]) or int(r["return_us"]) - int(r["capture_us"]) >= 46686 for r in calls)
        for call in calls:
            if call["return_us"]:
                i = bisect_left(timestamps, int(call["capture_us"]) + 46686)
                assert timestamps[i] == int(call["return_us"])
        first_return = min(int(c["return_us"]) for c in calls if c["return_us"])
        assert all(not r[a] for r in frames if r["timestamp_us"] < first_return for a in ACTIONS)
        assert len({r["gate_seconds"] for r in continuous if r["flight"] == flight}) == 1
        for r in (r for r in continuous if r["flight"] == flight):
            assert -1e-8 <= float(r["bad_gate_seconds"]) <= float(r["gate_seconds"]) + 1e-8
            assert float(r["unmatched_prediction_seconds"]) >= 0
    for matching in ("cyclic", "fixed_slots"):
        for radius in ("0.025", "0.05", "0.075", "0.1"):
            group = [r for r in continuous if r["matching"] == matching and r["radius"] == radius]
            assert len(group) == 9
            print("Continuous", matching, radius, *(round(mean(float(r["bad_percent"]) for r in group if r["action"]==a), 3) for a in ACTIONS))


if __name__ == "__main__":
    main()
