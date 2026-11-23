"""Exploratory bridge: score the frozen continuous predictions on the original windows."""
import csv
import gzip
import json

import numpy as np

from continuous import ACTIONS, FLIGHTS, HERE, PREVIOUS
from score_continuous import matched_radii
from sensitivity import duration_outages, m, original


def score(flight, anchors):
    rec = m.Record(flight)
    rec.build_tracks()
    with gzip.open(HERE / f"predictions_{flight}.jsonl.gz", "rt") as f:
        predictions = {r["stem"]:r for r in map(json.loads, f)}
    cache = {}
    results = []
    for anchor in anchors:
        tid, start = anchor["tid"], anchor["anchor_t"]
        times = []
        thresholds = {a: [] for a in ACTIONS}
        for fi in range(anchor["fi"] + 1, len(rec.frames)):
            frame = rec.frames[fi]
            if frame["t"] > start + .30 + 1e-6:
                break
            if fi not in cache:
                scoreable = [(i, g) for i, g in rec.track_rows[fi] if m.gt_diagonal(g) > 1e-6]
                positions = {i: j for j, (i, _) in enumerate(scoreable)}
                truth = [g for _, g in scoreable]
                pred = predictions[frame["stem"]]
                cache[fi] = positions, {a: matched_radii(truth, pred[a])[0][:, 0] for a in ACTIONS}
            positions, scores = cache[fi]
            if tid not in positions:
                break
            times.append(frame["t"])
            for a in ACTIONS:
                thresholds[a].append(scores[a][positions[tid]])
        assert times[-1] >= start + .25 - 1e-6
        outcomes = {a: duration_outages(times, thresholds[a], start) for a in ACTIONS}
        results.append(dict(flight=flight, tid=tid, anchor=anchor["anchor_index"],
                            **{a:1000*outcomes[a][.05, 0] for a in ACTIONS}))
    return results


if __name__ == "__main__":
    original_rows = json.loads((PREVIOUS / "alignment_clean_development.json").read_text())["rows"]
    lookup = json.loads((original.DATA / "anchors_checkpoint.json").read_text(encoding="utf-8"))["rows"]
    rows = []
    for flight in FLIGHTS:
        keys = {(r["tid"], r["anchor_index"]) for r in original_rows if r["rec"] == flight}
        anchors = [r for r in lookup[flight] if (r["tid"], r["anchor_index"]) in keys]
        assert len(anchors) == len(keys)
        scored = score(flight, anchors)
        print(flight, len(scored), [round(np.mean([r[a] for r in scored]), 3) for a in ACTIONS], flush=True)
        rows.extend(scored)
    assert len(rows) == 268
    with (HERE / "continuous_same_windows_ms.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

