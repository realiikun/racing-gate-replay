"""Read completed predictions and score against TXT annotations offline."""
import csv
import gzip
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from continuous import ACTIONS, DATA, FLIGHTS, HERE
from sensitivity import RADII, critical_radius, m


ORDERS = np.array([np.roll(base, shift) for base in (np.arange(4), np.arange(3, -1, -1)) for shift in range(4)])


def iou_matrix(truth, boxes):
    a = np.array([g["bbox"] for g in truth])[:, None, :]
    b = np.array(boxes)[None, :, :]
    intersection = np.maximum(0, np.minimum(a[..., 2:], b[..., 2:]) - np.maximum(a[..., :2], b[..., :2])).prod(axis=-1)
    area_a = np.maximum(0, a[..., 2:] - a[..., :2]).prod(axis=-1)
    area_b = np.maximum(0, b[..., 2:] - b[..., :2]).prod(axis=-1)
    return intersection / np.maximum(1e-9, area_a + area_b - intersection)


def matched_radii(truth, predictions):
    matched = 0
    radii = np.full((len(truth), 2), np.inf)
    if truth and predictions:
        overlap = iou_matrix(truth, [p[0] for p in predictions])
        bonus = min(overlap.shape) + 1
        rows, cols = linear_sum_assignment((overlap >= .3) * (bonus + overlap), maximize=True)
        for i, j in zip(rows, cols):
            if overlap[i,j] < .3:
                continue
            matched += 1
            p, valid = np.asarray(predictions[j][1]), np.asarray(predictions[j][2])
            thresholds = [critical_radius(p[o], valid[o], truth[i]) for o in ORDERS]
            radii[i] = [min(thresholds), thresholds[0]]
    return radii, matched


def assess(truth, predictions):
    radii, matched = matched_radii(truth, predictions)
    return (radii[:, :, None] <= RADII).sum(axis=0), matched


def score(flight, *, prediction_dir=HERE, runtime_file=None):
    folder = DATA / flight / ("label_" + flight)
    with gzip.open(prediction_dir / f"predictions_{flight}.jsonl.gz", "rt") as f:
        frames = [json.loads(line) for line in f]
    if runtime_file is None:
        runtime_file = prediction_dir / "continuous_runtime.json"
    assert len(frames) == json.loads(runtime_file.read_text())[flight]["frames"]
    available = {p.stem:p for p in folder.glob("*.txt")}
    totals = {action: np.zeros((2, len(RADII))) for action in ACTIONS}
    pair_time = dict.fromkeys(ACTIONS, 0.)
    unmatched_time = dict.fromkeys(ACTIONS, 0.)
    exposure = annotated_time = 0.
    n_intervals = 0
    for a,b in zip(frames, frames[1:]):
        dt = (b["timestamp_us"] - a["timestamp_us"]) / 1e6
        if dt > .025 or a["stem"] not in available or b["stem"] not in available:
            continue
        assert dt > 0
        n_intervals += 1
        annotated_time += dt
        truth = [g for g in m.parse_rows(available[a["stem"]]) if m.gt_diagonal(g) > 1e-6]
        exposure += dt * len(truth)
        for action in ACTIONS:
            good, pairs = assess(truth, a[action])
            totals[action] += dt * good
            pair_time[action] += dt * pairs
            unmatched_time[action] += dt * (len(a[action]) - pairs)
    rows = []
    for action, counts in totals.items():
        for idx, mode in enumerate(("cyclic", "fixed_slots")):
            for j, radius in enumerate(RADII):
                rows.append(dict(flight=flight, action=action, matching=mode, radius=radius,
                                 camera_frames=len(frames), annotated_intervals=n_intervals, annotated_seconds=annotated_time,
                                 gate_seconds=exposure, matched_gate_seconds=pair_time[action],
                                 unmatched_prediction_seconds=unmatched_time[action],
                                 bad_gate_seconds=exposure-counts[idx,j], bad_percent=100*(1-counts[idx,j]/exposure)))
    return rows


if __name__ == "__main__":
    results = [row for flight in FLIGHTS for row in score(flight)]
    with (HERE / "continuous_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    for r in results:
        if r["matching"] == "cyclic" and r["radius"] == .05:
            print(r["flight"], r["action"], round(r["bad_percent"], 3), flush=True)
