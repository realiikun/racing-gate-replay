"""Rescore fixed, paired development anchors with the original LK implementation."""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE / "inputs"
sys.path.insert(0, str(HERE))
import replay_alignment as original

m = original.m
RADII = (0.025, 0.05, 0.075, 0.10)
DURATIONS = (0, 25, 50, 75)
ACTIONS = ("wait", "aligned", "legacy")
PAIRS = np.tril_indices(4, -1)


def critical_radius(points, valid, reference):
    """Minimum normalized radius at which a suitably separated pair hits."""
    truth = reference["corners"]
    i, j = PAIRS
    distances = np.linalg.norm(truth[i] - truth[j], axis=1)
    visible = reference["vis"][i] & reference["vis"][j]
    scale = distances[visible].max(initial=0)
    if scale <= 1e-6:
        return None
    usable = visible & (distances >= 0.25 * scale) & valid[i] & valid[j]
    errors = np.linalg.norm(points - truth, axis=1) / scale
    return np.maximum(errors[i], errors[j])[usable].min(initial=np.inf)


def duration_outages(times, critical, anchor):
    times = np.asarray(times)
    lengths = np.maximum(0, np.minimum(times[1:], anchor + m.HORIZON) - times[:-1])
    output = {}
    for radius in RADII:
        bad = np.asarray(critical[:-1]) > radius
        edges = np.flatnonzero(np.diff(np.r_[False, bad, False].astype(int)))
        runs = np.array([lengths[a:b].sum() for a, b in edges.reshape(-1, 2)])
        for duration in DURATIONS:
            output[radius, duration] = runs[runs >= duration / 1000 - 1e-9].sum()
    return output


def replay(rec, det, cache, times, anchor):
    fi, tid, start = anchor["fi"], anchor["tid"], anchor["anchor_t"]
    last = int(np.argmin(abs(times - (start - anchor["age"]))))
    assert abs(times[last] - start + anchor["age"]) < 1e-5
    prior = original.detection(rec, det, last, tid)
    initial = original.transport(rec, cache, last, fi, *prior[:2])
    fresh = original.detection(rec, det, fi, tid)[:2]
    ret = times.searchsorted(start + m.D_SERVICE)
    aligned = original.transport(rec, cache, fi, ret, *fresh)
    # Intercept the existing branch's offline score only. Propagation, timing,
    # visibility censoring and update insertion are kept in the frozen code.
    previous = m.judge_state
    output = {}
    try:
        for name, replacement in (("wait", None), ("aligned", aligned), ("legacy", fresh)):
            critical = []
            def score(p, v, gt):
                threshold = critical_radius(p, v, gt)
                if threshold is None:
                    return "non_opportunity", 0
                critical.append(threshold)
                return ("good" if threshold <= 0.05 else "bad"), 1
            m.judge_state = score
            kwargs = {} if replacement is None else dict(update_fi=ret, update_points=replacement[0], update_valid=replacement[1])
            ts, _, _ = m.run_branch(rec, cache, fi, *initial, tid, start, **kwargs)
            output[name] = duration_outages(ts, critical, start)
    finally:
        m.judge_state = previous
    return output


def main():
    old = PREVIOUS
    target = json.loads((old / "alignment_clean_development.json").read_text())["rows"]
    checkpoints = json.loads((original.DATA / "anchors_checkpoint.json").read_text(encoding="utf-8"))["rows"]
    rows, worst = [], 0
    for name in m.DEV:
        rec = m.Record(name)
        rec.build_tracks()
        cache = m.FrameCache(rec)
        det = m.YoloSource(name)
        times = np.array([frame["t"] for frame in rec.frames])
        lookup = {(r["tid"], r["anchor_index"]): r for r in checkpoints[name]}
        for expected in (a for a in target if a["rec"] == name):
            key = expected["tid"], expected["anchor_index"]
            result = replay(rec, det, cache, times, lookup[key])
            for action, field in zip(ACTIONS, ("wait_s", "aligned_47ms_s", "legacy_47ms_s")):
                worst = max(worst, abs(result[action][.05, 0] - expected[field]))
                if worst > 1e-7:
                    raise ValueError((name, key, action, worst))
            for radius in RADII:
                for dur in DURATIONS:
                    rows.append(dict(flight=name, tid=key[0], anchor=key[1], radius=radius, duration_ms=dur,
                                     **{action: 1000 * result[action][radius, dur] for action in ACTIONS}))
        print(name, "completed", flush=True)
    assert len(rows) == 268 * len(RADII) * len(DURATIONS)
    def write(name, records):
        with (HERE / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    write("anchor_outages_ms.csv", rows)
    summaries = []
    for radius in RADII:
        for duration in DURATIONS:
            flight_means = []
            for name in m.DEV:
                sample = [r for r in rows if (r["radius"], r["duration_ms"], r["flight"]) == (radius, duration, name)]
                means = {a: np.mean([r[a] for r in sample]) for a in ACTIONS}
                flight_means.append(means)
                summaries.append(dict(flight=name, radius=radius, duration_ms=duration, **means))
            summaries.append(dict(flight="flight_equal", radius=radius, duration_ms=duration,
                                  **{a: np.mean([r[a] for r in flight_means]) for a in ACTIONS}))
    write("anchor_summary_ms.csv", summaries)
    print("maximum reproduction error (seconds)", worst, flush=True)


if __name__ == "__main__":
    main()
