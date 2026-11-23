"""Image-only replay. This executable never opens a label file or a flight-state CSV."""
import collections
import csv
import gzip
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
from config import DATA
PREVIOUS = HERE / "inputs"
from causal_tracker import associate
from lk_reference import propagate

SHA = "32412cb9782744fa0e8e2b1bc96ace4b596c76fa2e8faf554326b2cbc97c87c3"
FLIGHTS = ("flight-03a-ellipse", "flight-08a-lemniscate", "flight-16a-trackRATM")
ACTIONS = ("wait", "aligned", "legacy")
SERVICE_US = 46686
PERIOD_US = 1_000_000 / 15


def camera_frames(flight):
    camera = DATA / flight / ("camera_" + flight)
    return sorted((int(f.stem.split("_")[1]), f) for f in camera.iterdir() if f.suffix.lower() in (".jpg", ".png"))


def schedule(frames):
    due = frames[0][0]
    for i, (stamp, _) in enumerate(frames):
        if stamp >= due:
            yield i
            due = stamp + PERIOD_US


def prepare_detections(flight, paths, model_holder):
    from revision_cache import frozen
    result = frozen(flight)
    missing = [p for p in paths if p.stem not in result]
    if missing:
        if not model_holder:
            from ultralytics import YOLO
            model_holder.append(YOLO(os.environ["RAM_WEIGHTS"]))
        for image in missing:
            prediction = model_holder[0].predict(str(image), imgsz=640, conf=.25, verbose=False)[0]
            kp = prediction.keypoints
            points = kp.xy.cpu().numpy() if kp is not None else np.empty((0, 4, 2), np.float32)
            conf = kp.conf.cpu().numpy() if kp is not None and kp.conf is not None else np.ones(points.shape[:2], np.float32)
            result[image.stem] = (prediction.boxes.xyxy.cpu().numpy(), points, conf)
    print(flight, "new image inferences", len(missing), "of", len(paths), flush=True)
    return result, len(missing)


def copy_obs(obs):
    return [v.copy() for v in obs]


def advance(obs, previous, current):
    if not obs:
        return
    boxes, points, valid = (np.stack(x) for x in zip(*obs))
    updated = propagate(previous, current, boxes, points, valid)
    for idx, observation in enumerate(obs):
        observation[:] = [x[idx].copy() for x in updated[:3]]


def observations(detection, width, height):
    boxes, points, scores = detection
    valid = (scores >= .5) & np.isfinite(points).all(axis=2)
    valid &= (points[..., 0] >= 0) & (points[..., 0] < width) & (points[..., 1] >= 0) & (points[..., 1] < height)
    return [[b.copy(), p.copy(), v.copy()] for b, p, v in zip(boxes, points, valid) if v.sum() >= 2]


def run(flight, model_holder, *, output_dir=HERE, detection_loader=prepare_detections):
    frames = camera_frames(flight)
    assert all(right[0] > left[0] for left, right in zip(frames, frames[1:]))
    dispatch = set(schedule(frames))
    detections, new = detection_loader(flight, [frames[i][1] for i in sorted(dispatch)], model_holder)
    tracks = {action: {} for action in ACTIONS}
    counters = {action: collections.Counter() for action in ACTIONS}
    pending = None
    previous = None
    calls = []
    pred_file = output_dir / f"predictions_{flight}.jsonl.gz"
    tmp = pred_file.with_suffix(".part")
    start = time.perf_counter()
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as stream:
        for fi, (stamp, path) in enumerate(frames):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise OSError(path)
            if fi and stamp - frames[fi-1][0] > 25000:
                for pool in tracks.values():
                    pool.clear()
                pending = None
                previous = None
            if previous is not None:
                advancing = [item[0] for pool in tracks.values() for item in pool.values()]
                if pending is not None:
                    advancing.extend(pending["transported"])
                advance(advancing, previous, gray)
            previous = gray
            for pool in tracks.values():
                for key in list(pool):
                    if stamp - pool[key][1] > 500000:
                        del pool[key]
            if pending is not None and stamp - pending["stamp"] >= SERVICE_US:
                calls[-1]["return_us"] = stamp
                for action, pool in tracks.items():
                    for j, transported in enumerate(pending["transported"]):
                        target = pending["targets"][action].get(j)
                        if target is None or target not in pool or pool[target][0][2].sum() < 2:
                            if target in pool:
                                del pool[target]
                            if transported[2].sum() >= 2:
                                pool[(pending["fi"], j)] = [copy_obs(transported), stamp]
                                counters[action]["births"] += 1
                        else:
                            pool[target][1] = stamp
                            if action != "wait":
                                use = transported if action == "aligned" else pending["fresh"][j]
                                pool[target][0] = copy_obs(use)
                            counters[action]["retained_or_updated"] += 1
                pending = None
            if fi in dispatch:
                assert pending is None
                fresh = observations(detections[path.stem], gray.shape[1], gray.shape[0])
                targets = {}
                for action, pool in tracks.items():
                    ids = [key for key, item in pool.items() if item[0][2].sum() >= 2]
                    matches, _, _ = associate([pool[k][0][0] for k in ids], [o[0] for o in fresh])
                    targets[action] = {j: ids[i] for i, j in matches}
                pending = dict(stamp=stamp, fi=fi, fresh=fresh, transported=[copy_obs(o) for o in fresh], targets=targets)
                calls.append(dict(flight=flight, capture_us=stamp, return_us=None, detections=len(fresh)))
            frame = dict(stem=path.stem, timestamp_us=stamp)
            for action, pool in tracks.items():
                frame[action] = [[box.tolist(), p.tolist(), v.tolist()] for (box, p, v), _ in pool.values() if v.sum() >= 2]
            stream.write(json.dumps(frame, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(pred_file)
    with (output_dir / f"calls_{flight}.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=calls[0])
        out.writeheader()
        out.writerows(calls)
    delays = [(c["return_us"]-c["capture_us"])/1000 for c in calls if c["return_us"] is not None]
    result = dict(frames=len(frames), calls=len(calls), returns=len(delays), flight_seconds=(frames[-1][0]-frames[0][0])/1e6,
                  new_inferences=new, min_median_max_return_ms=np.percentile(delays,[0,50,100]).tolist(),
                  actions=counters, replay_compute_seconds=time.perf_counter()-start)
    print(flight, result, flush=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        import argparse
        from revision_delay import replay

        parser = argparse.ArgumentParser(description="Score a configured service delay using archived detector outputs.")
        parser.add_argument("--service-ms", type=float, required=True)
        parser.add_argument("--flight", choices=FLIGHTS, required=True)
        args = parser.parse_args()
        if not np.isfinite(args.service_ms) or args.service_ms < .001:
            parser.error("service delay must be at least one microsecond")
        delay = round(args.service_ms * 1000)
        cv2.setNumThreads(1)
        result = replay(args.flight, delay)
        folder = HERE / "results" / "delay_sweep"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{args.flight}_{delay}.json").write_text(json.dumps(result, indent=2))
        sys.exit(0)
    cv2.setNumThreads(1)
    os.environ["YOLO_CONFIG_DIR"] = str(HERE / ".yolo")
    if os.environ.get("RAM_WEIGHTS"):
        assert hashlib.sha256(Path(os.environ["RAM_WEIGHTS"]).read_bytes()).hexdigest() == SHA
    models = []
    results = {flight: run(flight, models) for flight in FLIGHTS}
    results["configuration"] = dict(opencv=cv2.__version__, numpy=np.__version__, weights_sha256=SHA,
                                    service_us=SERVICE_US, detection_period_us=PERIOD_US)
    (HERE / "continuous_runtime.json").write_text(json.dumps(results, indent=2))
