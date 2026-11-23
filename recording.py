import collections
import math
import io
from pathlib import Path
import numpy as np
from config import DATA
from revision_cache import frozen
from lk_reference import propagate
from causal_tracker import slot_mapping

W,H=640,480
SLOT=.25
DET_DT=1/15
HORIZON,SUPPORT,D_SERVICE=.25,.05,.046686
DEV=("flight-03a-ellipse","flight-08a-lemniscate","flight-16a-trackRATM")
def parse_rows(path):
    out = []
    for line in io.open(path, encoding='utf-8-sig', errors='replace'):
        vals = [float(x) for x in line.split()]
        if len(vals) != 17 or not all(math.isfinite(v) for v in vals):
            continue
        cx, cy, w, h = vals[1:5]
        if w <= 0 or h <= 0:
            continue
        kp = np.asarray(vals[5:17], float).reshape(4, 3)
        out.append(dict(bbox=np.array([(cx - w / 2) * W, (cy - h / 2) * H,
                                       (cx + w / 2) * W, (cy + h / 2) * H]),
                        corners=kp[:, :2] * np.array([W, H]),
                        vis=kp[:, 2] == 2))
    return out

class Record:
    def __init__(self, rec):
        self.rec = rec
        base = DATA / rec
        lab = next(base.glob('label_*'))
        cam = next(base.glob('camera_*'))
        frames = []
        for p in sorted(lab.glob('*.txt')):
            try:
                tcap = int(p.stem.split('_')[1])
            except (IndexError, ValueError):
                continue
            img = cam / (p.stem + '.jpg')
            if not img.exists():
                img = cam / (p.stem + '.png')
            if not img.exists():
                continue
            frames.append(dict(tcap=tcap, stem=p.stem, label=p, image=img))
        frames.sort(key=lambda f: f['tcap'])
        if not frames:
            raise RuntimeError(f'no frames for {rec}')
        self.t0 = frames[0]['tcap']
        for f in frames:
            f['t'] = (f['tcap'] - self.t0) / 1e6
            f['rows'] = None
        self.frames = frames
        # Lazily parsed annotations, one list per frame.
        self._parsed = {}
        self.tracks = None
        self.track_rows = None      # fi -> list of (tid, gt)

    def rows(self, fi):
        if fi not in self._parsed:
            self._parsed[fi] = parse_rows(self.frames[fi]['label'])
        return self._parsed[fi]

    def build_tracks(self):
        """Chain annotation detections into gate tracks (identity reference)."""
        tracks = []
        active = {}                 # tid -> (bbox, last_tcap)
        track_rows = collections.defaultdict(list)
        for fi, frame in enumerate(self.frames):
            used = set()
            for gt in self.rows(fi):
                best, best_tid = -1.0, None
                for tid, (bb, last_t) in active.items():
                    if tid in used or frame['tcap'] - last_t > 50000:
                        continue
                    inter = np.maximum(0, np.minimum(bb[2:], gt['bbox'][2:])
                                       - np.maximum(bb[:2], gt['bbox'][:2])).prod()
                    union = (np.maximum(0, bb[2:] - bb[:2]).prod()
                             + np.maximum(0, gt['bbox'][2:] - gt['bbox'][:2]).prod()
                             - inter)
                    iou = inter / max(union, 1e-9)
                    if iou < 0.3:
                        continue
                    c1 = (bb[:2] + bb[2:]) / 2
                    c2 = (gt['bbox'][:2] + gt['bbox'][2:]) / 2
                    diag = max(bb[2] - bb[0], 1e-6)
                    if np.linalg.norm(c1 - c2) <= 0.5 * diag and iou > best:
                        best, best_tid = iou, tid
                if best_tid is not None:
                    used.add(best_tid)
                    active[best_tid] = (gt['bbox'], frame['tcap'])
                    tid = best_tid
                else:
                    tid = len(tracks)
                    tracks.append(dict(tid=tid, rec=self.rec, dets={}, last_tcap=None))
                    active[tid] = (gt['bbox'], frame['tcap'])
                tracks[tid]['dets'][fi] = gt
                tracks[tid]['last_tcap'] = frame['tcap']
                track_rows[fi].append((tid, gt))
        self.tracks = tracks
        self.track_rows = track_rows
        return tracks

    def detection_frames(self):
        out, next_t = [], 0.0
        for fi, frame in enumerate(self.frames):
            if frame['t'] + 1e-9 >= next_t:
                out.append(fi)
                next_t = frame['t'] + DET_DT
        return out

    def anchor_frames(self):
        out, k = [], 0
        n = len(self.frames)
        while True:
            at = k * SLOT
            fi = bisect.bisect_left([f['t'] for f in self.frames], at - 1e-9)
            if fi >= n:
                break
            out.append((k, fi))
            k += 1
        return out

def iou(a, b):
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = (np.maximum(0, a[2:] - a[:2]).prod()
             + np.maximum(0, b[2:] - b[:2]).prod() - inter)
    return float(inter / max(union, 1e-9))

def gt_diagonal(gt):
    c, vis = gt['corners'], gt['vis']
    if vis.sum() < 2:
        return 0.0
    return float(max(np.linalg.norm(c[i] - c[j]) for i in range(4) for j in range(i) if vis[i] and vis[j]))

def judge_state(points, valid, gt, identity_known=True):
    D = gt_diagonal(gt)
    if D <= 1e-6:
        return 'non_opportunity', D
    return u2(points, valid, gt['corners'], gt['vis'], D, identity_known=identity_known), D

class FrameCache:
    def __init__(self, record):
        self.rec = record
        self.cache = collections.OrderedDict()

    def get(self, fi):
        import cv2
        if fi in self.cache:
            self.cache.move_to_end(fi)
            return self.cache[fi]
        gray = cv2.imread(str(self.rec.frames[fi]['image']), cv2.IMREAD_GRAYSCALE)
        self.cache[fi] = gray
        if len(self.cache) > 160:
            self.cache.popitem(last=False)
        return gray

def run_branch(record, cache, fi0, points0, valid0, tid, anchor_t,
               update_fi=None, update_points=None, update_valid=None):
    """LK continuation; optional one published detection at update_fi.

    Returns (times, statuses, downtime_seconds) where downtime counts the bad
    time within [anchor, anchor+HORIZON] on the judged sequence.
    """
    points, valid = points0.copy(), valid0.copy()
    box = StatePass._box(points, valid, None)
    times, statuses = [], []
    prev_gray = cache.get(fi0)
    fi = fi0
    end_t = anchor_t + HORIZON + SUPPORT + 1e-6
    while fi + 1 < len(record.frames):
        nfi = fi + 1
        t = record.frames[nfi]['t']
        if t > end_t:
            break
        gray = cache.get(nfi)
        if prev_gray is None or gray is None:
            break
        out = propagate(prev_gray, gray, box[None], points[None], valid[None])
        points, valid = out[1][0], out[2][0]
        if valid.sum() >= 2:
            p = points[valid]
            box = np.array([*p.min(axis=0), *p.max(axis=0)])
        if update_fi is not None and nfi == update_fi and update_points is not None:
            points, valid = update_points.copy(), update_valid.copy()
            if valid.sum() >= 2:
                p = points[valid]
                box = np.array([*p.min(axis=0), *p.max(axis=0)])
        gt = dict(record.track_rows.get(nfi, [])).get(tid)
        if gt is None:
            break                          # annotation gap -> right censor
        status, _ = judge_state(points, valid, gt)
        if status in ('non_opportunity', 'identity_unknown'):
            break                          # not scoreable -> right censor
        times.append(t)
        statuses.append(status)
        prev_gray = gray
        fi = nfi
    bad = 0.0
    for i in range(len(times) - 1):
        if statuses[i] == 'bad' and times[i] <= anchor_t + HORIZON:
            bad += min(times[i + 1], anchor_t + HORIZON) - times[i]
    return times, statuses, bad

class StatePass:
    @staticmethod
    def _box(points, valid, fallback):
            if valid.sum() >= 2:
                p = points[valid]
                return np.array([*p.min(axis=0), *p.max(axis=0)])
            return fallback if fallback is not None else np.array([0, 0, W, H], float)

def u2(points,valid,reference,scoreable,diagonal,identity_known=True):
    if not identity_known:return 'identity_unknown'
    reference=np.asarray(reference,float);mask=np.asarray(scoreable,bool)&np.isfinite(reference).all(axis=1)
    opportunity=any(mask[i] and mask[j] and np.linalg.norm(reference[i]-reference[j])>=.25*diagonal for i in range(4) for j in range(i))
    if not opportunity:return 'non_opportunity'
    if points is None:return 'bad'
    hit=mask&np.asarray(valid,bool)&np.isfinite(points).all(axis=1)&(np.linalg.norm(np.asarray(points)-reference,axis=1)<=.05*diagonal)
    return 'good' if any(hit[i] and hit[j] and np.linalg.norm(reference[i]-reference[j])>=.25*diagonal for i in range(4) for j in range(i)) else 'bad'

class YoloSource:
    def __init__(self, flight):
        self.cache = frozen(flight)

    def get(self, stem, image):
        return self.cache[stem]
