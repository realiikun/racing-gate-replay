"""Freeze a duration-stratified RGB sample without opening labels or predictions."""
import os
import csv
import random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RAM_AUTONOMOUS", HERE.parent/"ram"))
FLIGHTS = ("flight-03a-ellipse", "flight-08a-lemniscate", "flight-16a-trackRATM")


def main():
    rng = random.Random(20261123)
    rows, tiles = [], []
    for flight in FLIGHTS:
        frames = sorted((DATA / flight / ("camera_" + flight)).glob("*.jpg"))
        t = [int(f.stem.split("_")[1]) for f in frames]
        for block in range(4):
            lo = t[0] + (t[-1] - t[0]) * block / 4
            hi = t[0] + (t[-1] - t[0]) * (block+1) / 4
            eligible = [i for i in range(len(t)-24) if lo <= t[i] and t[i+24] < hi and max(np.diff(t[i:i+25])) <= 25000]
            start = rng.choice(eligible)
            clip = len(tiles)
            for k, path in enumerate(frames[start:start+25]):
                rows.append((clip, flight, k, path.stem, t[start+k], path.relative_to(DATA).as_posix()))
            im = cv2.resize(cv2.imread(str(frames[start+12])), (320,240))
            cv2.putText(im,str(clip),(5,24),0,.75,(0,255,255),2)
            tiles.append(im)
    with (HERE / "sample.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(("clip","flight","k","stem","us","path"));w.writerows(rows)
    cv2.imwrite(str(HERE/"overview.jpg"),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,12,4)]))


if __name__=="__main__":
    main()
