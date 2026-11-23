"""Repeat the recorded 72 decode-plus-inference samples on the current workstation."""
import argparse, collections, hashlib, json, time
from pathlib import Path
import cv2
import numpy as np
import torch
from detector_adapter import YoloDetector
def dump(path,value):
    path.write_text(json.dumps(value,indent=2))
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--raw",type=Path,required=True,help="root containing autonomous and piloted subfolders")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("output must be a new file")
    weights=args.weights
    assert hashlib.sha256(weights.read_bytes()).hexdigest()=="32412cb9782744fa0e8e2b1bc96ace4b596c76fa2e8faf554326b2cbc97c87c3"
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(20261106)
    cv2.setRNGSeed(20261106)
    records=json.loads(Path(__file__).with_name("frames.json").read_text())
    rows=[dict(r,image_path=args.raw/r["image_path"]) for r in records]
    windows=collections.defaultdict(list)
    for row in rows:windows[row["window_id"]].append(row)
    def sync():
        if torch.cuda.is_available():torch.cuda.synchronize()
    loaded_at=time.perf_counter()
    detector=YoloDetector(weights)
    first=cv2.imread(rows[0]['image_path']);assert first is not None
    for _ in range(3):detector(first);sync()
    warmup_s=time.perf_counter()-loaded_at
    timing=[]
    for wid, seq in windows.items():
        for ix in np.linspace(0,len(seq)-1,8).round().astype(int):
            r=seq[int(ix)];data=Path(r['image_path']).read_bytes();assert hashlib.sha256(data).hexdigest()==r['image_sha256']
            t=time.perf_counter();im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR);decoded=time.perf_counter()
            boxes,pts,confidence,valid=detector(im);sync();finished=time.perf_counter()
            assert pts.ndim==3 and pts.shape[1:]==(4,2) and confidence.shape==valid.shape==pts.shape[:2]
            assert np.isfinite(pts).all() and np.isfinite(boxes).all()
            timing.append(dict(window=wid,source_id=r['source_id'],decode_ms=(decoded-t)*1000,inference_ms=(finished-decoded)*1000,
                               total_ms=(finished-t)*1000,gates=len(boxes)))
    p95=float(np.percentile([x['total_ms'] for x in timing],95))
    dump(args.output,dict(n=len(timing),warmup_calls=3,warmup_and_load_s=warmup_s,
        median_service_ms=float(np.median([x['total_ms'] for x in timing])),p95_service_ms=p95,
        mean_service_ms=float(np.mean([x['total_ms']for x in timing])),samples=timing,
        note='source-window timing diagnostic; not a calibrated inference error or final same-budget qualification'))

if __name__=='__main__':main()
