"""Replay one original camera stream and compare all three trajectories frame by frame."""
import argparse
import gzip
import json
from itertools import zip_longest
from pathlib import Path
import cv2
import numpy as np
from continuous import ACTIONS,FLIGHTS,HERE,run
from revision_additional import flights as additional

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("flight",choices=(*FLIGHTS,*additional))
    cli.add_argument("--output",required=True,type=Path,help="new or empty directory for generated records")
    args=cli.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        cli.error("Use a new or empty output directory")
    args.output.mkdir(exist_ok=True,parents=True)
    cv2.setNumThreads(1)
    runtime=run(args.flight,[],output_dir=args.output)
    (args.output/"runtime.json").write_text(json.dumps(runtime,indent=2))
    folder=HERE if args.flight in FLIGHTS else HERE/"results/held_out"
    name=f"predictions_{args.flight}.jsonl.gz"
    worst=0.
    with gzip.open(folder/name,"rt") as a,gzip.open(args.output/name,"rt") as b:
        for r1,r2 in zip_longest(map(json.loads,a),map(json.loads,b)):
            assert r1 and r2 and r1["stem"]==r2["stem"] and r1["timestamp_us"]==r2["timestamp_us"]
            for action in ACTIONS:
                assert len(r1[action])==len(r2[action]),(r1["stem"],action)
                for before,after in zip(r1[action],r2[action]):
                    assert before[2]==after[2]
                    for j in [0,1]:
                        worst=max(worst,float(np.max(np.abs(np.asarray(before[j])-after[j]))))
    print("Maximum box/corner discrepancy (pixels):",worst,flush=True)
    assert worst<.001
if __name__=="__main__":main()
