"""Additional replay on the three recordings sealed in the previous detector study."""
import argparse
import csv
import hashlib
import json
import cv2
from continuous import HERE, DATA, SHA, run
from score_continuous import score
from sensitivity import m

flights=("flight-06a-ellipse","flight-12a-lemniscate","flight-18a-trackRATM")
out=HERE/"results/held_out"

if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("flight",choices=flights)
    flight=arg.parse_args().flight
    out.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    runtime=run(flight,[],output_dir=out)
    meta=out/f"runtime_{flight}.json"
    meta.write_text(json.dumps({flight:runtime},indent=2))
    rows=score(flight,prediction_dir=out,runtime_file=meta)
    with (out/f"scores_{flight}.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
