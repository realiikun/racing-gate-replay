"""Post-unblinding diagnostics; never modifies the frozen RGB reference."""
import os
import csv
import gzip
import hashlib
import json
from collections import defaultdict
import numpy as np
from compare import HERE, FLIGHTS, DATA, DIGEST, array_gate, bad_flags, old_gate_map, stats, save, m
from score_continuous import ORDERS

def run():
    raw=(HERE/"independent_reference.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==DIGEST
    refs=json.loads(raw); corner_rows=[]; locked=[]; manual=[]
    for f in FLIGHTS:
        rr=[r for r in refs if r["flight"]==f and r["gates"]]
        stems={r["stem"] for r in rr}
        with gzip.open(HERE.parent/f"predictions_{f}.jsonl.gz","rt") as src:
            predictions={r["stem"]:r for r in map(json.loads,src) if r["stem"] in stems}
        folder=DATA/f/("label_"+f)
        for r in rr:
            gates=[array_gate(g) for g in r["gates"].values()]
            old=[g for g in m.parse_rows(folder/(r["stem"]+".txt")) if m.gt_diagonal(g)>1e-6]
            mapping=old_gate_map(gates,old)
            # Keep original truth boxes and all competitor gates in the matcher.
            combined=old.copy()
            for i,j in mapping.items():
                a,b=gates[i],old[j]
                combined[j]=dict(a,bbox=b["bbox"])
                possibilities=[]
                for order in ORDERS:
                    common=a["vis"] & b["vis"][order]
                    error=np.linalg.norm(a["corners"][common]-b["corners"][order][common],axis=1)
                    if len(error)>=2:
                        possibilities.append((-len(error),error.mean(),error))
                if possibilities:
                    _,_,error=min(possibilities,key=lambda item:item[:2])
                    for e in error:
                        corner_rows.append(dict(flight=f,clip=r["clip"],gate=list(r["gates"])[i],k=r["k"],
                                                px=e,relative=e/m.gt_diagonal(a)))
            locked_flags=bad_flags(combined,predictions[r["stem"]],.05)
            blind_flags=bad_flags(gates,predictions[r["stem"]],.05)
            if r["k"] in (0,12,24):
                for w,a in blind_flags:
                    manual.append(dict(flight=f,dt=1,wait=w,aligned=a))
            if r["k"]<24:
                nxt=next(s for s in rr if s["clip"]==r["clip"] and s["k"]==r["k"]+1)
                for i,j in mapping.items():
                    w,a=locked_flags[j]
                    locked.append(dict(flight=f,dt=(nxt["us"]-r["us"])/1e6,wait=w,aligned=a))
    save("corner_disagreement.csv",corner_rows)
    for name,rows in (("original_assignment_blind_corners",locked),("keyframes_only",manual)):
        by=[stats([r for r in rows if r["flight"]==f])["advantage_pp"] for f in FLIGHTS]
        print(name,by,np.mean(by))
    for name,rows in (("all",corner_rows),("visual_keyframes",[r for r in corner_rows if r["k"] in (0,12,24)])):
        print(name,len(rows), "pixel median/P95",np.percentile([r["px"] for r in rows],[50,95]),
              "relative median/P95",np.percentile([r["relative"] for r in rows],[50,95]))
    base=list(csv.DictReader((HERE/"scored_intervals.csv").open()))
    segments=list(csv.DictReader((HERE/"failure_runs.csv").open()))
    rates={}
    for a in ("wait","aligned"):
        rates[a]=[]
        for f in FLIGHTS:
            exposure=sum(float(r["dt"]) for r in base if r["flight"]==f)
            sec=sum(float(r["duration_ms"])/1000 for r in segments if r["action"]==a and r["flight"]==f and r["at_least_50ms"]=="1")
            rates[a].append(100*sec/exposure)
    print("sustained equal-flight", {a:float(np.mean(v)) for a,v in rates.items()})
if __name__=="__main__":
    run()
