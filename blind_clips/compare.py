"""Unblind saved continuous predictions only after checking the RGB reference hash."""
import os
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from continuous import DATA, FLIGHTS
from score_continuous import matched_radii, iou_matrix
from sensitivity import m
from seeds import SEEDS

ACTIONS=("wait","aligned")
DIGEST="67e38b9c51b0c1b1d31d2786afeb3689fc60492550157e31dc9fa2823af51645"

def save(name,rows):
    with (HERE/name).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)

def array_gate(g):
    return {k:np.asarray(v) for k,v in g.items()}

def linear_gate(c,name,k,gate):
    i=min(k//12,1); w=(k-12*i)/12
    g=dict(gate); pts=gate["corners"].copy()
    for slot in np.where(gate["vis"])[0]:
        a,b=[np.array(s[1][slot]) for s in SEEDS[c][name][i:i+2]]
        pts[slot]=(1-w)*a+w*b
    g["corners"]=pts
    return g

def bad_flags(truth, predictions, radius):
    return np.column_stack([matched_radii(truth,predictions[a])[0][:,0]>radius for a in ACTIONS])

def old_gate_map(truth,old):
    if not truth or not old:
        return {}
    ious=iou_matrix(truth,[g["bbox"] for g in old])
    rows,cols=linear_sum_assignment((ious>=.3)*(min(ious.shape)+1+ious),maximize=True)
    return {i:j for i,j in zip(rows,cols) if ious[i,j]>=.3}

def stats(data):
    total=sum(r["dt"] for r in data)
    bad=[sum(r["dt"]*r[a] for r in data) for a in ACTIONS]
    return dict(gate_seconds=total,wait=100*bad[0]/total,aligned=100*bad[1]/total,
                advantage_pp=100*(bad[0]-bad[1])/total)

def run():
    raw=(HERE/"independent_reference.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==DIGEST
    refs=json.loads(raw)
    qc=defaultdict(list)
    for q in csv.DictReader((HERE/"registration_qc.csv").open()):
        e=float(q["error_px"]);qc[int(q["clip"]),q["gate"]].append((e,e/float(q["scale"])))
    keep={key for key,v in qc.items() if np.percentile(v,95,axis=0)[0]<=4 and np.percentile(v,95,axis=0)[1]<=.05}
    tests=defaultdict(list)
    for flight in FLIGHTS:
        names={r["stem"] for r in refs if r["flight"]==flight}
        with gzip.open(HERE.parent/f"predictions_{flight}.jsonl.gz","rt") as f:
            pred={r["stem"]:r for r in map(json.loads,f) if r["stem"] in names}
        assert len(pred)==len(names)
        label_dir=DATA/flight/("label_"+flight)
        for c in sorted({r["clip"] for r in refs if r["flight"]==flight}):
            rr=[r for r in refs if r["clip"]==c]
            for r,nxt in zip(rr,rr[1:]):
                dt=(nxt["us"]-r["us"])/1e6
                labels=list(r["gates"])
                truth=[array_gate(g) for g in r["gates"].values()]
                if not truth:continue
                linear=[linear_gate(c,n,r["k"],g) for n,g in zip(labels,truth)]
                predictions=pred[r["stem"]]
                path=label_dir/(r["stem"]+".txt")
                old=[]
                if path.exists() and (label_dir/(nxt["stem"]+".txt")).exists():
                    old=[g for g in m.parse_rows(path) if m.gt_diagonal(g)>1e-6]
                mapping=old_gate_map(truth,old)
                old_flags=bad_flags(old,predictions,.05)
                base=bad_flags(truth,predictions,.05)
                alternative=bad_flags(linear,predictions,.05)
                tight=bad_flags(truth,predictions,.04)
                loose=bad_flags(truth,predictions,.06)
                for j,name in enumerate(labels):
                    row=dict(flight=flight,clip=c,gate=name,k=r["k"],us=r["us"],end_us=nxt["us"],dt=dt)
                    comparisons={"blind":base[j],"linear":alternative[j],"r04":tight[j],"r06":loose[j]}
                    if (c,name) in keep: comparisons["qc"]=base[j]
                    if j in mapping:
                        comparisons["paired_blind"]=base[j];comparisons["paired_txt"]=old_flags[mapping[j]]
                    for variant,flags in comparisons.items():
                        tests[variant].append(dict(**row,wait=int(flags[0]),aligned=int(flags[1])))
    allstats=[]
    for variant,rows in tests.items():
        fs=[]
        for f in FLIGHTS:
            result=stats([r for r in rows if r["flight"]==f])
            allstats.append(dict(variant=variant,flight=f,**result));fs.append(result)
        record={k:np.mean([f[k] for f in fs]) for k in fs[0]}
        record["gate_seconds"]=sum(f["gate_seconds"] for f in fs)
        allstats.append(dict(variant=variant,flight="flight_equal",**record))
    save("comparison.csv",allstats)
    rows=tests["blind"];save("scored_intervals.csv",rows)
    runs=[];clips=[]
    for c,tracks in SEEDS.items():
        subset=[r for r in rows if r["clip"]==c]
        if subset:
            clips.append(dict(clip=c,flight=subset[0]["flight"],**stats(subset)))
        for name in tracks:
            seq=[r for r in subset if r["gate"]==name]
            assert len(seq)==24
            for a in ACTIONS:
                edges=np.flatnonzero(np.diff([0]+[r[a] for r in seq]+[0]))
                for first,end in edges.reshape(-1,2):
                    ms=(seq[end-1]["end_us"]-seq[first]["us"])/1000
                    runs.append(dict(clip=c,gate=name,flight=seq[0]["flight"],action=a,start_us=seq[first]["us"],
                                     end_us=seq[end-1]["end_us"],duration_ms=ms,at_least_50ms=int(ms>=50),
                                     left_clipped=int(first==0),right_clipped=int(end==len(seq))))
    save("failure_runs.csv",runs);save("by_clip.csv",clips)
    loo=[]
    for c in SEEDS:
        loo.append(np.mean([stats([r for r in rows if r["clip"]!=c and r["flight"]==f])["advantage_pp"] for f in FLIGHTS]))
    # Resample temporal snippets, never individual frames, within each existing flight.
    rng=np.random.default_rng(20261123)
    dist=[]
    for f in FLIGHTS:
        ids=sorted({r["clip"] for r in refs if r["flight"]==f})
        data=np.array([[sum(r["dt"] for r in rows if r["clip"]==i),
                        sum(r["dt"]*(r["wait"]-r["aligned"]) for r in rows if r["clip"]==i)] for i in ids])
        v=data[rng.integers(0,len(ids),(20000,len(ids)))].sum(axis=1)
        dist.append(np.divide(100*v[:,1],v[:,0],out=np.full(len(v),np.nan),where=v[:,0]>0))
    bs=np.mean(dist,axis=0)
    summary=dict(reference_hash=DIGEST,qc_tracks=sorted(keep),all_tracks=len(qc),
                 leave_one_clip_range=[min(loo),max(loo)],bootstrap_valid=int(np.isfinite(bs).sum()),
                 conditional_clip_bootstrap_percentiles=np.nanpercentile(bs,[2.5,97.5]).tolist())
    (HERE/"checks.json").write_text(json.dumps(summary,indent=2))
    for row in allstats:
        if row["flight"]=="flight_equal" or row["variant"]=="blind":print(row)
    print(summary)
    for a in ACTIONS:
        filt=[r for r in runs if r["action"]==a and r["at_least_50ms"]]
        print(a,"confirmed runs",len(filt),"observed ms",sum(r["duration_ms"] for r in filt),
              "touch boundary",sum(r["left_clipped"] or r["right_clipped"] for r in filt))
if __name__=="__main__":
    run()
