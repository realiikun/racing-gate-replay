"""Compare the same detector slots and reference gate across capture and return."""
import csv
import json
from bisect import bisect_left
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from continuous import HERE, FLIGHTS, camera_frames, observations, advance
from revision_cache import frozen
from score_continuous import iou_matrix, ORDERS
from sensitivity import m

def match(gates, obs):
    if not gates or not obs:
        return {}
    mat=iou_matrix([g for _,g in gates],[v[0] for v in obs])
    rr,cc=linear_sum_assignment((mat>=.3)*(min(mat.shape)+1+mat),maximize=True)
    return {j:gates[i] for i,j in zip(rr,cc) if mat[i,j]>=.3}

def slots(g,o):
    # Maximise paired visible slots before minimising their mean squared distance.
    def cost(order):
        mask=o[2] & g["vis"][order]
        return (-int(mask.sum()),np.mean(np.square(o[1][mask]-g["corners"][order][mask])) if mask.any() else np.inf)
    order=min(ORDERS,key=cost)
    return {i:np.linalg.norm(o[1][i]-g["corners"][order[i]])/m.gt_diagonal(g) for i in np.flatnonzero(o[2] & g["vis"][order])}

rows=[]
counts={}
cv2.setNumThreads(1)
for flight in FLIGHTS:
    rec=m.Record(flight)
    rec.build_tracks()
    truths={r["stem"]:[(t,g) for t,g in rec.track_rows.get(i,[]) if m.gt_diagonal(g)>1e-6] for i,r in enumerate(rec.frames)}
    imgs=camera_frames(flight)
    stamps=[t for t,_ in imgs]
    data=frozen(flight)
    n=Counter()
    for call in csv.DictReader((HERE/f"calls_{flight}.csv").open()):
        n["calls"]+=1
        if not call["return_us"]: continue
        a,b=[bisect_left(stamps,int(call[k])) for k in ["capture_us","return_us"]]
        pa,pb=imgs[a][1],imgs[b][1]
        first,last=truths.get(pa.stem,[]),truths.get(pb.stem,[])
        if not first or not last: continue
        n["both_scoreable"]+=1
        gray=cv2.imread(str(pa),0)
        obs=observations(data[pa.stem],gray.shape[1],gray.shape[0])
        start=match(first,obs)
        transport=[[v.copy() for v in o] for o in obs]
        for _,path in imgs[a+1:b+1]:
            next_gray=cv2.imread(str(path),0)
            advance(transport,gray,next_gray)
            gray=next_gray
        end=match(last,transport)
        size=len(rows)
        for j,(tid,g) in start.items():
            if j not in end or end[j][0]!=tid: continue
            raw,ret=slots(g,obs[j]),slots(end[j][1],transport[j])
            for k in sorted(raw.keys() & ret.keys()):
                rows.append(dict(flight=flight,capture=pa.stem,returned=pb.stem,tid=tid,slot=int(k),detection=raw[k],returned_error=ret[k],ratio=ret[k]/raw[k] if raw[k]>0 else None))
        n["paired_calls"]+=int(len(rows)>size)
    counts[flight]=n
    print(flight,n,flush=True)
out=HERE/"results"
with (out/"error_decomposition.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary={}
lines=["| Flight | Paired calls | Corners | Capture median / P75 / P95 | Return median / P75 / P95 | Ratio median / P75 / P95 |","|---|---:|---:|---|---|---|"]
for flight in (*FLIGHTS,"pooled"):
    sub=[r for r in rows if flight=="pooled" or r["flight"]==flight]
    item={}
    for k in ["detection","returned_error","ratio"]:
        values=np.array([r[k] for r in sub if r[k] is not None])
        item[k]=dict(quantiles=np.percentile(values,[50,75,95]).tolist(),n=len(values))
        if k!="ratio": item[k]["proportion_above"]={v:float(np.mean(values>v)) for v in [.025,.05,.075,.10]}
    summary[flight]=item
    calls=len({(r["flight"],r["capture"]) for r in sub})
    lines.append(f"| {flight} | {calls} | {len(sub)} | "+" | ".join(" / ".join(f"{x:.4f}" for x in item[k]["quantiles"]) for k in item)+" |")
(out/"error_decomposition.json").write_text(json.dumps(dict(counts=counts,distribution=summary),indent=2),encoding="utf8")
lines+=["","Matched on IoU ≥0.3 at both endpoints and on gate identity across endpoints. Each endpoint uses its own visible-gate scale. The cyclic or reversed slot order maximises visible correspondences then minimises squared distance. Ratios pair surviving detector slots. Return error includes the initial detector error, optical flow and reference variation. Missing returns and unmatched or lost corners are outside this conditional sample."]
(out/"error_decomposition.md").write_text("\n".join(lines),encoding="utf8")
