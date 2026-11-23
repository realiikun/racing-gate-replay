"""Continuous fixed-dispatch replay with overlapping in-flight calls and frozen detections."""
import argparse
import csv
import json
from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from continuous import ACTIONS, DATA, FLIGHTS, HERE, SERVICE_US, advance, associate, camera_frames, copy_obs, observations, schedule
from revision_cache import frozen
from score_continuous import ORDERS, iou_matrix
from sensitivity import m, PAIRS

def primary(truth, observations):
    if not truth or not observations:
        return len(truth),len(observations)
    overlap=iou_matrix(truth,[o[0] for o in observations])
    a,b=linear_sum_assignment((overlap>=.3)*(min(overlap.shape)+1+overlap),maximize=True)
    good=matched=0
    for i,j in zip(a,b):
        if overlap[i,j]<.3: continue
        matched+=1
        g=truth[i]
        p,valid=observations[j][1:]
        d=m.gt_diagonal(g)
        hits=(valid[ORDERS] & g["vis"] & (np.linalg.norm(p[ORDERS]-g["corners"],axis=2)<=.05*d))
        x,y=PAIRS
        good+=int(np.any(hits[:,x] & hits[:,y] & (np.linalg.norm(g["corners"][x]-g["corners"][y],axis=1)>=.25*d)))
    return len(truth)-good,len(observations)-matched

def replay(flight,delay):
    frames=camera_frames(flight)
    dispatch=set(schedule(frames))
    cache=frozen(flight)
    folder=DATA/flight/("label_"+flight)
    labels={p.stem:p for p in folder.glob("*.txt")}
    pools={a:{} for a in ACTIONS}
    pending=deque()
    totals={a:np.zeros(2) for a in ACTIONS}
    exposure=0
    previous=None
    returns=0
    for i,(t,path) in enumerate(frames):
        gray=cv2.imread(str(path),0)
        if i and t-frames[i-1][0]>25000:
            for pool in pools.values(): pool.clear()
            pending.clear()
            previous=None
        if previous is not None:
            active=[v[0] for pool in pools.values() for v in pool.values()]
            for call in pending: active.extend(call["moved"])
            advance(active,previous,gray)
        previous=gray
        for pool in pools.values():
            for k in list(pool):
                if t-pool[k][1]>500000: del pool[k]
        while pending and t-pending[0]["t"]>=delay:
            c=pending.popleft()
            returns+=1
            for a,pool in pools.items():
                for j,o in enumerate(c["moved"]):
                    k=c["links"][a].get(j)
                    if k not in pool or pool[k][0][2].sum()<2:
                        pool.pop(k,None)
                        if o[2].sum()>=2: pool[c["i"],j]=[copy_obs(o),t]
                    else:
                        pool[k][1]=t
                        if a!="wait": pool[k][0]=copy_obs(o if a=="aligned" else c["fresh"][j])
        if i in dispatch and t+delay<=frames[-1][0]:
            fresh=observations(cache[path.stem],gray.shape[1],gray.shape[0])
            links={}
            for a,pool in pools.items():
                keys=[k for k,v in pool.items() if v[0][2].sum()>=2]
                matches,_,_=associate([pool[k][0][0] for k in keys],[o[0] for o in fresh])
                links[a]={j:keys[k] for k,j in matches}
            pending.append(dict(t=t,i=i,links=links,fresh=fresh,moved=[copy_obs(o) for o in fresh]))
        if i+1<len(frames) and path.stem in labels and frames[i+1][1].stem in labels:
            dt=(frames[i+1][0]-t)/1e6
            if dt>.025: continue
            truth=[g for g in m.parse_rows(labels[path.stem]) if m.gt_diagonal(g)>1e-6]
            exposure+=dt*len(truth)
            for a,pool in pools.items():
                totals[a]+=dt*np.array(primary(truth,[v[0] for v in pool.values() if v[0][2].sum()>=2]))
    return dict(flight=flight,delay_us=delay,returns=returns,actions={a:dict(bad_percent=100*v[0]/exposure,unmatched_s=v[1]) for a,v in totals.items()})

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("flight",choices=FLIGHTS)
    parser.add_argument("--delays",type=int,nargs="+",default=[SERVICE_US,25000,50000,75000,100000,150000])
    args=parser.parse_args()
    cv2.setNumThreads(1)
    out=HERE/"results/delay_sweep"
    out.mkdir(parents=True,exist_ok=True)
    for delay in args.delays:
        result=replay(args.flight,delay)
        f=out/f"{args.flight}_{delay}.json"
        f.write_text(json.dumps(result,indent=2))
        if delay==SERVICE_US:
            for r in csv.DictReader((HERE/"continuous_scores.csv").open()):
                if r["matching"]=="cyclic" and r["radius"]=="0.05" and r["flight"]==args.flight:
                    assert abs(float(r["bad_percent"])-result["actions"][r["action"]]["bad_percent"])<1e-7,(r,result)
        print(f.name,flush=True)
