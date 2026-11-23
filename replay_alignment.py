import numpy as np
from config import HERE
import recording as m
DATA = HERE / "inputs"
def detection(rec,det,fi,tid):
    boxes,points,confidence=det.get(rec.frames[fi]['stem'],rec.frames[fi]['image'])
    gtmap=dict(rec.track_rows.get(fi,[]))
    if tid not in gtmap:return None
    gt=gtmap[tid]
    owners={}
    for tt,gg in gtmap.items():
        js=[j for j,b in enumerate(boxes) if m.iou(b,gg['bbox'])>=.5]
        if len(js)==1:owners[js[0]]=owners.get(js[0],0)+1
    js=[j for j,b in enumerate(boxes) if m.iou(b,gt['bbox'])>=.5]
    if len(js)!=1:return None
    j=js[0]
    if owners.get(j)!=1:return None
    order=m.slot_mapping(gt['corners'],points[j],mask=gt['vis']&(confidence[j]>=.5)&np.isfinite(points[j]).all(axis=1),diagonal=max(m.gt_diagonal(gt),1.))
    if order is None:return None
    p=points[j][order]
    v=(confidence[j][order]>=.5)&np.isfinite(p).all(axis=1)&(p[:,0]>=0)&(p[:,0]<m.W)&(p[:,1]>=0)&(p[:,1]<m.H)
    return p,v,float(confidence[j][order].mean())

def transport(rec,cache,first,last,p,v):
    p,v=p.copy(),v.copy()
    box=m.StatePass._box(p,v,None)
    for fi in range(first+1,last+1):
        out=m.propagate(cache.get(fi-1),cache.get(fi),box[None],p[None],v[None])
        p,v=out[1][0],out[2][0]
        if v.sum()>=2:box=m.StatePass._box(p,v,box)
    return p,v
