import numpy as np
from scipy.optimize import linear_sum_assignment

def associate(prior,observed):
    """Image-only assignment. Ambiguous near-ties remain unmatched, not truth reset."""
    prior=np.asarray(prior);observed=np.asarray(observed)
    if not len(prior) or not len(observed): return [],list(range(len(prior))),list(range(len(observed)))
    cost=np.full((len(prior),len(observed)),1e6)
    for i,a in enumerate(prior):
        for j,b in enumerate(observed):
            inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
            iou=inter/max(union,1);d=max(np.linalg.norm(a[2:]-a[:2]),1)
            distance=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/d
            if iou>=.3 and distance<=.5:cost[i,j]=1-iou+distance
    matches=[]
    for i,j in zip(*linear_sum_assignment(cost)):
        row=np.sort(cost[i]);col=np.sort(cost[:,j])
        if cost[i,j]<1e5 and (len(row)<2 or row[1]-row[0]>=.1) and (len(col)<2 or col[1]-col[0]>=.1): matches.append((int(i),int(j)))
    return matches,[i for i in range(len(prior)) if i not in [x[0] for x in matches]],[j for j in range(len(observed)) if j not in [x[1] for x in matches]]

def slot_mapping(prior,fresh,mask,diagonal):
    choices=[]
    for reverse in [False,True]:
        base=np.arange(4)[::-1] if reverse else np.arange(4)
        for shift in range(4):
            order=np.roll(base,shift);use=np.asarray(mask,bool)&np.isfinite(prior).all(axis=1)&np.isfinite(fresh[order]).all(axis=1)
            if use.sum()>=2: choices.append((float(np.mean(np.linalg.norm(prior[use]-fresh[order][use],axis=1)))/max(diagonal,1),order))
    choices.sort(key=lambda x:x[0])
    return None if len(choices)<2 or choices[1][0]-choices[0][0]<.05 else choices[0][1]
