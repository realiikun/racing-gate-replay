"""RGB-only annotation and QC; no detector weights, predictions, or TXT inputs."""
import os
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from seeds import SEEDS

HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader((HERE/"sample.csv").open()))

def register(frames, point, first, last):
    step=1 if last>first else -1
    xy=np.array(point,dtype=float)
    positions={first:xy.copy()}
    for i in range(first,last,step):
        template=cv2.getRectSubPix(frames[i], (15,15),tuple(xy))
        window=cv2.getRectSubPix(frames[i+step], (53,53),tuple(xy))
        corr=cv2.matchTemplate(window,template,cv2.TM_CCOEFF_NORMED)
        _,_,_,(x,y)=cv2.minMaxLoc(corr)
        def sub(a,b,c):
            denom=a+c-2*b
            return np.clip((a-c)/(2*denom),-.5,.5) if denom < -1e-6 else 0.
        dx=sub(corr[y,x-1],corr[y,x],corr[y,x+1]) if 0<x<38 else 0
        dy=sub(corr[y-1,x],corr[y,x],corr[y+1,x]) if 0<y<38 else 0
        xy+=np.array([x-19+dx,y-19+dy])
        positions[i+step]=xy.copy()
    return positions

def generate():
    output=[]; checks=[]; panels=[]
    for c,tracks in SEEDS.items():
        rr=[r for r in rows if int(r["clip"])==c]
        images=[cv2.imread(str(Path(os.environ.get("RAM_AUTONOMOUS", HERE.parent/"ram"))/r["path"])) for r in rr]
        frames=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]
        annotations=[{} for _ in frames]
        for name,anchors in tracks.items():
            for n,(a,b) in enumerate(zip(anchors, anchors[1:])):
                first,last=12*n,12*(n+1)
                pts=np.zeros((13,4,2))
                vis=np.array([q is not None for q in a[1]])
                scales=[max(np.linalg.norm(np.subtract(u,v)) for u in p[1] if u for v in p[1] if v) for p in (a,b)]
                for slot in np.where(vis)[0]:
                    left=register(frames,a[1][slot],first,last)
                    right=register(frames,b[1][slot],last,first)
                    for k in range(first,last+1):
                        w=(k-first)/12
                        pts[k-first,slot]=left[k]*(1-w)+right[k]*w
                        checks.append(dict(clip=c,gate=name,slot=int(slot),k=k,
                                           error_px=float(np.linalg.norm(left[k]-right[k])),
                                           scale=(1-w)*scales[0]+w*scales[1]))
                for i in range(13):
                    w=i/12
                    annotations[first+i][name]=dict(bbox=((1-w)*np.array(a[0])+w*np.array(b[0])).tolist(),
                                                     corners=pts[i].round(3).tolist(),vis=vis.tolist())
        for k,(r,targets) in enumerate(zip(rr,annotations)):
            output.append(dict(clip=c,k=k,flight=r["flight"],stem=r["stem"],us=int(r["us"]),gates=targets))
            for target in targets.values():
                for pt,v in zip(target["corners"],target["vis"]):
                    if v:
                        xy=tuple(np.rint(pt).astype(int))
                        cv2.circle(images[k],xy,5,(0,240,255),1)
                        cv2.circle(images[k],xy,1,(0,0,255),-1)
        if tracks:
            # Inspect intermediate frames as well as the three manual anchors.
            tiles=[]
            for k in (0,3,6,9,12,15,18,21,24):
                im=cv2.resize(images[k],(320,240))
                cv2.putText(im,str(k),(2,17),0,.5,(0,255,255),1)
                tiles.append(im)
            cv2.imwrite(str(HERE/f"qa_{c:02}.jpg"),np.vstack([np.hstack(tiles[i:i+3]) for i in (0,3,6)]))
            panels.append(cv2.resize(images[6],(320,240)))
        else:
            panels.append(cv2.resize(images[6],(320,240)))
    manifest=HERE/"independent_reference.json"
    manifest.write_text(json.dumps(output,separators=(",",":")),encoding="utf-8")
    with (HERE/"registration_qc.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=checks[0]);w.writeheader();w.writerows(checks)
    cv2.imwrite(str(HERE/"qa_overview.jpg"),np.vstack([np.hstack(panels[i:i+4]) for i in (0,4,8)]))
    print("SHA256",hashlib.sha256(manifest.read_bytes()).hexdigest())
    for c,tracks in SEEDS.items():
        for g in tracks:
            err=[v["error_px"] for v in checks if v["clip"]==c and v["gate"]==g]
            print(c,g,np.round(np.percentile(err,[50,95,100]),1))

if __name__=="__main__":
    generate()
