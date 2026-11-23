"""Render coordinate guides for blind viewing of native RGB images."""
import os
import csv
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
def grid(im):
    layer=im.copy()
    for x in range(0,640,40):
        cv2.line(layer,(x,0),(x,479),(110,170,200),1)
    for y in range(0,480,40):
        cv2.line(layer,(0,y),(639,y),(110,170,200),1)
    im=cv2.addWeighted(im,.72,layer,.28,0)
    for x in range(0,640,40):
        for y in (14,239,475):
            cv2.putText(im,str(x),(x+1,y),0,.33,(0,220,255),1)
    for y in range(40,480,40):
        for x in (2,321):
            cv2.putText(im,str(y),(x,y-1),0,.33,(0,220,255),1)
    return im

if __name__=="__main__":
    rows=list(csv.DictReader((HERE/"sample.csv").open()))
    for c in range(12):
        sel=[r for r in rows if int(r["clip"])==c and int(r["k"]) in (0,12,24)]
        cv2.imwrite(str(HERE/f"view_{c:02}.jpg"),np.vstack([grid(cv2.imread(str(Path(os.environ.get("RAM_AUTONOMOUS", HERE.parent/"ram"))/r["path"]))) for r in sel]))
