from pathlib import Path
import csv, gzip, json, pickle, sys
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon
from matplotlib import font_manager

from config import HERE as replay, DATA
from revision_cache import frozen
from lk_reference import propagate
out=replay/"figures"
out.mkdir(exist_ok=True)
flight = 'flight-08a-lemniscate'
camera = DATA/flight/('camera_'+flight)
frames = [next(camera.glob(f'{i:05d}_*.jpg')) for i in range(1408,1415)]
cap,ret = [int(p.stem.split('_')[1]) for p in (frames[0],frames[-1])]
call=next(r for r in csv.DictReader((replay/('calls_'+flight+'.csv')).open()) if int(r['capture_us'])==cap)
assert int(call['return_us'])==ret and ret-cap==49803
boxes,points,scores=frozen(flight)[frames[0].stem]
which=0
valid=(scores>=.5)&np.isfinite(points).all(axis=2)&(points[:,:,0]>=0)&(points[:,:,0]<640)&(points[:,:,1]>=0)&(points[:,:,1]<480)
records=[]
imgs=[np.array(Image.open(p).convert('RGB')) for p in frames]
for i,im in enumerate(imgs):
    if i:
        boxes,points,valid,*_=propagate(cv2.imread(str(frames[i-1]),0),cv2.imread(str(frames[i]),0),boxes,points,valid)
    assert valid[which].all()
    records.append(points[which].copy())
with gzip.open(replay/('predictions_'+flight+'.jsonl.gz'),'rt') as f:
    saved=next(v for v in map(json.loads,f) if v['stem']==frames[-1].stem)
for k,pts in [('legacy',records[0]),('aligned',records[-1])]:
    err=min(np.max(np.abs(np.array(o[1])-pts)) for o in saved[k])
    assert err<.15,(k,err)
    print(k,'max coordinate discrepancy vs saved return',err)
allpts=np.concatenate(records)
centre=(allpts.min(0)+allpts.max(0))/2
span=allpts.max(0)-allpts.min(0)
ch=int(np.ceil(max(span[1]+28,(span[0]+28)*48.7/62)))
cw=round(ch*62/48.7)
x0,y0=np.round(centre-[cw/2,ch/2]).astype(int)
assert x0>=0 and y0>=0 and x0+cw<=640 and y0+ch<=480
assert ((allpts>[x0+7,y0+7]) & (allpts<[x0+cw-7,y0+ch-7])).all()

plt.rcParams.update({'font.family':['Times New Roman','DejaVu Serif'],'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix','svg.hashsalt':'IR-method-gates'})
W=(11906-2*879)/20
ratio=W/(158/25.4*72)
H=405
fig=plt.figure(figsize=(W/72,H/72),facecolor='white')
ax=fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(H,0))
ax.set_axis_off()
ink='#212c32'; muted='#45535c'; teal='#147781'; pale='#e4eff3'; sand='#fae5cf'; light='#f3f4f3'; hair='#a8b1b3'
def t(x,y,s,size=8.1,ha='left',c=ink,b=False,it=False):
    return ax.text(x*ratio,y,s,fontsize=size,ha=ha,va='center',color=c,weight='bold' if b else 'normal',style='italic' if it else 'normal',linespacing=1.08,zorder=6)
def ln(coords,c=ink,lw=1.4,dash=None):
    x,y=zip(*coords)
    ax.plot(np.array(x)*ratio,y,color=c,lw=lw,linestyle=dash or '-',solid_capstyle='butt',dash_capstyle='butt',zorder=3)
def edge(coords,c=ink,lw=1.45,dash=None):
    ln(coords,c,lw,dash)
    a,b=np.array(coords[-2:])*[ratio,1];u=(b-a)/np.linalg.norm(b-a);n=np.array([-u[1],u[0]])
    ax.add_patch(Polygon([b,b-4.2*u+1.9*n,b-4.2*u-1.9*n],closed=True,facecolor=c,edgecolor=c,lw=.4,zorder=4))
def node(x,y,w,h,title,sub,fc):
    ax.add_patch(FancyBboxPatch((x*ratio,y),w*ratio,h,boxstyle='round,pad=0,rounding_size=2.1',fc=fc,ec=ink,lw=1.23,zorder=2))
    t(x+w/2,y+11,title,8.55,'center',b=True)
    t(x+w/2,y+h/2+5.4,sub,7.9,'center')
def divider(y):ln([(6,y),(W/ratio-6,y)],hair,.64,(0,(3,2.4)))
def header(x,y,letter,title):
    t(x,y,letter,10.4,c=teal,b=True)
    t(x+13,y,title,10.4,b=True)
def lightbox(x,y,w,h):
    ax.add_patch(Rectangle((x*ratio,y),w*ratio,h,fc=light,ec='none',zorder=0))

header(6,9,'A','One delayed detector call')
t(441,9,'Flight 08a',8.1,'right',it=True,c=muted)
labels=['Capture','Next frame','Next frame','Return']
positions=[7,78,149,230]
for j,(idx,x) in enumerate(zip([0,1,2,6],positions)):
    w,h=62*ratio,48.7*ratio; y=19; x=x*ratio
    crop=imgs[idx][y0:y0+ch,x0:x0+cw]
    ax.imshow(crop,extent=(x,x+w,y+h,y),aspect='auto',interpolation='lanczos',zorder=1)
    ax.add_patch(Rectangle((x,y),w,h,fill=False,ec=ink,lw=.65,zorder=3))
    fx=lambda pts:x+(pts[:,0]-x0)*w/cw
    fy=lambda pts:y+(pts[:,1]-y0)*h/ch
    if idx==6:
        ax.scatter(fx(records[0]),fy(records[0]),s=13,marker='o',facecolors='none',edgecolors='#ffffff',linewidths=.9,zorder=4)
    ax.scatter(fx(records[idx]),fy(records[idx]),s=11,marker='o',facecolors='#ffcb53',edgecolors='#2b281e',linewidths=.45,zorder=5)
    t((x+w/2)/ratio,81,f'{labels[j]}  {1408+idx:05d}',7.6,'center')
t(220,48,'···',10.2,'center',b=True)
t(7,92,'Filled dots: detected corners carried by optical flow.',7.9,c=muted)
t(7,99.5,'Open circles: capture coordinates on the return image.',7.9,c=muted)
t(311,29,'Release clock',9.3,b=True)
t(311,42,'Calls scheduled at 15 Hz.',8.1)
t(311,56,'Release on first frame with',8.1)
t(311,67,'elapsed time ≥ 46.686 ms.',8.1)
t(311,86,'This call: 49.803 ms',8.4,b=True,c=teal)
divider(106)
header(6,118,'B','Continuous replay')
dash=(0,(3.4,2.1))
# A control arrow links the replay clock to the update instant.
edge([(333,94),(333,132)],teal,1.12,dash)
nodes=[
    (7,59,'Input','Time-stamped\nflight frames',pale),
    (75,85,'Capture match','YOLO11 (frozen)\nMatch boxes to tracks',pale),
    (169,85,'LK propagation','Advance live tracks\n+ a copy of detections',pale),
    (263,88,'Return update','Apply replay rule\nto surviving match',sand),
    (360,78,'Saved tracks','Gate boxes, corners\nand validity flags','#e7eddf')]
for x,w,title,sub,fc in nodes:node(x,132,w,44,title,sub,fc)
for a,b in zip(nodes,nodes[1:]):edge([(a[0]+a[1],154),(b[0],154)])

# The expansion shares the module name. This short neutral leader has no flow semantics.
ln([(300,176),(300,185)],hair,.85)
lightbox(7,185,351,73)
t(13,194,'Return update',9.3,b=True)
t(77,194,'One rule per replay, same detector calls',8.1,it=True,c=muted)
for xx,title,detail in [(14,'Waiting','Keep current\ntracked corners'),(120,'Direct replacement','Write original\ncapture corners'),(240,'Aligned replacement','Write corners propagated\nto the return frame')]:
    t(xx,212,title,8.7,b=True)
    t(xx,230,detail,8.1)
t(13,250,'New or lost track: initialise from propagated corners if at least two are valid.',7.8)
edge([(399,176),(399,266),(58,266),(58,280)],ink,1.25)
edge([(218,266),(218,280)],ink,1.25)
edge([(380,266),(380,280)],ink,1.25)
t(406,220,'Offline\nscoring',7.9,c=muted)

header(7,291,'C','Full flights')
header(128,291,'D','Matched windows')
header(318,291,'E','Reference check')
for x in (120,311):
    ln([(x,285),(x,371)],hair,.7,(0,(3,2.4)))
t(7,308,'Gate failure at 0.05d',8.2,b=True)
t(7,320,'Unmatched output in track-s.',8.05)
for x,st,ha in [(7,'Rule','left'),(78,'Failed (%)','right'),(115,'Track-s','right')]:
    t(x,334,st,7.9,ha,b=True)
ln([(7,341),(115,341)],hair,.6)
for y,label,fail,seconds in [(348,'Waiting','61.60','39.19'),(359,'Direct','62.31','32.18'),(370,'Aligned','47.19','12.92')]:
    t(7,y,label,8.1)
    t(78,y,fail,8.1,'right',b=label=='Aligned')
    t(115,y,seconds,8.1,'right',b=label=='Aligned')
t(128,308,'268 identical gate windows · 250 ms',8.25,b=True)
t(128,320,'Reference-assisted reset or continuous tracking.',8.05)
for x,st in [(128,'Mean outage (ms)'),(213,'Reset'),(265,'Continuous')]:
    t(x,334,st,7.9,b=True)
ln([(128,341),(304,341)],hair,.6)
for y,label,reset,cont in [(348,'Waiting','14.93','104.19'),(359,'Direct','61.74','74.53'),(370,'Aligned','26.70','30.40')]:
    t(128,y,label,8.15)
    t(233,y,reset,8.1,'right',b=label=='Waiting')
    t(302,y,cont,8.1,'right',b=label=='Aligned')
t(318,308,'Same gates, frames and tracks',8.2,b=True)
t(318,322,'Score 11 clips with original labels,',8.05)
t(318,333,'then by masked image review.',8.05)
t(318,347,'Waiting minus aligned (pp)',8.0,it=True)
t(318,359,'Original: 27.34',8.05)
t(318,370,'Image review: 17.79',8.05)
divider(380)
edge([(8,389),(29,389)],ink,1.45)
t(33,389,'Data / predictions',7.75)
edge([(121,389),(142,389)],teal,1.1,dash)
t(146,389,'Timing control',7.75)
t(441,389,'LK = pyramidal Lucas–Kanade optical flow',7.75,'right',c=muted)
t(7,400,'Camera crops: TII Racing, Race Against the Machine (CC BY 4.0).',7.25,c=muted)
for ext in ('svg','pdf','png'):
    fig.savefig(out/('Gate_tracking_framework.'+ext),dpi=360,facecolor='white')
plt.close(fig)
