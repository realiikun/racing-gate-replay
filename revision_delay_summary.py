"""Aggregate delay runs and plot descriptive flight-cluster intervals."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent/"results/delay_sweep"
draws=np.random.default_rng(20261123).integers(0,3,(10000,3))
delays=[25000,46686,50000,75000,100000,150000]
actions=["wait","aligned","legacy"]
lines=["| D (ms) | Waiting (%) | Aligned (%) | Direct (%) | Waiting − aligned (pp, 95% CI) | Unmatched track-s (W / A / D) |","|---:|---:|---:|---:|---|---|"]
allrows=[]
for delay in delays:
    flights=[json.loads(f.read_text()) for f in sorted(out.glob(f"flight-*_{delay}.json"))]
    assert len(flights)==3
    values=np.array([[r["actions"][a]["bad_percent"] for a in actions] for r in flights])
    unmatched=np.array([[r["actions"][a]["unmatched_s"] for a in actions] for r in flights]).sum(axis=0)
    diff=values[:,0]-values[:,1]
    lo,hi=np.percentile(diff[draws].mean(axis=1),[2.5,97.5])
    avg=values.mean(axis=0)
    intervals=np.percentile(values[draws].mean(axis=1),[2.5,97.5],axis=0)
    item=dict(delay_ms=delay/1000,flights=flights,order=actions,mean=avg.tolist(),intervals=intervals.tolist(),diff=[diff.mean(),lo,hi],unmatched=unmatched.tolist())
    (out/f"delay_{delay}.json").write_text(json.dumps(item,indent=2))
    allrows.append(item)
    lines.append(f"| {delay/1000:g} | "+" | ".join(f"{v:.2f}" for v in avg)+f" | {diff.mean():.2f} [{lo:.2f}, {hi:.2f}] | "+" / ".join(f"{v:.2f}" for v in unmatched)+" |")
lines+=["","Each call is associated at capture and released at the first available frame after its configured delay. The capture schedule is fixed. Overlapping calls are retained in FIFO order, with aligned initialization when a capture-time target no longer exists. This assumes concurrent service capacity. Different outstanding calls can initialize separate tracks before earlier calls have returned. Both bad gate-time and unmatched track time matter for interpreting this regime. Gaps reset all pending calls. Two final-frame cache misses cannot return at any positive delay. Scores reproduce the 46.686 ms reference to 1e-7 percentage points. The 50 ms publication frame can differ because the first frame at or after 46.686 ms can precede 50 ms. Intervals resample the three flights as clusters (10,000 draws, seed 20261123)."]
(out/"summary.md").write_text("\n".join(lines),encoding="utf8")
fig,(ax,bx)=plt.subplots(2,1,figsize=(6.5,5.4),sharex=True,layout="constrained")
for j,(label,marker,col) in enumerate(zip(["Waiting","Aligned replacement","Direct replacement"],["o","s","^"],["#252525","#007474","#ad5119"])):
    rr=[r for r in allrows if r["delay_ms"]!=46.686]
    xx=np.array([r["delay_ms"] for r in rr]); yy=np.array([r["mean"][j] for r in rr])
    ci=np.array([r["intervals"] for r in rr])[:,:,j]
    ax.errorbar(xx+(j-1)*1.2,yy,yerr=[yy-ci[:,0],ci[:,1]-yy],label=label,marker=marker,color=col,capsize=3)
    bx.plot(xx,[r["unmatched"][j] for r in rr],marker=marker,color=col)
ax.set_ylabel("Failed gate-time (%)");ax.legend(frameon=False,fontsize=9,ncol=3)
bx.set_ylabel("Unmatched track time (s)");bx.set_xlabel("Configured service delay (ms)");bx.set_xticks(xx)
ax.set_ylim(0,100);bx.set_ylim(bottom=0)
for a in (ax,bx): a.spines[["top","right"]].set_visible(False)
fig.savefig(out/"delay_sweep.svg")
fig.savefig(out/"delay_sweep.pdf")
print("Wrote six delay summaries and two vector figures.")

