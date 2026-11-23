"""Paired flight-cluster estimates from the frozen development replay."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

def markdown(table):
    frame = table.reset_index()
    def entry(v):
        return f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    header = "| " + " | ".join(str(k) for k in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    return "\n".join([header, rule] + ["| " + " | ".join(entry(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
actions = ["wait", "aligned", "legacy"]
scores = pd.read_csv(ROOT / "continuous_scores.csv").query("matching == 'cyclic'")
windows = pd.read_csv(ROOT / "anchor_outages_ms.csv").query("radius == 0.05 and duration_ms == 0")
continuous = pd.read_csv(ROOT / "continuous_same_windows_ms.csv")
keys = ["flight", "tid", "anchor"]
assert len(windows.merge(continuous, on=keys, validate="one_to_one")) == 268
old = json.loads((ROOT / "inputs/alignment_clean_development.json").read_text())["rows"]
ref = pd.DataFrame(old).rename(columns={"rec":"flight","anchor_index":"anchor"})
pairs = windows.merge(ref, on=keys, validate="one_to_one")
assert len(pairs) == 268
error = max(abs(pairs[a]/1000-pairs[b]).max() for a,b in zip(actions,["wait_s","aligned_47ms_s","legacy_47ms_s"]))
assert error < 1e-7
table = scores.pivot(index=["radius","action"],columns="flight",values="bad_percent")
table["equal_flight"] = table.mean(axis=1)
table.to_csv(OUT/"tolerance_absolute_table.csv")
short = table.rename(columns=lambda c:c.split("-")[1] if "-" in c else "Mean").rename(index={"legacy":"Direct","wait":"Waiting","aligned":"Aligned"})
(OUT/"tolerance_absolute_table.tex").write_text(r"\begin{tabular}{llrrrr}"+"\n"+ "Radius & Rule & 03a & 08a & 16a & Mean "+chr(92)*2+"\n"+"\n".join(" & ".join([str(i),str(a)]+[f"{v:.2f}" for v in row])+chr(92)*2 for (i,a),row in short.iterrows())+"\n"+r"\end{tabular}",encoding="utf8")
(OUT/"tolerance_absolute_table.md").write_text(markdown(short.reset_index(level="radius").assign(radius=lambda x:x.radius.map(lambda v:f"{v:.3f}")))+"\n\nFailed gate-time (%). The ranking changes between 0.025d and 0.05d. The crossing is unresolved.",encoding="utf8")
draws = np.random.default_rng(20261123).integers(0,3,(10000,3))
results = {}
def interval(name, frame):
    diff = (frame.wait-frame.aligned).to_numpy()
    lo,hi = np.percentile(diff[draws].mean(axis=1),[2.5,97.5])
    results[name] = dict(mean=diff.mean(),lo=lo,hi=hi,per_flight=frame.to_dict("index"))
for radius in sorted(scores.radius.unique()):
    interval(str(radius),scores[scores.radius==radius].pivot(index="flight",columns="action",values="bad_percent"))
for name,frame in [("common",windows),("continuous",continuous)]:
    interval(name,frame.groupby("flight")[actions].mean())
method="Waiting minus aligned. Paired within windows and averaged within flight. Three flights weighted equally and resampled as whole clusters with replacement. Percentile 95% intervals, 10,000 draws, seed 20261123. Only three distinct flights contribute. Interpret the discrete intervals descriptively. The 16 radius-by-duration settings are exploratory, with no family-wise inference."
(OUT/"bootstrap_ci.json").write_text(json.dumps(dict(method=method,maximum_error_s=error,results=results),indent=2),encoding="utf8")
summary=markdown(pd.DataFrame(results).T.drop(columns="per_flight").astype(float).rename_axis("contrast"))
(OUT/"bootstrap_ci.md").write_text(method+"\n\n"+summary+"\n\nMean outage (ms) by flight and protocol\n\n"+markdown(pd.concat({n:pd.DataFrame(results[n]["per_flight"]).T for n in ["common","continuous"]},names=["protocol","flight"])),encoding="utf8")
print(summary)


