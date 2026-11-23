"""Produce the primary-endpoint table for the earlier test recordings."""
import csv
import json
from statistics import mean
from revision_additional import flights, out
from continuous import HERE

groups=[]
for flight in flights:
    runtime=json.loads((out/f"runtime_{flight}.json").read_text())[flight]
    rows=[r for r in csv.DictReader((out/f"scores_{flight}.csv").open()) if r["radius"]=="0.05" and r["matching"]=="cyclic"]
    assert len(rows)==3
    d=dict(flight=flight,frames=runtime["frames"],seconds=runtime["flight_seconds"],gate_s=float(rows[0]["gate_seconds"]),calls=runtime["calls"],returns=runtime["returns"],new_inferences=runtime["new_inferences"])
    for r in rows:
        d[r["action"]]=float(r["bad_percent"])
        d[r["action"]+"_unmatched_s"]=float(r["unmatched_prediction_seconds"])
    d["diff"]=d["wait"]-d["aligned"]
    groups.append(d)
pool={k:(mean if k in ["wait","aligned","legacy","diff"] else sum)(g[k] for g in groups) for k in groups[0] if k!="flight"}
pool["flight"]="aggregate"
with (out/"primary.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=groups[0]);w.writeheader();w.writerows(groups+[pool])
lines=["| Flight | Camera s | Gate-s | Waiting % | Aligned % | Direct % | W − A (pp) | Unmatched track-s W/A/D |","|---|---:|---:|---:|---:|---:|---:|---|"]
for r in groups+[pool]:
    lines.append("| "+r["flight"]+" | "+" | ".join(f"{r[k]:.2f}" for k in ["seconds","gate_s","wait","aligned","legacy","diff"])+" | "+" / ".join(f"{r[k+'_unmatched_s']:.2f}" for k in ["wait","aligned","legacy"])+" |")
lines+=["","Development equal-flight means at the identical endpoint: waiting 61.60%, aligned 47.19%, direct 62.31%. The additional recordings represent the same three course types and approximately 120 fps acquisition. Rates and exposure by flight are available in primary.csv. The original dataset's earlier protocol freeze and results identified these recordings as test data. This is retrospective validation on previously evaluated data. Their results were not pooled with development-flight intervals."]
(out/"comparison.md").write_text("\n".join(lines),encoding="utf8")
print(pool)
