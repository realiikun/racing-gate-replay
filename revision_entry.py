"""Classify each rule at the anchor frame before scoring the following window."""
import csv
import gzip
import json
from collections import defaultdict
from statistics import mean
from score_continuous import matched_radii
from sensitivity import m, original, HERE

out = HERE / "results"
anchors = json.loads((original.DATA/"anchors_checkpoint.json").read_text(encoding="utf8"))["rows"]
windows = list(csv.DictReader((HERE/"continuous_same_windows_ms.csv").open()))
rows = []
for flight in sorted({r["flight"] for r in windows}):
    rec = m.Record(flight)
    rec.build_tracks()
    lookup = {(r["tid"],r["anchor_index"]):r for r in anchors[flight]}
    with gzip.open(HERE/f"predictions_{flight}.jsonl.gz","rt") as f:
        predictions = {r["stem"]:r for r in map(json.loads,f)}
    for r in [w for w in windows if w["flight"]==flight]:
        tid = int(r["tid"])
        fi = lookup[tid,int(r["anchor"])]["fi"]
        stem = rec.frames[fi]["stem"]
        gates = [(t,g) for t,g in rec.track_rows[fi] if m.gt_diagonal(g)>1e-6]
        pos = [t for t,g in gates].index(tid)
        for action in ["wait","aligned"]:
            rad = matched_radii([g for t,g in gates],predictions[stem][action])[0][pos,0]
            rows.append(dict(flight=flight,tid=tid,anchor=r["anchor"],frame=stem,rule=action,critical_radius=rad,state="good" if rad<=.05 else "degraded",outage_ms=float(r[action])))
assert len(rows)==536
with (out/"entry_state_stratification.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
lines=["| Rule | Entry state | Windows | Mean outage (ms) |","|---|---|---:|---:|"]
for action in ["wait","aligned"]:
    for state in ["good","degraded"]:
        vals=[r["outage_ms"] for r in rows if r["rule"]==action and r["state"]==state]
        lines.append(f"| {action} | {state} | {len(vals)} | {mean(vals):.2f} |")
lines+=["","Rule-specific strata include unmatched tracks as degraded. Window-weighted descriptive means. Matched-state contrasts below restrict both rules to exactly the same windows.","","| Waiting entry | Aligned entry | Windows | Mean waiting minus aligned (ms) |","|---|---|---:|---:|"]
joint=defaultdict(list)
for a,b in zip(rows[::2],rows[1::2]):
    joint[a["state"],b["state"]].append(a["outage_ms"]-b["outage_ms"])
for (a,b),vals in sorted(joint.items()):
    lines.append(f"| {a} | {b} | {len(vals)} | {mean(vals):.2f} |")
(out/"entry_state_stratification.md").write_text("\n".join(lines),encoding="utf8")
print("\n".join(lines))
