"""Display the six manuscript tables from the released flight and anchor records."""
import csv
from pathlib import Path
from statistics import mean
from collections import defaultdict

root=Path(__file__).resolve().parent
def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))
def full(source):
    scores=defaultdict(dict)
    for r in source:
        if r["matching"]=="cyclic":
            scores[float(r["radius"]),r["flight"]][r["action"]]=r
    return scores
def record_table(scores,radius):
    items=[(f,d) for (k,f),d in sorted(scores.items()) if k==radius]
    totals=[]
    for flight,data in items:
        w,a,d=[float(data[k]["bad_percent"]) for k in ("wait","aligned","legacy")]
        vals=[float(data["wait"]["gate_seconds"]),w,a,d,w-a]+[float(data[k]["unmatched_prediction_seconds"]) for k in ("wait","aligned","legacy")]
        totals.append(vals)
        print(flight,*(f"{v:.2f}" for v in vals))
    summary=[(sum if j in (0,5,6,7) else mean)(col) for j,col in enumerate(zip(*totals))]
    print("All flights",*(f"{v:.2f}" for v in summary))
    return summary
def windows(path, common):
    groups=defaultdict(list)
    for r in rows(path):
        if common and (float(r["radius"])!=.05 or int(r["duration_ms"])!=0):
            continue
        groups[r["flight"]].append(r)
    assert sum(map(len,groups.values()))==268
    values={f:[mean(float(r[k]) for r in rr) for k in ("wait","aligned","legacy")] for f,rr in groups.items()}
    return values
if __name__=="__main__":
    dev=full(rows(root/"continuous_scores.csv"))
    print("I: flight, gate-seconds, failure % (waiting, aligned, direct), W-A pp, unmatched track-seconds (same order)")
    ref=record_table(dev,.05)
    assert round(ref[4],2)==14.41
    print("II: reference, gate-seconds, waiting %, aligned %, W-A pp")
    print("Full recordings",*(f"{ref[i]:.2f}" for i in (0,1,2,4)))
    clip=rows(root/"blind_clips/comparison.csv")
    for variant,expected in [("paired_txt",27.34),("paired_blind",17.79)]:
        trial=[r for r in clip if r["variant"]==variant and r["flight"]!="flight_equal"]
        values=[sum(float(r["gate_seconds"]) for r in trial)]+[mean(float(r[k]) for r in trial) for k in ("wait","aligned","advantage_pp")]
        assert round(values[-1],2)==expected
        print(variant,*(f"{v:.2f}" for v in values))
    print("III and IV: protocol, flight, mean outage ms (waiting, aligned, direct)")
    for protocol,name in [("common","anchor_outages_ms.csv"),("continuous","continuous_same_windows_ms.csv")]:
        data=windows(root/name,protocol=="common")
        for f,v in data.items():print(protocol,f,*(f"{x:.2f}" for x in v))
        print(protocol,"equal flight",*(f"{mean(v):.2f}" for v in zip(*data.values())))
    print("V: tolerance sensitivity, columns as in I")
    for radius in sorted({k for k,f in dev}):
        print(radius);record_table(dev,radius)
    print("VI: additional test flights, columns as in I")
    held=full([r for f in sorted((root/"results/held_out").glob("scores_*.csv")) for r in rows(f)])
    assert round(record_table(held,.05)[4],2)==8.38
