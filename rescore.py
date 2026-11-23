"""Regenerate full-flight scores from saved trajectories and archived RAM annotations."""
import csv
import json
import tempfile
import zipfile
from pathlib import Path
from math import isclose
import score_continuous
from continuous import HERE, FLIGHTS
from revision_additional import flights as additional

def main():
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(HERE / "inputs/ram_labels.zip") as archive:
            archive.extractall(tmp)
        score_continuous.DATA = Path(tmp)
        for flight in (*FLIGHTS,*additional):
            folder = HERE if flight in FLIGHTS else HERE / "results/held_out"
            name = "continuous_scores.csv" if flight in FLIGHTS else f"scores_{flight}.csv"
            runtime = "continuous_runtime.json" if flight in FLIGHTS else f"runtime_{flight}.json"
            with (folder / name).open() as stream:
                expected = {(r["action"],r["matching"],float(r["radius"])):r for r in csv.DictReader(stream) if r["flight"]==flight}
            actual = score_continuous.score(flight,prediction_dir=folder,runtime_file=folder/runtime)
            assert len(actual)==len(expected)==24
            for row in actual:
                reference=expected[row["action"],row["matching"],row["radius"]]
                for field in ("gate_seconds","bad_gate_seconds","bad_percent","unmatched_prediction_seconds"):
                    assert isclose(row[field],float(reference[field]),rel_tol=0,abs_tol=1e-7),(flight,field)
            print(f"{flight}: all 24 scores reproduced",flush=True)
if __name__=="__main__":
    main()
