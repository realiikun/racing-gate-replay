import gzip
import json
import numpy as np
from config import HERE

def frozen(flight):
    with gzip.open(HERE / "detections" / f"{flight}.jsonl.gz", "rt") as stream:
        return {r["stem"]: tuple(np.asarray(r[key],np.float32).reshape(shape) for key,shape in
                [("boxes",(-1,4)),("points",(-1,4,2)),("confidence",(-1,4))]) for r in map(json.loads,stream)}
