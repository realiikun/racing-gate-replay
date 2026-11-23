import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RAM_AUTONOMOUS", HERE / "ram"))
