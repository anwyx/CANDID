"""Evaluate the trained predictor without running the simulator."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model.predictor.evaluate import main


if __name__ == "__main__":
    main()
