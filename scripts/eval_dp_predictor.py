"""Entry point for separate DP + Predictor closed-loop evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from model.integration.evaluate import main


if __name__ == "__main__":
    main()
