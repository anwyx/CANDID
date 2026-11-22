"""Run the fixed-state 4/8-step local DR paired experiment."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from candid.eval.local_dr import main


if __name__ == "__main__":
    main()
