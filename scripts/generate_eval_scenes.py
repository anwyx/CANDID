"""Generate a reproducible random evaluation suite from a registered CANDID env."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from candid.eval.generate_suite import main


if __name__ == "__main__":
    main()
