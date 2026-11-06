"""Generate settled, replayable CANDID scenes without executing the robot."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from candid.runner.collect import main


if __name__ == "__main__":
    main(["--scene-only", *sys.argv[1:]])
