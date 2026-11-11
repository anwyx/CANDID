"""Render a GRITS-style trace video from one or two recorded episodes.

Example:
    python scripts/render_trajectory_video.py trajectory.npz --output-dir outputs/scene1_trajectory_video
    python scripts/render_trajectory_video.py refined.npz --second original.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candid.vis.trajectory_video import SCENE1_CAMERA_EYE, SCENE1_CAMERA_TARGET, render_trajectory_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="Evaluation trajectory.npz, or a pair stacked on axis 0")
    parser.add_argument("--second", type=Path, default=None, help="Second episode drawn in the yellow-red trace")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trajectory_video"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    summary = render_trajectory_video(
        args.trajectory,
        args.output_dir,
        args.second,
        eye=SCENE1_CAMERA_EYE,
        target=SCENE1_CAMERA_TARGET,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
