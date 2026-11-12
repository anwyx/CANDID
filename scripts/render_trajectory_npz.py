#!/usr/bin/env python3
"""Offline CANDID trajectory.npz -> MP4 (schematic reconstruction, not RGB camera footage).

Requires numpy, matplotlib, imageio-ffmpeg. No GPU, Vulkan, SAPIEN or physics replay.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def yaw(quat):
    w, x, y, z = np.asarray(quat, dtype=float)
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def get_scene(root, scene_id):
    if root is None:
        return {}
    if not scene_id:
        raise ValueError("--suite needs --scene-id or sibling result.json with scene_id")
    for line in (root / "scenes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["scene_id"] == scene_id:
                return row
    raise ValueError(f"Scene {scene_id!r} not found in {root}/scenes.jsonl")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--suite", type=Path, help="Optional eval/combined250 for precise primitive sizes and colors")
    p.add_argument("--scene-id", help="Optional; automatically read from sibling result.json")
    p.add_argument("--output", type=Path, help="Default: trajectory.mp4 beside the npz")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--stride", type=int, default=2, help="Show every Nth recorded frame")
    args = p.parse_args()
    if args.fps < 1 or args.stride < 1:
        p.error("--fps and --stride must be positive")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.animation import FFMpegWriter
    import imageio_ffmpeg

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    with np.load(args.trajectory, allow_pickle=False) as z:
        keys = ("tcp_pose", "target_pose", "goal_pos", "distractor_poses")
        if any(k not in z.files for k in keys):
            raise ValueError(f"Missing required fields {keys}; found {z.files}")
        tcp, target, goal, obs = (np.asarray(z[k], dtype=float) for k in keys)
        sizes = np.asarray(z["object_sizes_m"]) if "object_sizes_m" in z.files else None
        toppled = np.asarray(z["distractor_toppled"]) if "distractor_toppled" in z.files else None
    t, n = len(tcp), obs.shape[1]
    if t < 1 or tcp.shape != (t, 7) or target.shape != (t, 7) \
       or goal.shape != (t, 3) or obs.shape != (t, n, 7):
        raise ValueError("Expected tcp/target [T,7], goal [T,3], obstacles [T,N,7]")
    if not all(np.isfinite(v).all() for v in (tcp, target, goal, obs)):
        raise ValueError("Trajectory contains NaN/Inf")

    result_file = args.trajectory.parent / "result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    scene_id = args.scene_id or result.get("scene_id")
    scene = get_scene(args.suite, scene_id)
    spec = scene.get("scene_spec") or {}
    specs = [spec.get("target") or {}] + (spec.get("distractors") or [{} for _ in range(n)])
    if len(specs) != n + 1:
        raise ValueError("Trajectory and scene_spec have different distractor counts")

    def dims(i):
        size = specs[i].get("size")
        if size is not None:
            return float(size[0]), float(size[1])
        s = float(sizes[i - 1]) if i and sizes is not None else .04
        return s, s

    def color(i):
        c = specs[i].get("color")
        if c is not None:
            return tuple(float(np.clip(v, 0, 1)) for v in c[:3])
        palette = ("#E15252", "#E9A13B", "#71B5D8", "#79B998",
                   "#A8A1D3", "#D996A0", "#C7B07C")
        return palette[i % len(palette)]

    xy = np.concatenate((tcp[:, :2], target[:, :2], goal[:, :2],
                         obs[:, :, :2].reshape(-1, 2)), axis=0)
    lo, hi = xy.min(0), xy.max(0)
    center = (lo + hi) / 2
    span = np.maximum(hi - lo, [0.32, 0.25])
    pad = np.maximum(span * .13, [.04, .04])

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#F8FAFD")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.58, 1], height_ratios=[1, .40],
                          left=.06, right=.96, top=.89, bottom=.10, hspace=.29, wspace=.20)
    ax = fig.add_subplot(gs[:, 0])
    hz = fig.add_subplot(gs[0, 1])
    status = fig.add_subplot(gs[1, 1])
    status.axis("off")
    for a in (ax, hz):
        a.set_facecolor("white")
        a.grid(alpha=.18)
        a.set_axisbelow(True)
        for spine in a.spines.values():
            spine.set_color("#CAD2DE")
    fig.suptitle("CANDID  |  Recorded Trajectory Reconstruction",
                 x=.06, y=.965, ha="left", fontsize=19, fontweight="bold", color="#24344A")
    fig.text(.96, .948, scene_id or args.trajectory.parent.name,
             ha="right", fontsize=11, color="#50637A")

    ax.set_xlim(center[0] - span[0]/2 - pad[0], center[0] + span[0]/2 + pad[0])
    ax.set_ylim(center[1] - span[1]/2 - pad[1], center[1] + span[1]/2 + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top-down workspace (recorded poses)", loc="left",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    radius = float(spec.get("goal_marker_radius", .06))
    ax.add_patch(Circle(goal[0, :2], radius, fill=False,
                        edgecolor="#20A07A", linestyle="--", linewidth=2.5))
    ax.text(goal[0, 0], goal[0, 1] + radius + .008, "GOAL",
            ha="center", color="#16876A", weight="bold", fontsize=9)

    patches = []
    for i in range(n + 1):
        pose = target[0] if i == 0 else obs[0, i - 1]
        width, height = dims(i)
        if specs[i].get("kind") in ("sphere", "cylinder", "ycb"):
            patch = Circle(pose[:2], max(width, height)/2, facecolor=color(i),
                           edgecolor="#40516B", linewidth=1.2, alpha=.95)
        else:
            patch = Rectangle((pose[0] - width/2, pose[1] - height/2), width, height,
                              angle=yaw(pose[3:]), rotation_point="center",
                              facecolor=color(i), edgecolor="#40516B",
                              linewidth=1.2, alpha=.95)
        ax.add_patch(patch)
        patches.append(patch)

    (tcp_line,) = ax.plot([], [], color="#3368C1", lw=1.7, alpha=.7, label="TCP path")
    (tcp_point,) = ax.plot([], [], "X", color="#2259C2", ms=10)
    (target_line,) = ax.plot([], [], color="#E15252", lw=1.5, alpha=.7, label="Target path")
    ax.legend(loc="upper left", fontsize=9)
    steps = np.arange(t)
    hz.plot(steps, tcp[:, 2], color="#2259C2", lw=2, label="TCP Z")
    hz.plot(steps, target[:, 2], color="#E15252", lw=2, label="Target Z")
    zmin = min(tcp[:, 2].min(), target[:, 2].min())
    zmax = max(tcp[:, 2].max(), target[:, 2].max())
    zpad = max(.025, .12 * (zmax-zmin))
    hz.set_xlim(0, max(1, t - 1))
    hz.set_ylim(zmin-zpad, zmax+zpad)
    hz.set_xlabel("Recorded step")
    hz.set_ylabel("Height (m)")
    hz.set_title("Height during pick and place", loc="left",
                 fontsize=13, fontweight="bold")
    hz.legend(loc="upper right", fontsize=9)
    (cursor,) = hz.plot([0, 0], [zmin-zpad, zmax+zpad], color="#56637A", ls="--", lw=1.2)

    progress = status.text(0, .95, "", fontsize=14, fontweight="bold",
                           color="#24344A", va="top")
    status.text(0, .57, "Recorded poses  |  No physics re-simulation",
                fontsize=10, color="#50637A")
    status.text(0, .38, "Only TCP shown; no full robot mesh in this reconstruction.",
                fontsize=9, color="#697A90")
    if "task_success" in result:
        label = f"Final success: {bool(result['task_success'])}  |  Topple: {result.get('topple_count','?')}  |  Fall: {result.get('fall_count','?')}"
    else:
        label = "Schematic motion video, not simulator RGB footage."
    status.text(0, .17, label, fontsize=10, color="#455A72")

    output = args.output or args.trajectory.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = list(range(0, t, args.stride))
    if frames[-1] != t-1:
        frames.append(t-1)
    writer = FFMpegWriter(fps=args.fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    print(f"Rendering {len(frames)} frames to {output}", flush=True)
    with writer.saving(fig, str(output), 100):
        for k in frames:
            poses = np.concatenate((target[k:k+1], obs[k]), axis=0)
            for i, patch in enumerate(patches):
                pos = poses[i]
                width, height = dims(i)
                if isinstance(patch, Circle):
                    patch.center = pos[:2]
                else:
                    patch.set_xy((pos[0] - width/2, pos[1] - height/2))
                    patch.angle = yaw(pos[3:])
                is_toppled = (i > 0 and toppled is not None
                              and toppled.shape[:2] == (t, n) and toppled[k, i-1])
                patch.set_edgecolor("#D7263D" if is_toppled else "#40516B")
                patch.set_linewidth(2.3 if is_toppled else 1.2)
            tcp_line.set_data(tcp[:k+1,0], tcp[:k+1,1])
            tcp_point.set_data([tcp[k,0]], [tcp[k,1]])
            target_line.set_data(target[:k+1,0], target[:k+1,1])
            cursor.set_data([k,k], [zmin-zpad, zmax+zpad])
            progress.set_text(f"Step {k:03d} / {t-1:03d}   |   TCP z = {tcp[k,2]:.3f} m")
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
