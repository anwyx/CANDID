#!/usr/bin/env python3
"""Real ManiSkill RGB frames with Vanilla DP / CANDID recorded TCP overlays.

Input: two episode_replay.zip bundles, or two trajectory.npz files and the
frozen eval suite. No policy inference or physics resimulation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# The official DR implementation is pure NumPy; import it without ManiSkill.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from candid.data.metrics import disturbance_from_poses

VANILLA = ((255, 244, 82), (246, 181, 49), (239, 105, 42), (210, 42, 44))
GUIDED = ((105, 243, 252), (64, 196, 253), (42, 112, 245), (32, 38, 213))
KEYS = ("tcp_pose", "robot_qpos", "target_pose", "distractor_poses", "goal_pos")


def load_episode(path):
    """Read the original archived physics trace, including exact DR metadata."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        needed = {"trajectory.npz", "initial_state.npz", "scene.json", "result.json"}
        if not needed <= names:
            raise ValueError(f"Bundle {path} missing {sorted(needed - names)}")
        scene = json.loads(z.read("scene.json"))
        result = json.loads(z.read("result.json"))
        digest = hashlib.sha256(z.read("initial_state.npz")).hexdigest()
        with np.load(io.BytesIO(z.read("trajectory.npz")), allow_pickle=False) as a:
            physics = ("object_sizes_m", "distractor_toppled", "distractor_fallen")
            required = set(KEYS) | set(physics)
            if not required <= set(a.files):
                raise ValueError(
                    f"Trajectory {path} missing required physics fields "
                    f"{sorted(required - set(a.files))}; repack the original eval NPZ"
                )
            track = {k: a[k].copy() for k in required}
    if scene["scene_id"] != result["scene_id"]:
        raise ValueError("Bundle scene_id mismatch")
    t = len(track["tcp_pose"])
    if t < 2 or track["tcp_pose"].shape != (t, 7) or track["target_pose"].shape != (t, 7):
        raise ValueError("Invalid TCP or target shape")
    if track["robot_qpos"].ndim != 2 or track["robot_qpos"].shape[0] != t:
        raise ValueError("Invalid robot_qpos shape")
    poses = track["distractor_poses"]
    if poses.ndim != 3 or poses.shape[0] != t or poses.shape[-1] != 7:
        raise ValueError("Invalid distractor_poses: expected [T,N,7]")
    n = poses.shape[1]
    if track["goal_pos"].shape != (t, 3):
        raise ValueError("Invalid goal_pos shape")
    if track["object_sizes_m"].shape != (n,) or np.any(track["object_sizes_m"] <= 0):
        raise ValueError("Invalid object_sizes_m; need positive characteristic size for each distractor")
    for name in ("distractor_toppled", "distractor_fallen"):
        if track[name].shape != (t, n) or track[name].dtype.kind != "b":
            raise ValueError(f"{name}: expected bool [T,N]")
    if any(not np.isfinite(track[k]).all() for k in (*KEYS, "object_sizes_m")):
        raise ValueError("Non-finite recorded poses or sizes")
    return {"scene": scene, "result": result, "track": track, "sha": digest}

def check_pair(v, g):
    a, b = v["scene"], g["scene"]
    for key in ("scene_id", "env_id", "env_module", "env_kwargs", "scene_spec", "seed"):
        if a.get(key) != b.get(key):
            raise ValueError(f"Different frozen initial scene: {key}")
    for key in ("eval_seed", "original_env_id"):
        if v["result"].get(key) != g["result"].get(key):
            raise ValueError(f"Different eval protocol: {key}")
    if v["sha"] != g["sha"]:
        raise ValueError("Initial state NPZ hashes differ")
    for key in KEYS:
        x, y = v["track"][key][0], g["track"][key][0]
        if x.shape != y.shape:
            raise ValueError(f"First-frame {key} shapes differ")
        y = y.copy()
        if key in ("tcp_pose", "target_pose", "distractor_poses"):
            flip = (x[..., 3:] * y[..., 3:]).sum(axis=-1, keepdims=True) < 0
            y[..., 3:] = np.where(flip, -y[..., 3:], y[..., 3:])
        if np.max(np.abs(x - y)) > 2e-4:
            raise ValueError(f"First-frame {key} differs; unsafe to overlay")


class Camera:
    def __init__(self, spec, width, height, fovy=1.0):
        eye = np.asarray(spec["human_camera_eye"], float)
        look = np.asarray(spec["human_camera_look_at"], float)
        forward = look - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0., 0., 1.))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.eye = eye
        self.axes = np.stack((right, up, forward))
        self.focal = (height / 2) / np.tan(fovy / 2)
        self.width, self.height = width, height

    def project(self, points):
        points = np.asarray(points).reshape(-1, 3)
        p = (points - self.eye) @ self.axes.T
        good = p[:, 2] > .02
        xy = np.full((len(points), 2), np.nan)
        xy[good, 0] = self.width / 2 + self.focal * p[good, 0] / p[good, 2]
        xy[good, 1] = self.height / 2 - self.focal * p[good, 1] / p[good, 2]
        return xy, good


def gradient(colors, p):
    x = float(np.clip(p, 0, 1)) * (len(colors) - 1)
    i = min(int(x), len(colors) - 2)
    return tuple(int(c) for c in ((1 - (x - i)) * np.asarray(colors[i]) + (x - i) * np.asarray(colors[i + 1])))


def draw_trace(frame, xyz, step, colors, camera):
    count = min(step + 1, len(xyz))
    if not count:
        return
    points, visible = camera.project(xyz[:count])
    indices = sorted(set(list(range(0, count, max(1, math.ceil(count / 90)))) + [count - 1]))
    for i, j in zip(indices[:-1], indices[1:]):
        if visible[i] and visible[j]:
            cv2.line(frame, tuple(np.rint(points[i]).astype(int)),
                     tuple(np.rint(points[j]).astype(int)),
                     gradient(colors, j / max(1, len(xyz) - 1)), 4, cv2.LINE_AA)
    markers = indices[::max(1, len(indices)//20)]
    if count - 1 not in markers:
        markers.append(count - 1)
    for i in markers:
        if visible[i]:
            pos = tuple(np.rint(points[i]).astype(int))
            radius = 6 if i == count - 1 else 4
            cv2.circle(frame, pos, radius, gradient(colors, i / max(1, len(xyz) - 1)), -1, cv2.LINE_AA)
            cv2.circle(frame, pos, radius, (255, 255, 255), 1, cv2.LINE_AA)


def overlay(frame, vanilla, guided, step, camera):
    if tuple(frame.shape[:2]) != (camera.height, camera.width):
        raise ValueError("RGB frame resolution and camera projection disagree")
    out = np.ascontiguousarray(frame.copy())
    # The input is RGB, and OpenCV writes raw channel tuples as provided.
    draw_trace(out, vanilla["track"]["tcp_pose"][:, :3], step, VANILLA, camera)
    draw_trace(out, guided["track"]["tcp_pose"][:, :3], step, GUIDED, camera)
    return out


def font(size, bold=False):
    for name in (("DejaVuSans-Bold.ttf", "Arial.ttf") if bold else ("DejaVuSans.ttf", "Arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def metrics(item):
    r = item["result"]
    status = "SUCCESS" if r["task_success"] else "FAIL"
    return (f"{status} | DR mean {float(r['dr_mean']):.4f} | "
            f"Topple {int(r['topple_count'])} | Fall {int(r['fall_count'])}")


def physical_step_series(item):
    """Cumulative physical DR at EVERY recorded step, using CANDID's exact metric.

    Reference is the same settled first pose throughout, with peak translation
    normalized by the recorded object's characteristic length, peak rotation,
    and official rising-edge topple/fall counts. Not predictor estimates.
    """
    track = item["track"]
    poses = track["distractor_poses"]
    sizes = track["object_sizes_m"]
    toppled = track["distractor_toppled"]
    fallen = track["distractor_fallen"]
    series = []
    for t in range(len(poses)):
        dr = disturbance_from_poses(
            poses, sizes, start=0, stop=t,
            topple_flags=toppled, fall_flags=fallen,
        )
        series.append({
            "dr_mean": float(dr["dr_mean"]),
            "dr_max": float(dr["dr_max"]),
            "topple": int(dr["topple_count"]),
            "fall": int(dr["fall_count"]),
        })
    last = series[-1]
    for label, key in (("dr_mean", "dr_mean"), ("dr_max", "dr_max"),
                       ("topple", "topple_count"), ("fall", "fall_count")):
        target = float(item["result"][key])
        tolerance = 0.0005 if label.startswith("dr_") else 0.0
        if abs(last[label] - target) > tolerance:
            raise ValueError(
                f"Recorded trajectory and original evaluation disagree: {label} "
                f"reconstructed={last[label]:.6f}, saved={target:.6f}. "
                "Do not present inconsistent DR numbers."
            )
    return series


def make_step_rows(v, g):
    """Align real physics steps; never fabricate positions past an episode's end.

    The RGB background is a Guided replay, hence there is one row per Guided
    frame. If Vanilla terminates earlier, its later metrics are undefined
    (null in JSON, blank in CSV). Only common recorded steps can be selected.
    """
    a = physical_step_series(v)
    b = physical_step_series(g)
    overlap = min(len(a), len(b))
    rows = []
    for step, guided in enumerate(b):
        vanilla = a[step] if step < len(a) else None
        row = {
            "step": step,
            "both_recorded": vanilla is not None,
            "vanilla_dr_mean": vanilla["dr_mean"] if vanilla else None,
            "candid_dr_mean": guided["dr_mean"],
            "delta_dr_mean": vanilla["dr_mean"] - guided["dr_mean"] if vanilla else None,
            "vanilla_dr_max": vanilla["dr_max"] if vanilla else None,
            "candid_dr_max": guided["dr_max"],
            "delta_dr_max": vanilla["dr_max"] - guided["dr_max"] if vanilla else None,
            "vanilla_topple": vanilla["topple"] if vanilla else None,
            "candid_topple": guided["topple"],
            "delta_topple": vanilla["topple"] - guided["topple"] if vanilla else None,
            "vanilla_fall": vanilla["fall"] if vanilla else None,
            "candid_fall": guided["fall"],
            "delta_fall": vanilla["fall"] - guided["fall"] if vanilla else None,
            "tcp_gap_m": (
                float(np.linalg.norm(v["track"]["tcp_pose"][step, :3]
                                     - g["track"]["tcp_pose"][step, :3]))
                if vanilla else None
            ),
        }
        rows.append(row)
    return rows, overlap


def _unit_scale(values):
    a = np.asarray(values, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def select_auto_steps(rows, overlap, count, min_gap, smooth_window):
    if overlap < count:
        raise ValueError(
            f"Only {overlap} jointly recorded steps; cannot select {count} different frames"
        )
    common = rows[:overlap]
    tcp = _unit_scale([r["tcp_gap_m"] for r in common])
    dr_mean_gain = _unit_scale([max(r["delta_dr_mean"], 0.0) for r in common])
    dr_max_gain = _unit_scale([max(r["delta_dr_max"], 0.0) for r in common])
    topple_gain = _unit_scale([max(r["delta_topple"], 0) for r in common])
    fall_gain = _unit_scale([max(r["delta_fall"], 0) for r in common])
    # Visual difference + actual physical benefit. No predictor/guidance estimates.
    score = .40*tcp + .30*dr_mean_gain + .15*dr_max_gain + .10*topple_gain + .05*fall_gain
    if smooth_window > 1:
        left = smooth_window // 2
        right = smooth_window - 1 - left
        padded = np.pad(score, (left, right), mode="edge")
        score = np.convolve(padded, np.ones(smooth_window)/smooth_window, mode="valid")
    for idx, r in enumerate(rows):
        r["selection_score"] = float(score[idx]) if idx < overlap else None
        r["is_selected"] = False

    effective_gap = min(min_gap, (overlap - 1) // max(1, count - 1))
    # Local maxima first, then strongest remaining steps with separation.
    local = [
        t for t in range(overlap)
        if score[t] >= (score[t-1] if t else -np.inf)
        and score[t] >= (score[t+1] if t+1 < overlap else -np.inf)
    ]
    ordered = sorted(local, key=lambda t: (-score[t], -t))
    ordered += sorted((i for i in range(overlap) if i not in local),
                      key=lambda t: (-score[t], -t))
    selected = []
    # A greedy peak selection can occasionally block all remaining candidates;
    # gracefully relax the separation instead of failing on a valid episode.
    for gap in range(effective_gap, 0, -1):
        selected = []
        for t in ordered:
            if all(abs(t-u) >= gap for u in selected):
                selected.append(t)
            if len(selected) == count:
                break
        if len(selected) == count:
            effective_gap = gap
            break
    if len(selected) < count:
        raise RuntimeError("Could not select comparison frames")
    selected.sort()
    for t in selected:
        rows[t]["is_selected"] = True
    return selected, effective_gap


def _fmt(value, digits=4):
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def save_separate_frame(image, step, row, output):
    """One clean, presentation-ready PNG; metrics are outside the RGB scene."""
    im = Image.fromarray(image)
    top, footer = 56, 124
    canvas = Image.new("RGB", (im.width, im.height + top + footer), "#F8FAFC")
    canvas.paste(im, (0, top))
    d = ImageDraw.Draw(canvas)
    d.text((24, 12), f"STEP {step}   |   Recorded physical disturbance",
           font=font(23, True), fill="#263449")
    d.line((0, top+im.height, im.width, top+im.height), fill="#DCE4EE", width=2)
    base = top + im.height + 12
    # Keep all numbers outside the rendered robot image for LaTeX cropping.
    for i, (label, cols, prefix) in enumerate((
        ("Vanilla DP", VANILLA, "vanilla"),
        ("CANDID", GUIDED, "candid"),
    )):
        y = base + i*44
        x = 24
        for color in cols:
            d.ellipse((x, y+7, x+11, y+18), fill=color)
            x += 11
        d.text((76, y+2), label, font=font(17, True), fill="#24374D")
        line = (
            f"DR mean {_fmt(row[prefix+'_dr_mean'])}   "
            f"DR max {_fmt(row[prefix+'_dr_max'])}   "
            f"Topple {_fmt(row[prefix+'_topple'], 0)}   "
            f"Fall {_fmt(row[prefix+'_fall'], 0)}"
        )
        d.text((270, y+2), line, font=font(17), fill="#35475C")
    d.text((24, base + 88),
           "Cumulative DR up to this step | CANDID RGB background; Vanilla TCP overlay only",
           font=font(12), fill="#62728A")
    canvas.save(output)


def video_hud(frame, step, row):
    # Video annotation is compact; the PNGs carry the larger readable legend.
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle((12, 12, min(im.width-12, 750), 92),
                        radius=10, fill=(15, 31, 52, 195))
    d.text((24, 19), f"STEP {step} | Physical DR, cumulative", font=font(20, True),
           fill=(255, 255, 255, 255))
    d.text((24, 52),
           f"Vanilla DR {_fmt(row['vanilla_dr_mean'])} / {_fmt(row['vanilla_dr_max'])}"
           f"   |   CANDID DR {_fmt(row['candid_dr_mean'])} / {_fmt(row['candid_dr_max'])}",
           font=font(15), fill=(255, 255, 255, 255))
    return np.asarray(im.convert("RGB"))


def write_step_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def run_replay(replay, verb, *args):
    subprocess.run([sys.executable, str(replay), verb, *[str(v) for v in args]], check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vanilla-bundle", type=Path)
    p.add_argument("--guided-bundle", type=Path)
    p.add_argument("--vanilla-trajectory", type=Path)
    p.add_argument("--guided-trajectory", type=Path)
    p.add_argument("--eval-suite", type=Path)
    p.add_argument("--background-video", type=Path, help="Original stride-1 CANDID RGB MP4")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/paired_trajectory"))
    p.add_argument("--steps", type=int, nargs=3, metavar=("A", "B", "C"),
                   help="Optional manually chosen steps; otherwise select automatically")
    p.add_argument("--auto-steps", type=int, default=3)
    p.add_argument("--min-step-gap", type=int, default=8)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fovy", type=float, default=1.0)
    p.add_argument("--render-backend", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--no-overlay-video", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--replay-script", type=Path,
                   default=Path(__file__).resolve().parent / "replay_trajectory_rgb.py")
    a = p.parse_args(argv)
    if a.fps < 1 or a.width < 2 or a.height < 2 or a.width % 2 or a.height % 2:
        p.error("FPS must be positive and rendering resolution must be even")
    if not 0.1 < a.fovy < 3.0 or a.auto_steps < 1 or a.min_step_gap < 1 or a.smooth_window < 1:
        p.error("Invalid FOV, --auto-steps, --min-step-gap, or --smooth-window")
    if a.steps and a.auto_steps != 3:
        p.error("Use --steps or --auto-steps, not both")
    out = a.output_dir.resolve()
    frames_dir = out / "frames"
    out.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    movie = out / "paired_trajectory_overlay.mp4"
    if movie.exists() and not a.overwrite and not a.no_overlay_video:
        p.error("Overlay video exists; choose new output directory or --overwrite")
    with tempfile.TemporaryDirectory(prefix="candid-paired-") as tmp:
        tmp = Path(tmp)
        if a.vanilla_trajectory or a.guided_trajectory:
            if not (a.vanilla_trajectory and a.guided_trajectory and a.eval_suite):
                p.error("Both trajectory paths and --eval-suite are required")
            if a.vanilla_bundle or a.guided_bundle:
                p.error("Use either bundles OR raw trajectories")
            vb, gb = tmp / "vanilla.zip", tmp / "guided.zip"
            for trajectory, bundle in ((a.vanilla_trajectory, vb), (a.guided_trajectory, gb)):
                run_replay(a.replay_script, "pack", "--trajectory", trajectory,
                           "--eval-suite", a.eval_suite, "--output", bundle)
        elif a.vanilla_bundle and a.guided_bundle:
            vb, gb = a.vanilla_bundle.resolve(), a.guided_bundle.resolve()
        else:
            p.error("Provide paired ZIP bundles or paired trajectories + eval suite")
        vanilla, guided = load_episode(vb), load_episode(gb)
        check_pair(vanilla, guided)
        rows, overlap = make_step_rows(vanilla, guided)
        if a.steps is not None:
            steps = sorted(a.steps)
            if len(set(steps)) != 3 or min(steps) < 0 or max(steps) >= overlap:
                p.error(f"Manual --steps must be distinct and lie in shared recorded range [0,{overlap-1}]")
            _, gap = select_auto_steps(rows, overlap, 3, a.min_step_gap, a.smooth_window)
            for row in rows:
                row["is_selected"] = row["step"] in steps
        else:
            steps, gap = select_auto_steps(rows, overlap, a.auto_steps,
                                           a.min_step_gap, a.smooth_window)
        print(f"Vanilla {len(vanilla['track']['tcp_pose'])} frames, "
              f"CANDID {len(guided['track']['tcp_pose'])} frames; "
              f"shared measured prefix {overlap}. Selected: {steps}", flush=True)
        if a.background_video:
            background = a.background_video.resolve()
            if not background.is_file():
                raise FileNotFoundError(background)
            source = "existing CANDID RGB MP4"
        else:
            background = tmp / "guided_rgb.mp4"
            run_replay(a.replay_script, "render", "--bundle", gb,
                       "--output", background, "--render-backend", a.render_backend,
                       "--use-archived-scene-spec", "--allow-scene-spec-fallback",
                       "--width", a.width, "--height", a.height,
                       "--fps", a.fps, "--stride", 1)
            source = "auto-generated ManiSkill RGB replay"
        length = len(guided["track"]["tcp_pose"])
        reader = imageio.get_reader(str(background), format="FFMPEG")
        writer = None
        pictures = {}
        count = 0
        try:
            for step, rgb in enumerate(reader):
                if step >= length:
                    raise ValueError("Background contains more frames than Guided trajectory; use stride 1")
                if step == 0:
                    h, w = rgb.shape[:2]
                    cam = Camera(guided["scene"]["scene_spec"], w, h, a.fovy)
                    if not a.no_overlay_video:
                        writer = imageio.get_writer(str(movie), format="FFMPEG",
                                                    fps=a.fps, codec="libx264",
                                                    pixelformat="yuv420p", macro_block_size=None,
                                                    quality=8)
                painted = overlay(rgb, vanilla, guided, step, cam)
                if step in steps:
                    pictures[step] = painted.copy()
                if writer is not None:
                    writer.append_data(video_hud(painted, step, rows[step]))
                count += 1
            if count != length:
                raise ValueError(f"Background has {count} frames but CANDID recorded {length}. "
                                 "Use an unedited stride-1 RGB replay")
        finally:
            reader.close()
            if writer is not None:
                writer.close()
        selected = []
        for step in steps:
            png = frames_dir / f"step_{step:03d}.png"
            if png.exists() and not a.overwrite:
                raise FileExistsError(f"{png} already exists; pass --overwrite")
            save_separate_frame(pictures[step], step, rows[step], png)
            selected.append({**rows[step], "image": str(png)})
        csv_path = out / "step_metrics.csv"
        json_path = out / "selected_steps.json"
        write_step_csv(csv_path, rows)
        json_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False)+"\n",
                             encoding="utf-8")
        summary = {
            "scene_id": vanilla["scene"]["scene_id"],
            "same_frozen_initial_state_verified": True,
            "eval_seed": vanilla["result"]["eval_seed"],
            "vanilla_recorded_frames": len(vanilla["track"]["tcp_pose"]),
            "candid_recorded_frames": length,
            "shared_recorded_frames": overlap,
            "selected_steps": steps,
            "selection_gap": gap,
            "background_source": source,
            "step_metric": "Official cumulative physical DR from first pose to each recorded step",
            "dr_source": "candid.data.metrics.disturbance_from_poses",
            "difference_definition": "Vanilla minus CANDID; positive DR delta means improvement",
            "different_rollout_lengths": "Compare actual shared prefix only; later Vanilla metrics N/A",
            "trajectory_semantics": "Independent closed-loop rollouts from same frozen scene, not same-noise action chunks",
            "vanilla_result": {k: vanilla["result"].get(k) for k in
                               ("task_success","safe_success","dr_mean","dr_max","topple_count","fall_count")},
            "candid_result": {k: guided["result"].get(k) for k in
                              ("task_success","safe_success","dr_mean","dr_max","topple_count","fall_count")},
            "frames": [i["image"] for i in selected],
            "step_metrics_csv": str(csv_path),
            "selected_steps_json": str(json_path),
            "overlay_video": None if a.no_overlay_video else str(movie),
        }
        (out / "paired_trajectory_summary.json").write_text(json.dumps(summary, indent=2)+"\n",
                                                               encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)




if __name__ == "__main__":
    main()
