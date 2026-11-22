#!/usr/bin/env python3
"""Real ManiSkill RGB frames with Vanilla DP / CANDID recorded TCP overlays.

Input: two episode_replay.zip bundles, or two trajectory.npz files and the
frozen eval suite. No policy inference or physics resimulation.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
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


def draw_trace(frame, xyz, step, colors, camera, future_steps=12):
    """Only plot the RECORDED local TCP path [step, step+future_steps].

    The background shows the robot at 'step'. The coloured line shows its
    subsequent, already-recorded path, NOT a predictor forecast. No earlier
    global trajectory is included. Terminated episodes are never extrapolated.
    """
    if step < 0 or step >= len(xyz):
        return
    end = min(len(xyz), step + future_steps + 1)
    pixels, visible = camera.project(xyz[step:end])
    count = len(pixels)
    for i in range(1, count):
        if not (visible[i - 1] and visible[i]):
            continue
        cv2.line(
            frame,
            tuple(np.rint(pixels[i - 1]).astype(int)),
            tuple(np.rint(pixels[i]).astype(int)),
            gradient(colors, i / max(1, count - 1)), 5, cv2.LINE_AA,
        )
    for i in range(count):
        if not visible[i]:
            continue
        # A dot at each actual recorded step, bigger at the start and end.
        point = tuple(np.rint(pixels[i]).astype(int))
        radius = 8 if i in (0, count - 1) else 5
        color = gradient(colors, i / max(1, count - 1))
        cv2.circle(frame, point, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)


def overlay(frame, vanilla, guided, step, camera, future_steps=12):
    if tuple(frame.shape[:2]) != (camera.height, camera.width):
        raise ValueError("RGB frame resolution and camera projection disagree")
    out = np.ascontiguousarray(frame.copy())
    draw_trace(out, vanilla["track"]["tcp_pose"][:, :3], step,
               VANILLA, camera, future_steps=future_steps)
    draw_trace(out, guided["track"]["tcp_pose"][:, :3], step,
               GUIDED, camera, future_steps=future_steps)
    return out


def focus_crop_box(camera, vanilla, guided, steps, zoom=1.65, future_steps=12):
    """Fixed crop shared by all comparison frames; transform pixels AFTER overlay.

    Determine the centre from projected target, obstacle, goal and local TCP
    poses in the selected frames. Using the same crop for all three outputs
    preserves a comparable viewpoint. Zoom is a 2-D crop, NOT a new camera.
    """
    if zoom <= 1.0001:
        return (0, 0, camera.width, camera.height)
    points = []
    for t in steps:
        for episode in (vanilla, guided):
            track = episode["track"]
            if t >= len(track["tcp_pose"]):
                continue
            points.extend(track["distractor_poses"][t, :, :3])
            points.append(track["target_pose"][t, :3])
            points.append(track["goal_pos"][t, :3])
            end = min(len(track["tcp_pose"]), t + future_steps + 1)
            points.extend(track["tcp_pose"][t:end, :3])
    if not points:
        return (0, 0, camera.width, camera.height)
    projected, valid = camera.project(np.asarray(points, dtype=float))
    valid = valid & np.isfinite(projected).all(axis=1)
    valid &= (projected[:, 0] >= 0) & (projected[:, 0] < camera.width)
    valid &= (projected[:, 1] >= 0) & (projected[:, 1] < camera.height)
    if not valid.any():
        cx, cy = camera.width / 2, camera.height / 2
    else:
        cx, cy = np.median(projected[valid], axis=0)
    # Look a little above the tabletop so the gripper is also in view.
    cy -= .065 * camera.height
    crop_w = max(2, round(camera.width / zoom))
    crop_h = max(2, round(camera.height / zoom))
    x0 = int(np.clip(round(cx - crop_w / 2), 0, camera.width - crop_w))
    y0 = int(np.clip(round(cy - crop_h / 2), 0, camera.height - crop_h))
    return x0, y0, x0 + crop_w, y0 + crop_h


def crop_and_resize(rgb, box, width, height):
    """Zoom into the image while keeping the original robot/trajectory alignment."""
    if box == (0, 0, width, height):
        return rgb
    return np.asarray(
        Image.fromarray(rgb).crop(box).resize(
            (width, height), Image.Resampling.LANCZOS
        )
    )


@lru_cache(maxsize=64)
def font(size, bold=False):
    """Load a REAL scalable font on macOS or Linux.

    An earlier revision silently returned Pillow's tiny bitmap default font
    when a named TTF couldn't be found on macOS. That made font(40) appear
    ~10 px, regardless of the requested size. Never silently degrade.
    """
    if bold:
        candidates = (
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
            "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "Arial Bold.ttf",
        )
    else:
        candidates = (
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
            "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "Arial.ttf",
        )
    for location in candidates:
        try:
            return ImageFont.truetype(location, size)
        except (OSError, ValueError):
            pass
    # A failing lookup is preferable to unreadably small numbers in a research
    # figure. Files are never bundled, copied or shared.
    raise RuntimeError(
        "No usable TrueType/OpenType font found. Install a system Arial, "
        "Helvetica or DejaVu Sans font before rendering scientific figures."
    )


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


def select_auto_steps(rows, overlap, count, min_gap, smooth_window, future_steps=0):
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

    # Whenever possible, leave enough real recorded future steps for the
    # local trace. Otherwise fall back to the measured shared interval.
    eligible = overlap - future_steps if overlap - future_steps >= count else overlap
    effective_gap = min(min_gap, (eligible - 1) // max(1, count - 1))
    # Local maxima first, then strongest remaining steps with separation.
    local = [
        t for t in range(eligible)
        if score[t] >= (score[t-1] if t else -np.inf)
        and score[t] >= (score[t+1] if t+1 < eligible else -np.inf)
    ]
    ordered = sorted(local, key=lambda t: (-score[t], -t))
    ordered += sorted((i for i in range(eligible) if i not in local),
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


def save_separate_frame(image, step, row, output, future_steps=12):
    """Clean Beamer-ready PNG with genuinely large, high-contrast metrics.

    A compact dark masthead, untouched real RGB footage, and a two-row
    scientific comparison table. Avoid oversized empty footer cards and
    decorative text that shrink into illegibility in a three-column slide.
    """
    im = Image.fromarray(image)
    w, h = im.size
    unit = w / 1280.0

    def px(n):
        return max(1, int(round(n * unit)))

    header_h, footer_h = px(76), px(215)
    panel_y = header_h + h
    canvas = Image.new("RGB", (w, panel_y + footer_h), "#F5F8FC")
    d = ImageDraw.Draw(canvas)

    # Dark heading anchors the frame with only the information needed.
    d.rectangle((0, 0, w, header_h), fill="#172A3F")
    d.text((px(28), px(16)), f"STEP {step}",
           font=font(px(35), True), fill="white")
    d.text((px(305), px(24)),
           f"LOCAL TRAJECTORY  |  NEXT {future_steps} RECORDED STEPS",
           font=font(px(21)), fill="#B6C9DD")
    canvas.paste(im, (0, header_h))

    # Strong hierarchy: method and four physical results at full readable size.
    ink = "#172A3F"
    muted = "#687D92"
    d.text((px(30), panel_y + px(12)), "RECORDED PHYSICAL DISTURBANCE",
           font=font(px(20), True), fill=ink)
    d.text((px(30), panel_y + px(46)), "METHOD",
           font=font(px(18), True), fill=muted)
    col_x = [px(v) for v in (365, 600, 865, 1070)]
    headings = ("DR MEAN", "DR MAX", "TOPPLE", "FALL")
    for x, title in zip(col_x, headings):
        d.text((x, panel_y + px(46)), title,
               font=font(px(18), True), fill=muted)

    for i, (label, prefix, bg, accent) in enumerate((
        ("Vanilla DP", "vanilla", "#FFF5F0", "#C7502E"),
        ("CANDID", "candid", "#EDF5FF", "#2156A8"),
    )):
        y = panel_y + px(78 + i * 58)
        d.rounded_rectangle(
            (px(20), y, w - px(20), y + px(53)),
            radius=px(8), fill=bg
        )
        d.rectangle((px(20), y, px(28), y + px(53)), fill=accent)
        d.text((px(48), y + px(10)), label,
               font=font(px(25), True), fill=accent)
        for x, field, digits in zip(
            col_x, ("dr_mean", "dr_max", "topple", "fall"), (4, 4, 0, 0)
        ):
            value = _fmt(row[prefix + "_" + field], digits)
            d.text((x, y + px(1)), value,
                   font=font(px(35), True), fill=ink)

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
    p.add_argument("--trace-future", type=int, default=12,
                   help="Show only this many RECORDED steps after each chosen step (default: 12)")
    p.add_argument("--zoom", type=float, default=1.65,
                   help="Fixed crop zoom around target, obstacles and gripper (default: 1.65; 1=no crop)")
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
    if not 0.1 < a.fovy < 3.0 or a.auto_steps < 1 or a.min_step_gap < 1 or a.smooth_window < 1 or a.trace_future < 1 or not 1.0 <= a.zoom <= 3.0:
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
            _, gap = select_auto_steps(rows, overlap, 3, a.min_step_gap, a.smooth_window,
                                       future_steps=a.trace_future)
            for row in rows:
                row["is_selected"] = row["step"] in steps
        else:
            steps, gap = select_auto_steps(rows, overlap, a.auto_steps,
                                           a.min_step_gap, a.smooth_window,
                                           future_steps=a.trace_future)
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
                    crop = focus_crop_box(cam, vanilla, guided, steps,
                                          zoom=a.zoom, future_steps=a.trace_future)
                    print(f"Shared focus crop {crop}; zoom={a.zoom:.2f}x", flush=True)
                    if not a.no_overlay_video:
                        writer = imageio.get_writer(str(movie), format="FFMPEG",
                                                    fps=a.fps, codec="libx264",
                                                    pixelformat="yuv420p", macro_block_size=None,
                                                    quality=8)
                painted = overlay(rgb, vanilla, guided, step, cam,
                                  future_steps=a.trace_future)
                painted = crop_and_resize(painted, crop, w, h)
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
            save_separate_frame(pictures[step], step, rows[step], png,
                                future_steps=a.trace_future)
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
            "local_tcp_path": f"actual archived steps [t, t+{a.trace_future}] (clamped to episode length), not a forecast",
            "focus_crop_box": list(crop),
            "zoom": a.zoom,
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
