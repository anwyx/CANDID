#!/usr/bin/env python3
"""Real ManiSkill RGB frames with Vanilla DP / CANDID recorded TCP overlays.

Input: two episode_replay.zip bundles, or two trajectory.npz files and the
frozen eval suite. No policy inference or physics resimulation.
"""
from __future__ import annotations

import argparse
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

VANILLA = ((255, 244, 82), (246, 181, 49), (239, 105, 42), (210, 42, 44))
GUIDED = ((105, 243, 252), (64, 196, 253), (42, 112, 245), (32, 38, 213))
KEYS = ("tcp_pose", "robot_qpos", "target_pose", "distractor_poses", "goal_pos")


def load_episode(path):
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        needed = {"trajectory.npz", "initial_state.npz", "scene.json", "result.json"}
        if not needed <= names:
            raise ValueError(f"Bundle {path} missing {sorted(needed - names)}")
        scene = json.loads(z.read("scene.json"))
        result = json.loads(z.read("result.json"))
        digest = hashlib.sha256(z.read("initial_state.npz")).hexdigest()
        with np.load(io.BytesIO(z.read("trajectory.npz")), allow_pickle=False) as a:
            if not set(KEYS) <= set(a.files):
                raise ValueError(f"Trajectory missing {sorted(set(KEYS) - set(a.files))}")
            track = {k: a[k].copy() for k in KEYS}
    if scene["scene_id"] != result["scene_id"]:
        raise ValueError("Bundle scene_id mismatch")
    t = len(track["tcp_pose"])
    if t < 2 or track["tcp_pose"].shape != (t, 7) or track["target_pose"].shape != (t, 7):
        raise ValueError("Invalid TCP or target shape")
    if track["robot_qpos"].ndim != 2 or track["robot_qpos"].shape[0] != t:
        raise ValueError("Invalid robot_qpos shape")
    if track["distractor_poses"].ndim != 3 or track["distractor_poses"].shape[0] != t or track["distractor_poses"].shape[-1] != 7:
        raise ValueError("Invalid distractor_poses shape")
    if track["goal_pos"].shape != (t, 3):
        raise ValueError("Invalid goal_pos shape")
    if not all(np.isfinite(v).all() for v in track.values()):
        raise ValueError("Non-finite recorded poses")
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


def montage(frames, steps, vanilla, guided, tile_width):
    h, w = frames[0].shape[:2]
    th = round(h * tile_width / w)
    left, gap, top, bottom = 24, 14, 138, 95
    width = left*2 + 3*tile_width + 2*gap
    output = Image.new("RGB", (width, top + th + bottom), "#F8FAFC")
    d = ImageDraw.Draw(output)
    d.text((left, 13), "Unguided vs Guided Trajectory",
           font=font(27, True), fill="#1C2B40")
    x = left
    for colors, label in ((VANILLA, "Vanilla DP (unguided)"), (GUIDED, "CANDID (guided)")):
        for color in colors:
            d.ellipse((x, 65, x+16, 81), fill=color)
            x += 13
        x += 11
        d.text((x, 62), label, font=font(18, True), fill="#2B3D52")
        x += 260 if label.startswith("Vanilla") else 0
    d.text((left, 104), "Vanilla: " + metrics(vanilla) + "    CANDID: " + metrics(guided),
           font=font(14), fill="#52647A")
    for i, (frame, step) in enumerate(zip(frames, steps)):
        x = left + i*(tile_width+gap)
        image = Image.fromarray(frame).resize((tile_width, th), Image.Resampling.LANCZOS)
        output.paste(image, (x, top))
        d.rectangle((x, top, x+tile_width-1, top+th-1), outline="#CBD4E0", width=2)
        label = f"STEP = {step}"
        bbox = d.textbbox((0, 0), label, font=font(21, True))
        lw = bbox[2] - bbox[0]
        d.rounded_rectangle((x + tile_width-lw-30, top+10,
                             x + tile_width-10, top+46), radius=7, fill="#263649")
        d.text((x + tile_width-lw-20, top+13), label, font=font(21, True), fill="white")
    d.text((left, top+th+14), f"Scene: {vanilla['scene']['scene_id']}  |  Same frozen initial state and eval seed",
           font=font(14, True), fill="#33465D")
    d.text((left, top+th+46),
           "Background: CANDID RGB replay. Vanilla DP is shown only as an overlaid recorded TCP path.",
           font=font(13), fill="#61748C")
    return output


def video_hud(frame, step):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle((16, 16, min(im.width-16, 535), 79), radius=10, fill=(14, 30, 47, 190))
    d.text((29, 20), f"STEP {step}", font=font(21, True), fill="white")
    d.text((29, 50), "Yellow-red: Vanilla DP   Cyan-blue: CANDID",
           font=font(13), fill="white")
    return np.asarray(im)


def run_replay(replay, verb, *args):
    subprocess.run([sys.executable, str(replay), verb, *[str(v) for v in args]], check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vanilla-bundle", type=Path)
    p.add_argument("--guided-bundle", type=Path)
    p.add_argument("--vanilla-trajectory", type=Path)
    p.add_argument("--guided-trajectory", type=Path)
    p.add_argument("--eval-suite", type=Path)
    p.add_argument("--background-video", type=Path,
                   help="Optional existing unedited stride-1 CANDID RGB MP4")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/paired_trajectory"))
    p.add_argument("--steps", type=int, nargs=3, metavar=("A","B","C"))
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--tile-width", type=int, default=560)
    p.add_argument("--fovy", type=float, default=1.0)
    p.add_argument("--render-backend", choices=("cpu","gpu"), default="cpu")
    p.add_argument("--no-overlay-video", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--replay-script", type=Path,
                   default=Path(__file__).resolve().parent / "replay_trajectory_rgb.py")
    a = p.parse_args(argv)

    if a.fps < 1 or a.width < 2 or a.height < 2 or a.width%2 or a.height%2 or a.tile_width < 100:
        p.error("Require fps>=1, even positive render dimensions and tile-width>=100")
    if not (.1 < a.fovy < 3.):
        p.error("--fovy should be radians in (0.1,3.0)")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    png = out / "paired_trajectory_triptych.png"
    movie = out / "paired_trajectory_overlay.mp4"
    if not a.overwrite and (png.exists() or (not a.no_overlay_video and movie.exists())):
        p.error("Output already exists; choose another --output-dir or --overwrite")

    with tempfile.TemporaryDirectory(prefix="candid-paired-") as tmp:
        tmp = Path(tmp)
        if a.vanilla_trajectory or a.guided_trajectory:
            if not (a.vanilla_trajectory and a.guided_trajectory and a.eval_suite):
                p.error("Both raw trajectory paths require --eval-suite")
            if a.vanilla_bundle or a.guided_bundle:
                p.error("Choose bundles OR raw trajectories, not both")
            vb, gb = tmp/"vanilla.zip", tmp/"guided.zip"
            for trajectory, bundle in ((a.vanilla_trajectory, vb), (a.guided_trajectory, gb)):
                run_replay(a.replay_script, "pack",
                           "--trajectory", trajectory, "--eval-suite", a.eval_suite, "--output", bundle)
        elif a.vanilla_bundle and a.guided_bundle:
            vb, gb = a.vanilla_bundle.resolve(), a.guided_bundle.resolve()
        else:
            p.error("Provide both --vanilla-bundle and --guided-bundle")
        vanilla, guided = load_episode(vb), load_episode(gb)
        check_pair(vanilla, guided)

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
        steps = a.steps or [round((length-1)*k) for k in (.25,.5,.75)]
        if len(set(steps)) != 3 or min(steps) < 0 or max(steps) >= length:
            p.error(f"Require 3 distinct --steps in [0,{length-1}]")
        reader = imageio.get_reader(str(background), format="FFMPEG")
        writer = None
        screenshots = {}
        frames = 0
        try:
            for step, rgb in enumerate(reader):
                if step >= length:
                    raise ValueError("Background MP4 has more frames than Guided trajectory")
                if step == 0:
                    h, w = rgb.shape[:2]
                    cam = Camera(guided["scene"]["scene_spec"], w, h, a.fovy)
                    if not a.no_overlay_video:
                        writer = imageio.get_writer(str(movie), format="FFMPEG", fps=a.fps,
                                                    codec="libx264", pixelformat="yuv420p",
                                                    macro_block_size=None, quality=8)
                painted = overlay(rgb, vanilla, guided, step, cam)
                if step in steps:
                    screenshots[step] = painted.copy()
                if writer is not None:
                    writer.append_data(video_hud(painted, step))
                frames += 1
            if frames != length:
                raise ValueError(f"Background has {frames} frames, Guided trajectory has {length}; render with --stride 1")
        finally:
            reader.close()
            if writer is not None:
                writer.close()
        montage([screenshots[i] for i in steps], steps,
                vanilla, guided, a.tile_width).save(png)
        summary = {
            "scene_id": vanilla["scene"]["scene_id"],
            "same_frozen_initial_state_verified": True,
            "eval_seed": vanilla["result"]["eval_seed"],
            "visualization": "CANDID RGB background; Vanilla recorded TCP overlay",
            "pair_semantics": "Independent closed-loop episodes on the same frozen initial scene; not same-noise action chunks",
            "background_source": source,
            "steps": steps,
            "frame_count": frames,
            "vanilla": {k: vanilla["result"].get(k) for k in
                        ("task_success","safe_success","dr_mean","dr_max","topple_count","fall_count")},
            "candid": {k: guided["result"].get(k) for k in
                       ("task_success","safe_success","dr_mean","dr_max","topple_count","fall_count")},
            "triptych": str(png),
            "video": None if a.no_overlay_video else str(movie),
        }
        (out/"paired_trajectory_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
