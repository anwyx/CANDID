"""Offline tests for the paired RGB overlay; no ManiSkill/Vulkan required."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_paired_trajectory.py"
spec = importlib.util.spec_from_file_location("candid_paired_video", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _bundle(path, *, guided, scene_id="scene3__random_eval_920001_00035"):
    t, n = 12, 2
    poses = np.zeros((t, n, 7), np.float32)
    poses[:, :, 3] = 1
    tcp = np.zeros((t, 7), np.float32)
    tcp[:, 3] = 1
    tcp[:, 0] = np.linspace(-.20, .08, t)
    tcp[:, 1] = np.linspace(-.08, .05, t)
    if guided:
        tcp[:, 1] += np.linspace(0, .06, t)
    target = np.zeros((t, 7), np.float32)
    target[:, 3] = 1
    joints = np.tile(np.arange(7, dtype=np.float32)[None, :], (t, 1))
    # A 7D joint vector is NOT a quaternion pose.
    goal = np.tile(np.array([[.1, 0, .001]], np.float32), (t, 1))
    buf = io.BytesIO()
    np.savez_compressed(buf, tcp_pose=tcp, robot_qpos=joints, target_pose=target,
                        distractor_poses=poses, goal_pos=goal)
    scene = {
        "scene_id": scene_id, "seed": 900001, "env_id": "Dummy-v1",
        "env_module": "envs.dummy", "env_kwargs": {},
        "scene_spec": {
            "human_camera_eye": [.52, .65, .58],
            "human_camera_look_at": [-.06, 0., .12],
        },
    }
    result = {
        "scene_id": scene_id, "eval_seed": 13502026,
        "original_env_id": "Dummy-v1",
        "task_success": guided, "safe": guided, "safe_success": guided,
        "dr_mean": .0086 if guided else .2035,
        "dr_max": .05 if guided else .6,
        "topple_count": 0 if guided else 1, "fall_count": 0,
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("trajectory.npz", buf.getvalue())
        z.writestr("initial_state.npz", b"identical archived initial state")
        z.writestr("scene.json", json.dumps(scene))
        z.writestr("result.json", json.dumps(result))
    return path


def test_matching_scene_and_pose(tmp_path):
    v = module.load_episode(_bundle(tmp_path/"v.zip", guided=False))
    g = module.load_episode(_bundle(tmp_path/"g.zip", guided=True))
    module.check_pair(v, g)
    g["result"]["eval_seed"] += 1
    with pytest.raises(ValueError, match="eval protocol"):
        module.check_pair(v, g)


def test_reject_different_scene(tmp_path):
    v = module.load_episode(_bundle(tmp_path/"v.zip", guided=False))
    g = module.load_episode(_bundle(tmp_path/"g.zip", guided=True, scene_id="scene4__different"))
    with pytest.raises(ValueError, match="initial scene"):
        module.check_pair(v, g)


def test_projection_and_overlay_progress(tmp_path):
    v = module.load_episode(_bundle(tmp_path/"v.zip", guided=False))
    g = module.load_episode(_bundle(tmp_path/"g.zip", guided=True))
    cam = module.Camera(g["scene"]["scene_spec"], 320, 180, 1.0)
    xy, good = cam.project([g["scene"]["scene_spec"]["human_camera_look_at"]])
    assert good.tolist() == [True]
    np.testing.assert_allclose(xy[0], [160, 90], atol=1e-4)
    bg = np.full((180, 320, 3), 220, dtype=np.uint8)
    early = module.overlay(bg, v, g, 0, cam)
    late = module.overlay(bg, v, g, 10, cam)
    assert not np.array_equal(early, late)


def test_generated_png_and_mp4(tmp_path):
    vb = _bundle(tmp_path/"vanilla.zip", guided=False)
    gb = _bundle(tmp_path/"guided.zip", guided=True)
    background = tmp_path / "guided_rgb.mp4"
    with imageio.get_writer(str(background), format="FFMPEG", fps=10, codec="libx264",
                            pixelformat="yuv420p", macro_block_size=None) as writer:
        for i in range(12):
            writer.append_data(np.full((180, 320, 3), [212, 221, 230], dtype=np.uint8))
    out = tmp_path / "out"
    module.main([
        "--vanilla-bundle", str(vb), "--guided-bundle", str(gb),
        "--background-video", str(background), "--output-dir", str(out),
        "--steps", "2", "6", "10", "--fps", "10", "--tile-width", "250",
    ])
    assert (out / "paired_trajectory_triptych.png").stat().st_size > 1000
    assert (out / "paired_trajectory_overlay.mp4").stat().st_size > 1000
    summary = json.loads((out / "paired_trajectory_summary.json").read_text())
    assert summary["frame_count"] == 12
    assert summary["steps"] == [2, 6, 10]
