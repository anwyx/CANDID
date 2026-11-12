"""Headless paired-RGB visualization tests (no ManiSkill/Vulkan required)."""
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
SPEC = importlib.util.spec_from_file_location(
    "candid_paired_video", ROOT / "scripts" / "render_paired_trajectory.py"
)
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def bundle(path, length, *, guided, scene_id="scene3__random_eval_920001_00035"):
    n = 2
    poses = np.zeros((length, n, 7), np.float32)
    poses[:, :, 3] = 1
    poses[:, 0, 0] = -0.05
    poses[:, 1, 1] = 0.07
    if guided:
        poses[:, 0, 0] += np.linspace(0, .02, length)
    else:
        poses[:, 0, 0] += np.linspace(0, .10, length)
    topple = np.zeros((length, n), bool)
    if not guided:
        topple[4:, 0] = True
    fall = np.zeros((length, n), bool)
    sizes = np.array([.04, .05], dtype=np.float32)
    tcp = np.zeros((length, 7), np.float32)
    tcp[:, 3] = 1.
    tcp[:, 0] = np.linspace(-.2, .08, length)
    tcp[:, 1] = np.linspace(-.08, .03 if guided else -.03, length)
    target = np.zeros((length, 7), np.float32)
    target[:, 3] = 1
    goal = np.tile(np.array([[.1, 0, .001]], np.float32), (length, 1))
    qpos = np.tile(np.arange(7, dtype=np.float32)[None], (length, 1))
    # For test fixture correctness, derive final metrics from the official function.
    from candid.data.metrics import disturbance_from_poses
    dr = disturbance_from_poses(
        poses, sizes, topple_flags=topple, fall_flags=fall
    )
    with io.BytesIO() as b:
        np.savez_compressed(
            b, tcp_pose=tcp, robot_qpos=qpos, target_pose=target,
            distractor_poses=poses, goal_pos=goal, object_sizes_m=sizes,
            distractor_toppled=topple, distractor_fallen=fall,
        )
        blob = b.getvalue()
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
        "original_env_id": "Dummy-v1", "task_success": guided,
        "safe_success": guided,
        "dr_mean": dr["dr_mean"], "dr_max": dr["dr_max"],
        "topple_count": dr["topple_count"], "fall_count": dr["fall_count"],
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("trajectory.npz", blob)
        z.writestr("initial_state.npz", b"same frozen initial state")
        z.writestr("scene.json", json.dumps(scene))
        z.writestr("result.json", json.dumps(result))
    return path


def write_background(path, n):
    with imageio.get_writer(
        str(path), format="FFMPEG", fps=10, codec="libx264",
        pixelformat="yuv420p", macro_block_size=None
    ) as writer:
        for _ in range(n):
            writer.append_data(np.full((180, 320, 3), (210, 220, 230), np.uint8))


def test_different_lengths_and_exact_physical_dr(tmp_path):
    v = mod.load_episode(bundle(tmp_path/"v.zip", 17, guided=False))
    g = mod.load_episode(bundle(tmp_path/"g.zip", 11, guided=True))
    mod.check_pair(v, g)
    rows, overlap = mod.make_step_rows(v, g)
    assert overlap == 11
    assert len(rows) == 11
    assert rows[0]["vanilla_dr_mean"] == 0
    assert rows[0]["candid_dr_max"] == 0
    assert rows[-1]["delta_dr_mean"] > 0
    assert rows[-1]["vanilla_topple"] == 1
    assert rows[-1]["candid_topple"] == 0
    steps, gap = mod.select_auto_steps(rows, overlap, 3, 8, 5)
    assert len(steps) == len(set(steps)) == 3
    assert all(0 <= step < overlap for step in steps)


def test_longer_guided_episode_marks_missing_vanilla(tmp_path):
    v = mod.load_episode(bundle(tmp_path/"v.zip", 10, guided=False))
    g = mod.load_episode(bundle(tmp_path/"g.zip", 18, guided=True))
    rows, overlap = mod.make_step_rows(v, g)
    assert overlap == 10
    assert rows[9]["vanilla_dr_mean"] is not None
    assert rows[10]["vanilla_dr_mean"] is None
    assert rows[10]["both_recorded"] is False
    steps, _ = mod.select_auto_steps(rows, overlap, 3, 8, 5)
    assert max(steps) < 10


def test_generate_three_individual_pngs_and_step_csv(tmp_path):
    vanilla = bundle(tmp_path/"vanilla.zip", 18, guided=False)
    guided = bundle(tmp_path/"guided.zip", 12, guided=True)
    background = tmp_path/"background.mp4"
    write_background(background, 12)
    out = tmp_path/"out"
    mod.main([
        "--vanilla-bundle", str(vanilla),
        "--guided-bundle", str(guided),
        "--background-video", str(background),
        "--output-dir", str(out), "--fps", "10",
        "--auto-steps", "3", "--min-step-gap", "4",
        "--no-overlay-video",
    ])
    pngs = list((out/"frames").glob("step_*.png"))
    assert len(pngs) == 3
    for png in pngs:
        assert png.stat().st_size > 1000
    rows = (out/"step_metrics.csv").read_text().splitlines()
    assert len(rows) == 13  # header + 12 CANDID frames
    summary = json.loads((out/"paired_trajectory_summary.json").read_text())
    assert summary["shared_recorded_frames"] == 12
    assert len(summary["selected_steps"]) == 3


def test_reject_different_scene(tmp_path):
    v = mod.load_episode(bundle(tmp_path/"v.zip", 12, guided=False))
    g = mod.load_episode(bundle(tmp_path/"g.zip", 12, guided=True,
                                scene_id="different_scene"))
    with pytest.raises(ValueError, match="initial scene"):
        mod.check_pair(v, g)
