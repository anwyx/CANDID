"""CPU-only balanced sampling, replay diagnostics, and quality auditing."""
from __future__ import annotations

import json
from collections import Counter

import h5py
import numpy as np
import pytest

from candid.data.balanced_counterfactual import (
    choose_balanced_probes, stratum_quotas,
)
from candid.data.replay_diagnostics import observation_error_breakdown
from scripts.report_counterfactual_quality import summarize


def test_stratum_quotas_exact_and_balanced():
    for count in (1, 2, 16, 32, 600):
        quotas = stratum_quotas(count)
        assert sum(quotas.values()) == count
        assert all(v >= 0 for v in quotas.values())
    assert stratum_quotas(600) == {
        "near_contact": 180, "low_dr": 150, "pre_topple": 120,
        "slight_motion": 90, "high_dr": 60,
    }


def test_observation_replay_diagnostics_reports_named_field():
    # Scene1 canonical obs_dim=151; robot qpos/qvel each 12 channels.
    source = np.zeros((2, 151), np.float32)
    live = source.copy()
    live[1, 12 + 12 + 7 + 16 + 7] = .35  # distractor_state element 7
    stats = observation_error_breakdown(live, source)
    assert stats["largest_field"] == "distractor_states"
    assert stats["per_field"]["distractor_states"]["max_abs"] == pytest.approx(.35)
    assert stats["per_field"]["distractor_states"]["time_index_of_max"] == 1
    assert stats["per_field"]["robot_qpos"]["max_abs"] == 0


def _make_expert(tmp_path):
    root = tmp_path/"expert"
    root.mkdir()
    rows = []
    num = 40
    for idx in range(num):
        scene = f"scene-{idx//4:03d}"
        traj = f"trajectory-{idx:03d}"
        trace = root/f"{traj}.npz"
        T = 45
        poses = np.zeros((T+1, 2, 7), np.float32)
        poses[:, :, 3] = 1.0
        poses[:, 0, :3] = [0.02, 0.01, 0.02]
        poses[:, 1, :3] = [0.3, .3, .02]
        topple = np.zeros((T+1, 2), bool)
        fall = np.zeros((T+1, 2), bool)
        kind = idx % 5
        if kind == 0:
            # near contact, stable.
            tcp_x = .03
        elif kind == 1:
            # quiet low risk, far from the obstacles.
            tcp_x = 1.
        elif kind == 2:
            # topple onset at t=20, probes near t=12.
            tcp_x = 1.
            topple[20:, 0] = True
        elif kind == 3:
            # mild obstacle movement; movement is per action.
            tcp_x = 1.
            poses[5:15, 0, 0] = np.arange(10)*.002 + .02
        else:
            # high DR within 8 steps, no topple.
            tcp_x = 1.
            poses[16:, 0, 0] = .08

        np.savez(trace, distractor_poses=poses, object_sizes_m=np.asarray([.04,.04]),
                 distractor_toppled=topple, distractor_fallen=fall)
        h5path = root/f"{traj}.h5"
        with h5py.File(h5path, "w") as f:
            g = f.create_group("trajectory")
            obs = g.create_group("obs")
            extra = obs.create_group("extra")
            tcp = np.zeros((T+1, 7), np.float32)
            tcp[:, 0] = tcp_x
            tcp[:, 2] = .03
            tcp[:, 3] = 1
            extra.create_dataset("tcp_pose", data=tcp)
            extra.create_dataset(
                "distractor_dimensions", data=np.tile(
                    [[.04,.04,.04], [.04,.04,.04]], (T+1,1,1)))
        rows.append({
            "scene_id": scene, "trajectory_id": traj, "saved_hdf5": True,
            "num_steps": T, "disturbance_trace": trace.name,
            "hdf5_file": h5path.name, "hdf5_trajectory_key": "trajectory",
        })
    (root/"trajectories.jsonl").write_text(
        "".join(json.dumps(x)+"\n" for x in rows))
    return root, {r["scene_id"] for r in rows}


def test_balanced_sampler_distinct_trajectories_and_scene_caps(tmp_path):
    root, scenes = _make_expert(tmp_path)
    probes = choose_balanced_probes(
        root, scenes, count=10, seed=3, reserve_count=10, max_per_scene=4)
    assert len(probes) >= 10
    assert len({p.trajectory_id for p in probes}) == len(probes)
    assert max(Counter(p.scene_id for p in probes).values()) <= 4
    primary_counts = Counter(p.stratum for p in probes[:10])
    assert primary_counts == Counter(stratum_quotas(10))


def test_balanced_sampler_raises_if_category_impossible(tmp_path):
    root, scenes = _make_expert(tmp_path)
    with pytest.raises(ValueError, match="Balanced sampling cannot fill"):
        choose_balanced_probes(
            root, scenes, count=100, seed=3, max_per_scene=2)


def test_quality_report_tracks_risk_pairs_and_delayed_topple(tmp_path):
    root = tmp_path/"cf"
    (root/"samples").mkdir(parents=True)
    labels = np.zeros((3, 4, 4), np.float32)
    labels[1, 1, 2] = 1.
    labels[2, 3, 2] = 1.
    np.savez(root/"samples"/"s.npz", targets_multi=labels)
    (root/"states.jsonl").write_text(json.dumps({
        "state_id":"s", "file":"samples/s.npz", "scene_id":"s0",
        "stratum":"low_dr",
    })+"\n")
    (root/"skipped_probes.jsonl").write_text(json.dumps({
        "reason":"Archived expert observation replay mismatch",
        "details":{"observation":{"largest_field":"robot_qvel"}}
    })+"\n")
    (root/"summary.json").write_text(json.dumps({
        "num_states":1, "complete":True, "num_candidates_per_state":3,
        "consequence_horizons":[4,8,16,32], "probe_strategy":"balanced",
        "requested_strata":{"low_dr":1},
    }))
    r = summarize(root)
    assert r["eligible_ranking_pairs"] == 3
    assert r["delayed_events"]["h8_safe_to_h32_topple"] == 1
    assert r["observation_mismatch_largest_fields"] == {"robot_qvel":1}
    assert r["stratum_shortfall"]["low_dr"] == 0
