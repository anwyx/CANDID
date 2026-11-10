"""CPU-only tests for train-only counterfactual sampling and perturbations."""
import json

import numpy as np
import pytest

from candid.data.counterfactual import (choose_probes, make_candidates,
                                        make_multiscale_candidates, stable_seed)


def _trajectory(root, scene, index, topple=False, displacement=0.0):
    length = 24
    poses = np.zeros((length+1, 2, 7), dtype=np.float32)
    poses[:, :, 3] = 1
    poses[:, :, 2] = .05
    poses[12:, 0, 0] = displacement
    toppled = np.zeros((length+1, 2), dtype=bool)
    if topple:
        toppled[12:, 0] = True
    fallen = np.zeros_like(toppled)
    filename = f"{scene}-{index}.npz"
    np.savez_compressed(root / filename, distractor_poses=poses,
                        object_sizes_m=np.array([.1,.1]),
                        distractor_toppled=toppled, distractor_fallen=fallen)
    return {"scene_id":scene, "trajectory_id":f"{scene}-{index}",
            "num_steps":length, "disturbance_trace":filename,
            "saved_hdf5":True}


def test_smooth_candidates_are_bounded_deterministic_and_keep_expert():
    expert = np.full((8,7), .5, dtype=np.float32)
    low, high = np.zeros(7), np.ones(7)
    got = make_candidates(expert, low, high, count=5, sigma=.02, seed=31)
    again = make_candidates(expert, low, high, count=5, sigma=.02, seed=31)
    assert got.shape == (5,8,7)
    np.testing.assert_array_equal(got, again)
    np.testing.assert_array_equal(got[0], expert)
    np.testing.assert_array_equal(got[:,:,-1], np.full((5,8), .5))
    assert np.all(got>=low) and np.all(got<=high)
    assert all(np.max(np.abs(x-expert)) > .002 for x in got[1:])
    assert np.max(np.abs(np.diff(got[1]-expert, axis=0))) < .08


def test_sampling_excludes_test_scenes_and_covers_strata(tmp_path):
    rows = [_trajectory(tmp_path,"train-A",0,topple=True),
            _trajectory(tmp_path,"train-B",0,displacement=.025),
            _trajectory(tmp_path,"train-C",0),
            _trajectory(tmp_path,"test-X",0,topple=True)]
    (tmp_path/"trajectories.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    probes = choose_probes(tmp_path,["train-A","train-B","train-C"],
                          count=3,seed=1,max_per_scene=2)
    assert len(probes)==3
    assert {p.scene_id for p in probes}=={"train-A","train-B","train-C"}
    assert {p.stratum for p in probes}=={"pre_topple","high_dr","low_dr"}
    onset = next(p for p in probes if p.stratum=="pre_topple")
    assert onset.step==4
    assert stable_seed(12,onset.trajectory_id,onset.step)==stable_seed(12,onset.trajectory_id,onset.step)
    assert stable_seed(12,onset.trajectory_id,onset.step)!=stable_seed(12,onset.trajectory_id,5)


def test_invalid_candidate_config_and_insufficient_data(tmp_path):
    expert=np.full((8,7),.5)
    with pytest.raises(ValueError,match="candidate"):
        make_candidates(expert,np.zeros(7),np.ones(7),count=1,seed=0)
    with pytest.raises(ValueError,match="bounds"):
        make_candidates(expert,np.ones(7),np.zeros(7),count=2,seed=0)
    with pytest.raises(ValueError,match="exceed"):
        make_candidates(expert*3,np.zeros(7),np.ones(7),count=2,seed=0)
    row=_trajectory(tmp_path,"train-A",0)
    (tmp_path/"trajectories.jsonl").write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="Requested 2 train probes"):
        choose_probes(tmp_path,["train-A"],count=2,seed=0)


def test_probe_prefix_bound(tmp_path):
    row=_trajectory(tmp_path,"train-A",0,topple=True)
    (tmp_path/"trajectories.jsonl").write_text(json.dumps(row)+"\n")
    probe=choose_probes(tmp_path,["train-A"],count=1,max_prefix_steps=3)[0]
    assert probe.step<=3
    with pytest.raises(ValueError,match="max_prefix_steps"):
        choose_probes(tmp_path,["train-A"],count=1,max_prefix_steps=0)


def test_multiscale_pairs_cover_signed_directions_and_scales():
    expert = np.full((8, 7), 0.5, dtype=np.float32)
    low, high = np.zeros(7), np.ones(7)
    chunks, sigmas = make_multiscale_candidates(
        expert, low, high, count=7, seed=42,
        sigma_levels=(0.02, 0.06, 0.12))
    repeat, _ = make_multiscale_candidates(
        expert, low, high, count=7, seed=42,
        sigma_levels=(0.02, 0.06, 0.12))
    np.testing.assert_array_equal(chunks, repeat)
    np.testing.assert_array_equal(chunks[0], expert)
    np.testing.assert_allclose(sigmas, [0, .02, .02, .06, .06, .12, .12], atol=1e-7)
    assert chunks.shape == (7, 8, 7)
    assert np.isfinite(chunks).all()
    assert (chunks >= low).all() and (chunks <= high).all()
    np.testing.assert_array_equal(chunks[:, :, -1], np.full((7, 8), 0.5))
    for i in (1, 3, 5):
        np.testing.assert_allclose(chunks[i] + chunks[i+1], 2*expert, atol=1e-6)
    scales = [np.linalg.norm((chunks[i] - expert).ravel()) for i in (1, 3, 5)]
    assert scales[0] < scales[1] < scales[2]


def test_multiscale_validates_invalid_candidates():
    expert = np.full((8, 7), .5, dtype=np.float32)
    with pytest.raises(ValueError, match=">=3"):
        make_multiscale_candidates(expert, np.zeros(7), np.ones(7), count=2, seed=0)
    with pytest.raises(ValueError, match="sigma levels"):
        make_multiscale_candidates(expert, np.zeros(7), np.ones(7), count=3,
                                   seed=0, sigma_levels=(-.02,))
