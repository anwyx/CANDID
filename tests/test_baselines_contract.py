"""Baseline interface and frozen-split contract smoke tests (CPU)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src", ROOT / "baselines" / "act", ROOT / "baselines" / "bc"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def test_demo_export_frozen_split(tmp_path):
    from candid.data.synthetic import write_synthetic_dataset
    from baselines.common.export_demos import export_demos
    import h5py
    root = write_synthetic_dataset(tmp_path / "data")
    output = tmp_path / "bc_train.h5"
    summary = export_demos(root, output)
    assert summary["trajectories"] > 0
    assert summary["action_dim"] == 7
    with h5py.File(output, "r") as f:
        assert f["traj_0/actions"].shape[1] == 7
        assert len(f["traj_0/obs"]) == len(f["traj_0/actions"]) + 1
    with pytest.raises(FileExistsError):
        export_demos(root, output)


def test_benchmark_policy_shape():
    pytest.importorskip("gymnasium")
    from baselines.common.eval_baseline import BaselinePolicy
    from baselines.bc.bc import Actor
    model = Actor(state_dim=12, action_dim=7)
    policy = BaselinePolicy(model, "bc", "cpu")
    assert policy.act(np.zeros((1, 12), dtype=np.float32)).shape == (1, 7)


def test_ppo_sac_actor_network_shapes():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")
    from baselines.ppo.ppo import Agent
    from baselines.sac.sac import Actor
    fake = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
    )
    x = torch.zeros((1, 24), dtype=torch.float32)
    assert Agent(fake).get_action(x, deterministic=True).shape == (1, 7)
    assert Actor(fake).get_eval_action(x).shape == (1, 7)


def test_act_state_model_forward():
    pytest.importorskip("gymnasium")
    pytest.importorskip("diffusers")
    from baselines.act.train import Agent, Args
    fake = SimpleNamespace(
        single_observation_space=SimpleNamespace(shape=(24,)),
        single_action_space=SimpleNamespace(shape=(7,)),
    )
    args = Args(num_queries=8)
    model = Agent(fake, args)
    with torch.no_grad():
        action = model.get_action(torch.zeros((1, 24)))
    assert action.shape == (1, 8, 7)
