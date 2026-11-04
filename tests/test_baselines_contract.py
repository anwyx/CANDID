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
    root = write_synthetic_dataset(tmp_path / "data", num_scenes=6)
    output = tmp_path / "bc_train.h5"
    summary = export_demos(root, output, max_trajectories=2)
    assert summary["trajectories"] == 2
    assert summary["action_dim"] == 7
    with h5py.File(output, "r") as f:
        assert f["traj_0/actions"].shape[1] == 7
        assert len(f["traj_0/obs"]) == len(f["traj_0/actions"]) + 1
    import json
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["candid"]["source_environments"]
    assert metadata["candid"]["env_info_is_representative"] is True
    assert metadata["candid"]["max_trajectories"] == 2
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
    pytest.importorskip("mani_skill")
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


def test_act_ema_matches_upstream_default_decay():
    """ACT's EMA must preserve diffusers 0.32.2 non-warmup behavior."""
    from baselines.act.ema import EMAModel

    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    ema = EMAModel([parameter], power=0.75)
    with torch.no_grad():
        parameter.fill_(2.0)
    ema.step([parameter])
    assert ema.get_decay(1) == 0.0

    with torch.no_grad():
        parameter.fill_(4.0)
    ema.step([parameter])
    assert np.isclose(ema.get_decay(2), 2.0 / 11.0)
    destination = torch.nn.Parameter(torch.zeros_like(parameter))
    ema.copy_to([destination])
    expected = (2.0 / 11.0) * 2.0 + (9.0 / 11.0) * 4.0
    assert torch.allclose(destination, torch.tensor([expected]))


def test_cpu_vector_wrapper_exposes_canonical_observation_space():
    """Regression: CPUGymWrapper must not expose its base state_dict space."""
    gym = pytest.importorskip("gymnasium")
    from baselines.common.candid_env import restore_cpu_canonical_space

    raw_space = gym.spaces.Dict({
        "agent": gym.spaces.Dict({"qpos": gym.spaces.Box(-1, 1, (2,), dtype=np.float32)})
    })
    canonical = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    class DummyCPU(gym.Env):
        observation_space = raw_space
        action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        def reset(self, *, seed=None, options=None):
            return np.zeros(4, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(4, dtype=np.float32), 0.0, False, False, {}
    wrapped = restore_cpu_canonical_space(DummyCPU(), canonical)
    assert wrapped.observation_space.shape == (4,)
    with gym.vector.SyncVectorEnv([lambda: wrapped]) as vector:
        obs, _ = vector.reset()
        assert obs.shape == (1, 4)


def test_cpu_imitation_wrapper_vector_reset_contract():
    """Reproduce CPUGymWrapper's Dict-space overwrite, then vector reset.

    Does not require ManiSkill/SAPIEN: the fake wrapper reproduces its
    relevant reset/space behavior, and ensures the common adapter repairs it.
    """
    gym = pytest.importorskip("gymnasium")
    from baselines.common.candid_env import wrap_cpu_canonical_state

    old_space = gym.spaces.Dict({
        "extra": gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
    })
    canonical = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)

    class FakeCanonical(gym.Env):
        def __init__(self):
            self.single_observation_space = canonical
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1, 4), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)
            self.base_raw_space = old_space

        def reset(self, *, seed=None, options=None):
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            return np.ones(4, dtype=np.float32), 0.0, False, False, {}

    class FakeCPUGymWrapper(gym.Wrapper):
        def __init__(self, env, ignore_terminations=False, record_metrics=False):
            super().__init__(env)
            self.observation_space = env.base_raw_space

    def make():
        return wrap_cpu_canonical_state(
            FakeCanonical(), FakeCPUGymWrapper,
            ignore_terminations=True, record_metrics=True
        )

    with gym.vector.SyncVectorEnv([make]) as envs:
        assert isinstance(envs.single_observation_space, gym.spaces.Box)
        obs, _ = envs.reset()
        assert obs.shape == (1, 4)
        obs, *_ = envs.step(np.zeros((1, 7), dtype=np.float32))
        assert obs.shape == (1, 4)


def test_act_and_bc_cpu_make_env_use_space_repair():
    """Prevent a repeat of the prior bug: helper existed but was never called."""
    root = ROOT / "baselines"
    for rel in ("bc/behavior_cloning/make_env.py", "act/act/make_env.py"):
        source = (root / rel).read_text(encoding="utf-8")
        assert "wrap_cpu_canonical_state(" in source, rel
        assert "env = CPUGymWrapper(env," not in source, rel
