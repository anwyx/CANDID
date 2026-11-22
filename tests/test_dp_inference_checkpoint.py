"""Checkpoint inference must ignore dataset-split bookkeeping, not model parameters."""
from __future__ import annotations

import inspect

import torch

from model.vanilla_dp.checkpoint import load_policy
from model.vanilla_dp.config import DPConfig
from model.vanilla_dp.normalizer import ActionRangeNormalizer
from model.vanilla_dp.policy import Agent


def test_load_checkpoint_with_zero_test_ratio(tmp_path):
    cfg = DPConfig(obs_horizon=2, act_horizon=8, pred_horizon=16,
                   num_diffusion_iters=100)
    agent = Agent(obs_dim=20, action_dim=7, config=cfg)
    from dataclasses import asdict
    config = asdict(cfg)
    config["test_ratio"] = 0.0
    config["val_ratio"] = 0.0
    path = tmp_path / "checkpoint.pt"
    torch.save({
        "kind": "vanilla_dp", "config": config, "obs_dim": 20, "action_dim": 7,
        "ema_agent": agent.state_dict(),
        "action_min": torch.full((7,), -1.0),
        "action_max": torch.full((7,), 1.0),
    }, path)
    loaded = load_policy(path, device="cpu")
    assert loaded.agent.act_horizon == cfg.act_horizon
    assert loaded.agent.pred_horizon == cfg.pred_horizon
    assert loaded.agent.num_diffusion_iters == cfg.num_diffusion_iters
    assert loaded.agent.obs_dim == 20
    assert loaded.agent.act_dim == 7


def test_inference_loader_does_not_override_architecture():
    source = inspect.getsource(load_policy)
    assert 'config_values.pop("test_ratio", None)' in source
    assert 'config_values.pop("val_ratio", None)' in source
    assert 'agent.load_state_dict(payload["ema_agent"])' in source
