"""Save and reload Vanilla DP weights, EMA weights, and action bounds."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from model.vanilla_dp.config import DPConfig, MANISKILL_COMMIT
from model.vanilla_dp.normalizer import ActionRangeNormalizer, ObservationNormalizer
from model.vanilla_dp.policy import Agent, VanillaDPPolicy


def save_checkpoint(path, agent: Agent, ema_agent: Agent, config: DPConfig, normalizer: ActionRangeNormalizer,
                    obs_normalizer: ObservationNormalizer | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "vanilla_dp",
            "maniskill_commit": MANISKILL_COMMIT,
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
            "config": asdict(config),
            "action_min": normalizer.minimum.detach().cpu(),
            "action_max": normalizer.maximum.detach().cpu(),
            "obs_dim": agent.obs_dim,
            "action_dim": agent.act_dim,
            "obs_normalizer": None if obs_normalizer is None else {
                "mean": obs_normalizer.mean, "std": obs_normalizer.std, "clip": obs_normalizer.clip,
            },
        },
        path,
    )
    return path


def load_policy(path, device="cpu") -> VanillaDPPolicy:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "vanilla_dp":
        raise ValueError(f"Expected a vanilla_dp checkpoint, found {payload.get('kind')}")
    # Dataset split ratios are training-only bookkeeping.  Some checkpoints
    # originate from external or all-data training workflows that store
    # test_ratio=0; inference should not reject their trained EMA weights.
    # Reconstruct all architecture / diffusion settings unchanged, while
    # letting split-only fields take harmless DPConfig defaults.
    config_values = dict(payload["config"])
    config_values.pop("val_ratio", None)
    config_values.pop("test_ratio", None)
    config = DPConfig(**config_values)
    agent = Agent(payload["obs_dim"], payload["action_dim"], config).to(device)
    agent.load_state_dict(payload["ema_agent"])
    agent.eval()
    normalizer = ActionRangeNormalizer(payload["action_min"], payload["action_max"]).to(device)
    stats = payload.get("obs_normalizer")
    obs_normalizer = None if stats is None else ObservationNormalizer(stats["mean"], stats["std"], stats["clip"])
    return VanillaDPPolicy(agent, normalizer, obs_normalizer)
