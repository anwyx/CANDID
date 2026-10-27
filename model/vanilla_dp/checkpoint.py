"""Save and reload Vanilla DP weights, EMA weights, and action bounds."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from model.vanilla_dp.config import DPConfig, MANISKILL_COMMIT
from model.vanilla_dp.normalizer import ActionRangeNormalizer
from model.vanilla_dp.policy import Agent, VanillaDPPolicy


def save_checkpoint(path, agent: Agent, ema_agent: Agent, config: DPConfig, normalizer: ActionRangeNormalizer):
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
        },
        path,
    )
    return path


def load_policy(path, device="cpu") -> VanillaDPPolicy:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "vanilla_dp":
        raise ValueError(f"Expected a vanilla_dp checkpoint, found {payload.get('kind')}")
    config = DPConfig(**payload["config"])
    agent = Agent(payload["obs_dim"], payload["action_dim"], config).to(device)
    agent.load_state_dict(payload["ema_agent"])
    agent.eval()
    normalizer = ActionRangeNormalizer(payload["action_min"], payload["action_max"]).to(device)
    return VanillaDPPolicy(agent, normalizer)
