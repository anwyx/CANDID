"""Evaluate ManiSkill ACT/BC/PPO/SAC on CANDID's frozen TEST scenes.

All four policies implement candid.eval.rollout's act(window) protocol, and
therefore share episode DR / topple / fall / success definitions with Vanilla DP.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "src", ROOT / "baselines" / "act", ROOT / "baselines" / "bc"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import gymnasium as gym
import numpy as np
import torch

from candid.eval.metrics import SafetyLimits
from candid.eval.rollout import evaluate_dataset


def _checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Unsupported checkpoint payload")
    return payload


def _dummy_env(obs_dim, action_dim):
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        single_action_space=gym.spaces.Box(-1, 1, (action_dim,), np.float32),
    )


class BaselinePolicy:
    obs_horizon = 1

    def __init__(self, model, method, device, stats=None, queries=1, temporal_agg=False):
        self.model = model.eval()
        self.method = method
        self.device = torch.device(device)
        self.stats = stats or {}
        self.num_queries = int(queries)
        self.temporal_agg = bool(temporal_agg)
        if method == "act":
            self.action_horizon = 1 if self.temporal_agg else self.num_queries
        else:
            self.action_horizon = 1
        self.obs_dim = int(self.model_obs_dim())
        self.action_dim = int(self.model_action_dim())
        self.begin_episode(0)

    def model_obs_dim(self):
        if self.method == "act":
            return self.model.state_dim
        if self.method == "bc":
            return self.model.net[0].in_features
        if self.method == "ppo":
            return self.model.actor_mean[0].in_features
        return self.model.backbone[0].in_features

    def model_action_dim(self):
        if self.method == "act":
            return self.model.act_dim
        if self.method == "bc":
            return self.model.net[-1].out_features
        if self.method == "ppo":
            return self.model.actor_mean[-1].out_features
        return self.model.fc_mean.out_features

    def begin_episode(self, seed):
        self.step_index = 0
        self.previous_chunks = []
        torch.manual_seed(int(seed))

    def _tensor_stat(self, name, default):
        return torch.as_tensor(self.stats.get(name, default), dtype=torch.float32,
                               device=self.device)

    @torch.no_grad()
    def act(self, observation_window):
        obs = np.asarray(observation_window, dtype=np.float32)
        if obs.shape != (1, self.obs_dim):
            raise ValueError(f"Expected [1,{self.obs_dim}] observation, got {obs.shape}")
        state = torch.from_numpy(obs).to(self.device)

        if self.method == "act":
            state = (state - self._tensor_stat("state_mean", 0.0)) / self._tensor_stat("state_std", 1.0)
            output = self.model.get_action(state)[0]
            output = output * self._tensor_stat("action_std", 1.0) + self._tensor_stat("action_mean", 0.0)
            if self.temporal_agg:
                self.previous_chunks.append((self.step_index, output))
                active = [value[self.step_index - start] for start, value in self.previous_chunks
                          if start <= self.step_index < start + len(value)]
                if not active:
                    raise RuntimeError("ACT temporal aggregation has no predictions")
                k = torch.arange(len(active), device=self.device, dtype=torch.float32)
                weights = torch.exp(-0.01 * k)
                weights /= weights.sum()
                output = (torch.stack(active, 0) * weights[:, None]).sum(0, keepdim=True)
                self.previous_chunks = [(start, value) for start, value in self.previous_chunks
                                        if start + len(value) > self.step_index + 1]
                self.step_index += 1
            else:
                self.step_index += len(output)
        elif self.method == "bc":
            state = (state - self._tensor_stat("observation_mean", 0.0)) / self._tensor_stat("observation_std", 1.0)
            output = self.model(state)
        elif self.method == "ppo":
            output = self.model.get_action(state, deterministic=True)
        elif self.method == "sac":
            output = self.model.get_eval_action(state)
        else:
            raise ValueError(self.method)
        result = output.cpu().numpy().astype(np.float32)
        if result.ndim != 2 or result.shape != (self.action_horizon, self.action_dim):
            raise RuntimeError(f"Invalid action shape: {result.shape}")
        return result


def load_baseline(method, checkpoint, device="cpu"):
    payload = _checkpoint(checkpoint, device)
    if method == "bc":
        from baselines.bc.bc import Actor
        weights = payload["actor"]
        obs_dim = weights["net.0.weight"].shape[1]
        act_dim = weights["net.4.weight"].shape[0]
        model = Actor(obs_dim, act_dim).to(device)
        model.load_state_dict(weights)
        stats = {"observation_mean": payload.get("observation_mean", 0.0),
                 "observation_std": payload.get("observation_std", 1.0)}
        return BaselinePolicy(model, method, device, stats=stats)

    if method == "act":
        from baselines.act.train import Agent, Args
        cfg = payload.get("candid_config")
        if cfg is None:
            raise ValueError("ACT checkpoint missing candid_config; retrain with CANDID ACT")
        params = Args(**cfg)
        weights = payload["ema_agent"]
        obs_dim = weights["model.input_proj_robot_state.weight"].shape[1]
        act_dim = weights["model.action_head.weight"].shape[0]
        model = Agent(_dummy_env(obs_dim, act_dim), params).to(device)
        model.load_state_dict(weights)
        return BaselinePolicy(model, method, device, stats=payload["norm_stats"],
                              queries=params.num_queries, temporal_agg=params.temporal_agg)

    if method == "ppo":
        from baselines.ppo.ppo import Agent
        weights = payload.get("agent", payload)
        obs_dim = weights["actor_mean.0.weight"].shape[1]
        act_dim = weights["actor_mean.6.weight"].shape[0]
        model = Agent(_dummy_env(obs_dim, act_dim)).to(device)
        model.load_state_dict(weights)
        return BaselinePolicy(model, method, device)

    if method == "sac":
        from baselines.sac.sac import Actor
        weights = payload["actor"]
        obs_dim = weights["backbone.0.weight"].shape[1]
        act_dim = weights["fc_mean.weight"].shape[0]
        model = Actor(_dummy_env(obs_dim, act_dim)).to(device)
        model.load_state_dict(weights)
        return BaselinePolicy(model, method, device)
    raise ValueError(f"Unknown method: {method}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("act", "bc", "ppo", "sac"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--safe-dr-mean-max", type=float, required=True)
    parser.add_argument("--safe-dr-max-max", type=float, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--episodes-per-scene", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--exec-horizon", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-stop-on-success", action="store_true")
    args = parser.parse_args(argv)
    policy = load_baseline(args.method, args.checkpoint, args.device)
    if policy.action_dim != 7:
        raise ValueError("CANDID requires 7-dimensional pd_joint_pos actions")
    if args.exec_horizon is None:
        exec_horizon = 1 if args.method != "act" or policy.temporal_agg else policy.action_horizon
    else:
        exec_horizon = args.exec_horizon
    result = evaluate_dataset(
        args.dataset, policy, args.output_dir,
        SafetyLimits(args.safe_dr_mean_max, dr_max_max=args.safe_dr_max_max),
        checkpoint=args.checkpoint, max_scenes=args.max_scenes,
        episodes_per_scene=args.episodes_per_scene, max_steps=args.max_steps,
        exec_horizon=exec_horizon, eval_seed=args.eval_seed,
        save_video=args.save_video, stop_on_success=not args.no_stop_on_success,
    )
    print(f"{args.method.upper()} evaluation: {result['success_rate']:.4f} success, "
          f"{result['safe_success_rate']:.4f} safe_success")
    return result


if __name__ == "__main__":
    main()
