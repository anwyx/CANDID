"""Diffusion Policy agent.

Algorithm source: ManiSkill commit 62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3,
class ``Agent`` in ``examples/baselines/diffusion_policy/train.py``.

``compute_loss`` adds scheduler noise to the action chunk, predicts epsilon
with ``ConditionalUnet1D``, and returns MSE against that noise. ``get_action``
denoises a ``pred_horizon`` chunk and returns the ``act_horizon`` actions that
start at index ``obs_horizon - 1``.

Deviations from that file
-------------------------
1. CANDID actions are absolute ``pd_joint_pos`` values, not an action space
   that is already limited to [-1, 1]. The training loop maps each dimension
   from its training minimum and maximum onto [-1, 1] before ``add_noise``.
   ``DDPMScheduler`` still clips samples to [-1, 1]. ``VanillaDPPolicy.act``
   maps the executed slice back to joint targets. Observations are not
   normalized, matching the upstream state trainer.
2. The agent is built from observation and action sizes. Upstream reads those
   sizes from a ManiSkill environment and asserts the action space is [-1, 1].
3. Noise and timestep tensors are created on the batch device. Upstream uses
   one global ``device`` variable captured by the training script.
4. ``set_timesteps(num_diffusion_iters)`` is called once at construction so
   inference uses the 100-step training schedule. Upstream skips that call
   when the installed scheduler already fills ``timesteps`` the same way.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from model.vanilla_dp.conditional_unet1d import ConditionalUnet1D
from model.vanilla_dp.normalizer import ActionRangeNormalizer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(action_dim)
        self.obs_horizon = config.obs_horizon
        self.act_horizon = config.act_horizon
        self.pred_horizon = config.pred_horizon
        self.num_diffusion_iters = config.num_diffusion_iters
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=self.obs_horizon * self.obs_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=list(config.unet_dims),
            n_groups=config.n_groups,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

    def compute_loss(self, obs_seq, action_seq):
        batch = obs_seq.shape[0]
        obs_cond = obs_seq.flatten(start_dim=1)
        noise = torch.randn(
            (batch, self.pred_horizon, self.act_dim), device=action_seq.device
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch,),
            device=action_seq.device,
        ).long()
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        batch = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)
            noisy_action_seq = torch.randn(
                (batch, self.pred_horizon, self.act_dim), device=obs_seq.device
            )
            for step in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=step,
                    global_cond=obs_cond,
                )
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=step,
                    sample=noisy_action_seq,
                ).prev_sample
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]


class VanillaDPPolicy:
    """Thin closed-loop adapter around ``Agent.get_action``."""

    def __init__(self, agent: Agent, normalizer: ActionRangeNormalizer):
        self.agent = agent
        self.normalizer = normalizer
        self.obs_horizon = agent.obs_horizon
        self.obs_dim = agent.obs_dim
        self.action_dim = agent.act_dim
        self.action_horizon = agent.act_horizon

    def begin_episode(self, seed: int) -> None:
        torch.manual_seed(int(seed))

    def act(self, observation_window: np.ndarray) -> np.ndarray:
        window = np.asarray(observation_window, dtype=np.float32)
        if window.shape != (self.obs_horizon, self.obs_dim):
            raise ValueError(
                f"Expected observation window {(self.obs_horizon, self.obs_dim)}, got {window.shape}"
            )
        device = next(self.agent.parameters()).device
        observations = torch.from_numpy(window).unsqueeze(0).to(device)
        normalized = self.agent.get_action(observations)
        command = self.normalizer.denormalize(normalized)
        return command[0].detach().cpu().numpy().astype(np.float32)
