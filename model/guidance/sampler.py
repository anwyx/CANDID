"""GRITS-style inference guidance for the existing epsilon/DDPM Vanilla DP.

The denoiser stays under no_grad; gradients flow only from the frozen
predictor through the estimated clean action A_{0|k} back to A_k. The
predicted epsilon is treated as constant for this first-order approximation.
This avoids backpropagation through the UNet.
"""
from __future__ import annotations

import torch

from model.guidance.objective import chunk_risk
from model.guidance.timing import PhaseTimer


class ConsequenceGuidedSampler:
    def __init__(self, dp_policy, predictor, predictor_normalizer, config):
        self.dp_policy = dp_policy
        self.agent = dp_policy.agent
        self.dp_normalizer = dp_policy.normalizer
        self.predictor = predictor
        self.predictor_normalizer = predictor_normalizer
        self.config = config
        self.start = self.agent.obs_horizon - 1
        self.stop = self.start + self.agent.act_horizon
        if self.stop > self.agent.pred_horizon:
            raise ValueError("DP execution window exceeds its prediction horizon")
        if (self.agent.obs_horizon, self.agent.obs_dim) != (predictor.obs_horizon, predictor.obs_dim):
            raise ValueError("DP/Predictor observation dimensions or horizons disagree")
        if (self.agent.act_horizon, self.agent.act_dim) != (predictor.action_horizon, predictor.action_dim):
            raise ValueError("DP/Predictor action dimensions or horizons disagree")
        if self.config.start_timestep >= self.agent.num_diffusion_iters:
            raise ValueError("start_timestep must be below the DP diffusion step count")
        if self.agent.noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError("Guidance requires an epsilon-prediction DDPM scheduler")
        self.agent.eval()
        self.predictor.eval()
        self.agent.requires_grad_(False)
        self.predictor.requires_grad_(False)

    @staticmethod
    def _limit_l2(values, limit):
        norm = values.flatten(start_dim=1).norm(dim=1).clamp_min(1e-12)
        factor = (limit / norm).clamp(max=1.0)
        return values * factor.view(-1, *([1] * (values.ndim - 1)))

    def sample(self, obs, bounds=None, *, profile_timing=False):
        """Return normalized executed actions and JSON-safe diagnostics.

        Bounds is a tuple of low/high tensors in environment action units.
        The risk model scores the same clipped commands the rollout executes.
        CUDA profiling measures kernels by events, without per-step GPU sync.
        """
        agent = self.agent
        if obs.ndim != 3 or obs.shape[1:] != (agent.obs_horizon, agent.obs_dim):
            raise ValueError("Invalid DP observation window")
        if not torch.isfinite(obs).all():
            raise ValueError("Observation window contains NaN/Inf")
        if self.config.strength > 0 and bounds is None:
            raise ValueError("Active guidance requires actual environment action bounds")
        if bounds is not None:
            low, high = bounds
            low = low.to(device=obs.device, dtype=obs.dtype)
            high = high.to(device=obs.device, dtype=obs.dtype)
            if low.shape != (agent.act_dim,) or high.shape != (agent.act_dim,):
                raise ValueError("Environment action bounds have wrong dimension")
            if torch.isnan(low).any() or torch.isnan(high).any() or (low > high).any():
                raise ValueError("Invalid environment action bounds")
        else:
            low = high = None

        timer = PhaseTimer(obs.device, enabled=profile_timing)
        obs_cond = obs.flatten(start_dim=1)
        noisy = torch.randn(
            (len(obs), agent.pred_horizon, agent.act_dim),
            device=obs.device, dtype=obs.dtype,
        )
        guided_steps = 0
        # Accumulate on-device; a .item() inside every guided step would
        # synchronize CUDA tens of thousands of times in a scene1 evaluation.
        largest_gradient = torch.zeros((), device=obs.device, dtype=obs.dtype)
        largest_update = torch.zeros((), device=obs.device, dtype=obs.dtype)

        # Both scheduler noise draws and the unconditional strength=0 path
        # match the original Vanilla DP exactly.
        for timestep in agent.noise_scheduler.timesteps:
            stamp = timer.start()
            with torch.no_grad():
                epsilon = agent.noise_pred_net(
                    sample=noisy, timestep=timestep, global_cond=obs_cond
                )
            timer.stop("denoiser", stamp)

            t = int(timestep)
            correction = None
            if self.config.strength > 0 and t <= self.config.start_timestep:
                stamp = timer.start()
                with torch.enable_grad():
                    variable = noisy.detach().requires_grad_(True)
                    alpha_bar = agent.noise_scheduler.alphas_cumprod[t].to(
                        device=variable.device, dtype=variable.dtype
                    )
                    x0 = (
                        variable - (1.0 - alpha_bar).sqrt() * epsilon.detach()
                    ) / alpha_bar.sqrt()
                    if agent.noise_scheduler.config.clip_sample:
                        clip_range = float(getattr(
                            agent.noise_scheduler.config, "clip_sample_range", 1.0
                        ))
                        x0 = x0.clamp(-clip_range, clip_range)
                    candidate = self.dp_normalizer.denormalize(
                        x0[:, self.start:self.stop]
                    )
                    candidate = torch.maximum(torch.minimum(candidate, high), low)
                    risk = chunk_risk(
                        obs, candidate, self.predictor,
                        self.predictor_normalizer, self.config,
                    )
                    gradient = torch.autograd.grad(
                        risk.sum(), variable, allow_unused=True
                    )[0]
                if gradient is None:
                    raise RuntimeError("Risk is not differentiable with respect to the action chunk")
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Guidance gradient contains NaN/Inf")
                mask = torch.zeros_like(gradient)
                mask[:, self.start:self.stop] = 1
                gradient = gradient * mask
                largest_gradient = torch.maximum(
                    largest_gradient,
                    gradient.flatten(start_dim=1).norm(dim=1).max(),
                )
                gradient = self._limit_l2(gradient, self.config.max_grad_norm)
                correction = self._limit_l2(
                    self.config.strength * gradient, self.config.max_step_norm
                )
                largest_update = torch.maximum(
                    largest_update,
                    correction.flatten(start_dim=1).norm(dim=1).max(),
                )
                guided_steps += 1
                timer.stop("predictor_gradient", stamp)

            stamp = timer.start()
            with torch.no_grad():
                noisy = agent.noise_scheduler.step(
                    model_output=epsilon, timestep=timestep, sample=noisy
                ).prev_sample
                if correction is not None:
                    noisy = noisy - correction
            timer.stop("ddpm_update", stamp)

        result = noisy[:, self.start:self.stop]
        if not torch.isfinite(result).all():
            raise FloatingPointError("Guided diffusion generated NaN/Inf actions")
        timing = timer.finish()
        stats = {
            "guided_steps": guided_steps,
            "max_raw_grad_norm": float(largest_gradient.item()),
            "max_guidance_update_norm": float(largest_update.item()),
        }
        if timing is not None:
            stats["timing_ms"] = timing
        return result, stats
