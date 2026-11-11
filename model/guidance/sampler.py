"""Risk-gradient correction inside the existing Vanilla DP DDPM loop.

The denoiser and ``DDPMScheduler.step`` are the checkpoint's own objects.
This module does not replace them.

For a guided timestep the order is:

1. Predict epsilon with the frozen denoiser.
2. Estimate the clean action ``x0`` with the scheduler's epsilon formula.
   This estimate is an intermediate sample, not the final action chunk.
3. Score only the executable window with the predictor and differentiate
   that scalar risk with respect to the normalized window.
4. Call ``scheduler.step`` once. That call is what consumes the DDPM noise.
5. Subtract ``strength * gradient`` from the resulting noisy sample.

The correction therefore lands on the next noisy sample, after the DDPM
update. It does not overwrite ``x0``, and a per-step before/after pair is
not a second full denoising chain.
"""
from __future__ import annotations

import torch

from model.guidance.objective import chunk_risk


class ConsequenceGuidedSampler:
    def __init__(self, agent, action_normalizer, predictor, predictor_normalizer, weights):
        self.agent = agent
        self.action_normalizer = action_normalizer
        self.predictor = predictor
        self.predictor_normalizer = predictor_normalizer
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.obs_horizon = agent.obs_horizon
        self.act_horizon = agent.act_horizon
        self.window_start = self.obs_horizon - 1
        self.window_end = self.window_start + self.act_horizon

    def sample(self, obs_seq, *, strength, start_timestep, trace=False, obs_normalizer=None):
        """Return the normalized executable window ``[1, act_horizon, action_dim]``.

        ``obs_seq`` is the raw observation window. ``obs_normalizer``, when
        given, is applied only before the denoiser.
        """
        if strength < 0:
            raise ValueError("strength must be nonnegative")
        if obs_seq.ndim != 3 or obs_seq.shape[1] != self.obs_horizon:
            raise ValueError(
                f"Expected obs [B,{self.obs_horizon},D], got {tuple(obs_seq.shape)}"
            )
        denoiser_obs = obs_seq if obs_normalizer is None else obs_normalizer(obs_seq)
        obs_cond = denoiser_obs.flatten(start_dim=1)
        scheduler = self.agent.noise_scheduler
        noisy = torch.randn(
            (obs_seq.shape[0], self.agent.pred_horizon, self.agent.act_dim),
            device=obs_seq.device,
            dtype=obs_seq.dtype,
        )
        rows = []
        for step in scheduler.timesteps:
            timestep = int(step.item()) if hasattr(step, "item") else int(step)
            guided = float(strength) > 0 and timestep <= int(start_timestep)
            with torch.no_grad():
                noise_pred = self.agent.noise_pred_net(
                    sample=noisy,
                    timestep=step,
                    global_cond=obs_cond,
                )
                updated = scheduler.step(
                    model_output=noise_pred,
                    timestep=step,
                    sample=noisy,
                )
            correction = torch.zeros_like(updated.prev_sample)
            record = {
                "diffusion_timestep": timestep,
                "guided": guided,
                "intermediate_x0": True,
                "pred_dr_mean": None,
                "pred_dr_max": None,
                "pred_topple_score": None,
                "pred_fall_score": None,
                "pred_weighted_risk": None,
                "raw_gradient_l2": 0.0,
                "applied_correction_l2": 0.0,
                "max_abs_correction": 0.0,
            }
            window_correction = None
            x0_window = None
            if guided:
                window_correction, x0_window, record = self._guide_window(
                    obs_seq, updated.pred_original_sample, record, strength
                )
                correction[:, self.window_start:self.window_end] = window_correction
            noisy = (updated.prev_sample - correction).detach()
            if trace:
                record["x0_action_chunk"] = (
                    None if x0_window is None else x0_window.detach().cpu().numpy()
                )
                record["correction"] = (
                    None if window_correction is None else window_correction.detach().cpu().numpy()
                )
                rows.append(record)
        if noisy.shape[1] < self.window_end:
            raise RuntimeError("Predicted horizon is shorter than the executable window")
        return noisy[:, self.window_start:self.window_end], rows

    def _guide_window(self, obs_seq, x0_estimate, record, strength):
        """Differentiate risk w.r.t. the normalized executable window of ``x0``."""
        window = x0_estimate[:, self.window_start:self.window_end].detach().requires_grad_(True)
        physical = self.action_normalizer.denormalize(window)
        weights = self.weights.to(device=physical.device, dtype=physical.dtype)
        risk, predictions = chunk_risk(
            self.predictor,
            self.predictor_normalizer,
            obs_seq,
            physical,
            weights,
        )
        gradient = torch.autograd.grad(risk.sum(), window)[0]
        correction = float(strength) * gradient
        record.update({
            "pred_dr_mean": float(predictions["dr_mean"][0].detach()),
            "pred_dr_max": float(predictions["dr_max"][0].detach()),
            "pred_topple_score": float(predictions["topple_score"][0].detach()),
            "pred_fall_score": float(predictions["fall_score"][0].detach()),
            "pred_weighted_risk": float(risk[0].detach()),
            "raw_gradient_l2": float(torch.linalg.vector_norm(gradient).detach()),
            "applied_correction_l2": float(torch.linalg.vector_norm(correction).detach()),
            "max_abs_correction": float(correction.detach().abs().max()),
        })
        return correction, window.detach(), record
