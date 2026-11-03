"""Optional temporal losses for CANDID's unchanged diffusion-policy network.

The usual DDPM noise-prediction loss remains the main training objective.
For an expert trajectory, sample two windows four steps apart. The diffusion
noise and timestep are matched on the four overlapping executed actions, so
consistency measures different observation contexts, not different noise draws.
Only low-noise timesteps are used for x0 reconstruction regularizers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_temporal_losses(
    agent,
    obs_a: torch.Tensor,
    actions_a: torch.Tensor,
    obs_b: torch.Tensor,
    actions_b: torch.Tensor,
    *,
    stride: int,
    max_timestep: int,
) -> dict[str, torch.Tensor]:
    """Return unweighted {noise, delta, overlap} losses.

    Expected actions are normalized [-1,1] absolute pd_joint_pos targets.
    The first six dimensions are arm joints; dimension seven is the gripper.
    Alignment is computed in executed-action coordinates: A[stride:8] and
    B[:8-stride], where each 8-step slice starts at obs_horizon-1.
    """
    if actions_a.shape != actions_b.shape or obs_a.shape != obs_b.shape:
        raise ValueError("Paired observation/action batches must match shape")
    if actions_a.ndim != 3 or actions_a.shape[-1] != 7:
        raise ValueError("CANDID expects [B, pred_horizon, 7] actions")
    if obs_a.shape[0] != actions_a.shape[0]:
        raise ValueError("Observation and action batch size mismatch")
    if not 0 < stride < agent.act_horizon:
        raise ValueError("Require 0 < stride < act_horizon")
    start = agent.obs_horizon - 1
    end = start + agent.act_horizon
    overlap = agent.act_horizon - stride
    if end > agent.pred_horizon:
        raise ValueError("Executed action slice exceeds pred_horizon")

    # Verify that the selected windows refer to the same expert timestamps.
    a_shared = actions_a[:, start + stride:end]
    b_shared = actions_b[:, start:start + overlap]
    if not torch.allclose(a_shared, b_shared, rtol=0, atol=1e-6):
        raise ValueError("Paired expert actions are not aligned in time")

    batch = actions_a.shape[0]
    actions = torch.cat((actions_a, actions_b), dim=0)
    observations = torch.cat((obs_a, obs_b), dim=0)
    timesteps_one = torch.randint(
        0, agent.num_diffusion_iters, (batch,), device=actions.device
    ).long()
    timesteps = torch.cat((timesteps_one, timesteps_one), dim=0)
    noise = torch.randn_like(actions)
    # Same true action + same noise + same timestep on overlapping positions.
    noise[batch:, start:start + overlap] = noise[:batch, start + stride:end]

    noisy_actions = agent.noise_scheduler.add_noise(actions, noise, timesteps)
    eps_pred = agent.noise_pred_net(
        noisy_actions, timesteps, global_cond=observations.flatten(start_dim=1)
    )
    noise_loss = F.mse_loss(eps_pred, noise)
    zero = noise_loss * 0.0

    selected = timesteps_one <= int(max_timestep)
    if not bool(selected.any()):
        return {"noise": noise_loss, "delta": zero, "overlap": zero}

    # Reconstruct x0 only at low-noise steps. At t ~ 99 (cosine DDPM),
    # dividing by sqrt(alpha_bar) would amplify prediction error massively.
    selected_t = timesteps_one[selected]
    indexes = torch.nonzero(selected, as_tuple=False).flatten()
    indexes = torch.cat((indexes, indexes + batch), dim=0)
    alpha_bar = agent.noise_scheduler.alphas_cumprod.to(
        device=actions.device, dtype=actions.dtype
    )[selected_t]
    alpha_bar = torch.cat((alpha_bar, alpha_bar), dim=0).reshape(-1, 1, 1)
    x0_pred = (
        noisy_actions[indexes] - torch.sqrt(1.0 - alpha_bar) * eps_pred[indexes]
    ) / torch.sqrt(alpha_bar)
    x0_true = actions[indexes]

    # Match the expert arm joint deltas; do not penalize intended motion
    # or smear the discrete open/close gripper transition.
    pred_delta = x0_pred[:, 1:, :6] - x0_pred[:, :-1, :6]
    true_delta = x0_true[:, 1:, :6] - x0_true[:, :-1, :6]
    delta_loss = F.smooth_l1_loss(pred_delta, true_delta)

    n = selected_t.numel()
    shared_a = x0_pred[:n, start + stride:end, :6]
    shared_b = x0_pred[n:, start:start + overlap, :6]
    overlap_loss = F.smooth_l1_loss(shared_a, shared_b)
    return {"noise": noise_loss, "delta": delta_loss, "overlap": overlap_loss}
