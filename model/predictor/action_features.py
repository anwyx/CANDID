"""Scene1 Predictor v3 features; never alter DP or environment actions.

The observation vector uses candid.data.adapter. 151 = 12 qpos + 12 qvel
+ 7 TCP pose + 16 target + 6*16 distractors + 6 masks + 2 goal.
The first six qpos values are the robot arm joint positions.
"""
from __future__ import annotations

import numpy as np
import torch

SCENE1_OBS_DIM = 151
ARM_DOF = 6
TARGET_OFFSET = 12 + 12 + 7
DISTRACTOR_OFFSET = TARGET_OFFSET + 16


def absolute_to_delta_q(obs, actions):
    """Convert absolute pd_joint_pos commands to Delta-q for the arm only.

    Supports NumPy and torch; in torch the transform is differentiable with
    respect to the original absolute action tensor. Never modify the gripper.
    obs: [...,2,151], actions: [...,8,7], shared leading dimensions.
    """
    if obs.shape[-2:] != (2, SCENE1_OBS_DIM) or actions.shape[-2:] != (8, 7):
        raise ValueError("Scene1 v3 requires obs [...,2,151] and actions [...,8,7]")
    if obs.shape[:-2] != actions.shape[:-2]:
        raise ValueError("Observation/action batch dimensions disagree")
    current = obs[..., -1, :ARM_DOF]
    if isinstance(actions, torch.Tensor):
        if not isinstance(obs, torch.Tensor):
            raise TypeError("Torch actions require torch observations")
        return torch.cat(
            (actions[..., :ARM_DOF] - current.unsqueeze(-2),
             actions[..., ARM_DOF:]), dim=-1)
    if not isinstance(actions, np.ndarray) or not isinstance(obs, np.ndarray):
        raise TypeError("Expected matching NumPy arrays or torch tensors")
    result = actions.copy()
    result[..., :ARM_DOF] -= current[..., None, :]
    return result


def action_observation_mask(obs_dim=SCENE1_OBS_DIM):
    """Boolean mask: zero only object linear/angular velocities in action branch.

    The full state branch keeps all velocities. Do not mask arm qpos/TCP.
    """
    if obs_dim != SCENE1_OBS_DIM:
        raise ValueError("Velocity layout verified only for scene1 obs_dim=151")
    mask = np.ones((obs_dim,), dtype=np.float32)
    mask[TARGET_OFFSET + 7:TARGET_OFFSET + 13] = 0
    for obj in range(6):
        start = DISTRACTOR_OFFSET + 16 * obj
        mask[start + 7:start + 13] = 0
    assert int((mask == 0).sum()) == 42
    return mask
