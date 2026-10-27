"""Canonical State-based observation interface for both DP and Predictor.

ManiSkill's `state` mode flattens keys into an opaque vector. We intentionally
record `state_dict` so this adapter can read stable named inputs across scenes.
"""
from __future__ import annotations
import numpy as np


def canonical_observation(raw_obs_group, num_distractors, max_distractors=6):
    """Read one RecordEpisode HDF5 `obs` group into named arrays [T+1,...].

    Do not include raw pose-trace labels or future states in model input.
    Both models call this adapter with exactly the same inputs.
    """
    if not 0<=num_distractors<=max_distractors:
        raise ValueError("Invalid number of distractors")
    agent=raw_obs_group["agent"]
    extra=raw_obs_group["extra"]
    get=lambda g,k: np.asarray(g[k],dtype=np.float32)
    robot_qpos=get(agent,"qpos")
    robot_qvel=get(agent,"qvel")
    target_pose=get(extra,"target_pose")
    target_linear=get(extra,"target_linear_velocity")
    target_angular=get(extra,"target_angular_velocity")
    target_dims=get(extra,"target_dimensions")
    N=num_distractors
    d_pose=get(extra,"distractor_poses").reshape(-1,N,7)
    d_linear=get(extra,"distractor_linear_velocities").reshape(-1,N,3)
    d_angular=get(extra,"distractor_angular_velocities").reshape(-1,N,3)
    d_dims=get(extra,"distractor_dimensions").reshape(-1,N,3)
    target=np.concatenate([target_pose,target_linear,target_angular,target_dims],axis=-1)
    d_state=np.concatenate([d_pose,d_linear,d_angular,d_dims],axis=-1)
    T1=len(robot_qpos)
    if any(len(x)!=T1 for x in (robot_qvel,target,d_state)):
        raise ValueError("Observation arrays are not time-aligned")
    padded=np.zeros((T1,max_distractors,16),dtype=np.float32)
    padded[:,:N]=d_state
    mask=np.zeros((T1,max_distractors),dtype=np.float32)
    mask[:,:N]=1
    return {
        "robot_qpos":robot_qpos,
        "robot_qvel":robot_qvel,
        "tcp_pose":get(extra,"tcp_pose"),
        "target_state":target,                      # [T+1,16]
        "distractor_states":padded,                  # [T+1,6,16]
        "distractor_mask":mask,                     # [T+1,6]
        "goal_xy":get(extra,"goal_pos")[:,:2],     # [T+1,2]
    }


OBSERVATION_FIELDS = (
    "robot_qpos",
    "robot_qvel",
    "tcp_pose",
    "target_state",
    "distractor_states",
    "distractor_mask",
    "goal_xy",
)


def observation_vector(obs_window: dict) -> np.ndarray:
    """Stack named state fields into ``[time, features]``.

    ``obs_window`` is either a full trajectory or one model window. Every
    field must share the same time length. Future disturbance labels are not
    part of this vector.
    """
    parts = []
    horizon = None
    for key in OBSERVATION_FIELDS:
        array = np.asarray(obs_window[key], dtype=np.float32)
        if horizon is None:
            horizon = array.shape[0]
        if array.shape[0] != horizon:
            raise ValueError(f"Observation field {key} has a different horizon")
        parts.append(array.reshape(horizon, -1))
    return np.concatenate(parts, axis=-1)
