"""Shared sliding-window rules for Vanilla DP and Predictor datasets.

This module contains NO physical simulator dependencies. It accepts already
loaded arrays, with per-time observation dictionary arrays aligned to actions.
"""
from __future__ import annotations
import numpy as np
from candid.data.metrics import disturbance_from_poses


def windows(num_actions, observation_horizon=2, action_horizon=12):
    if observation_horizon<1 or action_horizon<1:
        raise ValueError("Both horizons must be positive")
    # A sample at time t uses states [t-history+1, ..., t] and actions
    # [t, ..., t+H-1], whose physical consequences end at state t+H.
    for t in range(observation_horizon-1,num_actions-action_horizon+1):
        yield t


def prepare_window(observations, actions, trace, t, observation_horizon=2,
                   action_horizon=12, translation_weight=0.7,rotation_weight=0.3):
    """Returns inputs shared by DP/Predictor, plus predictor-only targets.

    `observations`: dict of [T+1,*] arrays already normalised/flattened by a
    separate observation adapter. Do not put future ground truth in input.
    """
    actions=np.asarray(actions)
    states=np.asarray(trace["distractor_poses"])
    sizes=np.asarray(trace["object_sizes_m"])
    T=len(actions)
    if states.shape[0]!=T+1:
        raise ValueError("Actions and physics trace are not T / T+1 aligned")
    if not (observation_horizon-1<=t and t+action_horizon<=T):
        raise ValueError("Window extends outside available sequence")
    start=t-observation_horizon+1
    obs={}
    for k,val in observations.items():
        arr=np.asarray(val)
        if len(arr)!=T+1:
            raise ValueError(f"Observation {k} must have T+1 states")
        obs[k]=arr[start:t+1]
    action_chunk=actions[t:t+action_horizon]
    labels=disturbance_from_poses(states,sizes,t,t+action_horizon,
                                 w_translation=translation_weight,
                                 w_rotation=rotation_weight)
    return {"obs":obs,"actions":action_chunk,"predictor_labels":labels}
