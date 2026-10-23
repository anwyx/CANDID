"""No user-written env logging code needed: inspect standard actor registry.

Only physics truth is recorded in a separate trace, not fed to the planner or
observation. Scene creation/settling is complete before reset returns here.
"""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from candid.data.metrics import disturbance_from_poses, event_flags_from_poses


def as_numpy(value):
    if hasattr(value, "detach"):
        value=value.detach().cpu().numpy()
    return np.asarray(value)


class PhysicsTrace(gym.Wrapper):
    def __init__(self, env, w_translation=0.7, w_rotation=0.3):
        super().__init__(env)
        self.w_translation=float(w_translation)
        self.w_rotation=float(w_rotation)
        self.reset_count=0
        self.steps=0
        self.pose_trace=[]
        self.linear_trace=[]
        self.angular_trace=[]
        self.names=[]
        self.sizes=[]
        self.kinds=[]

    def _actors(self):
        base=self.env.unwrapped
        if hasattr(base,"distractor_actors"):
            return list(base.distractor_actors)
        if hasattr(base,"clutter_cubes"):
            return list(base.clutter_cubes)
        if hasattr(base,"clutter_objects"):
            return list(base.clutter_objects)
        raise RuntimeError("Environment has no distractor actor registry")

    def _capture(self):
        actors=self._actors()
        poses=np.stack([as_numpy(a.pose.raw_pose).reshape(-1,7)[0] for a in actors],axis=0)
        linear=np.stack([as_numpy(a.get_linear_velocity()).reshape(-1,3)[0]
                         for a in actors],axis=0)
        angular=np.stack([as_numpy(a.get_angular_velocity()).reshape(-1,3)[0]
                          for a in actors],axis=0)
        self.pose_trace.append(poses.copy())
        self.linear_trace.append(linear.copy())
        self.angular_trace.append(angular.copy())

    def reset(self,*args,**kwargs):
        obs,info=self.env.reset(*args,**kwargs)
        self.reset_count+=1
        self.steps=0
        self.pose_trace=[]
        self.linear_trace=[]
        self.angular_trace=[]
        base=self.env.unwrapped
        if hasattr(base,"distractor_metadata"):
            self.names=[m.spec.name for m in base.distractor_metadata]
            self.sizes=[m.characteristic_size for m in base.distractor_metadata]
            self.kinds=[m.spec.kind for m in base.distractor_metadata]
        else:
            actors=self._actors()
            self.names=[getattr(a,"name",f"obj_{i}") for i,a in enumerate(actors)]
            self.kinds=["unknown"]*len(actors)
            if hasattr(base,"get_distractor_sizes_for_logging"):
                self.sizes=list(as_numpy(base.get_distractor_sizes_for_logging()).reshape(-1))
            elif hasattr(base,"DISTRACTOR_HALF_SIZE"):
                self.sizes=[2*base.DISTRACTOR_HALF_SIZE]*len(actors)
            else:
                raise RuntimeError("Unknown characteristic sizes: add shared actor metadata")
        self._capture()
        return obs,info

    def step(self,action):
        result=self.env.step(action)
        self.steps+=1
        self._capture()
        return result

    def arrays(self):
        poses=np.asarray(self.pose_trace,dtype=np.float32)
        linear=np.asarray(self.linear_trace,dtype=np.float32)
        angular=np.asarray(self.angular_trace,dtype=np.float32)
        if poses.shape!=(self.steps+1,len(self.sizes),7):
            raise RuntimeError(f"Pose/action timeline mismatch: {poses.shape} vs {self.steps}")
        if linear.shape!=(self.steps+1,len(self.sizes),3) or angular.shape!=linear.shape:
            raise RuntimeError("Velocity/action timeline mismatch")
        return {"distractor_poses":poses,
                "distractor_linear_velocities":linear,
                "distractor_angular_velocities":angular,
                "object_sizes_m":np.asarray(self.sizes,dtype=np.float32),
                **event_flags_from_poses(poses,kinds=self.kinds)}

    def summary(self):
        data=self.arrays()
        return disturbance_from_poses(data["distractor_poses"],self.sizes,
                                      w_translation=self.w_translation,
                                      w_rotation=self.w_rotation,
                                      topple_flags=data["distractor_toppled"],
                                      fall_flags=data["distractor_fallen"])
