"""Check multi-author dataset compatibility before any model training."""
from __future__ import annotations
import json
from pathlib import Path
import h5py
import numpy as np


def _rows(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _first_dataset(group):
    if isinstance(group,h5py.Dataset):return group
    for entry in group.values():
        got=_first_dataset(entry)
        if got is not None:return got
    return None


def validate_dataset(root):
    root=Path(root)
    scenes=_rows(root/"scenes.jsonl")
    trajectories=_rows(root/"trajectories.jsonl")
    scene_ids=[x["scene_id"] for x in scenes]
    traj_ids=[x["trajectory_id"] for x in trajectories]
    if len(set(scene_ids))!=len(scene_ids):raise ValueError("Duplicate scene_id")
    if len(set(traj_ids))!=len(traj_ids):raise ValueError("Duplicate trajectory_id")
    catalog=set(scene_ids)
    h5_handles={}
    validated=0
    try:
        for scene in scenes:
            with np.load(root/scene["initial_state"],allow_pickle=False) as data:
                if not len(data.files):raise ValueError("Empty initial simulator snapshot")
        for row in trajectories:
            if row["scene_id"] not in catalog:raise ValueError(f"Unknown scene: {row['scene_id']}")
            with np.load(root/row["disturbance_trace"],allow_pickle=False) as data:
                T=int(row["num_steps"])
                poses=data["distractor_poses"]
                linear=data["distractor_linear_velocities"]
                angular=data["distractor_angular_velocities"]
                sizes=data["object_sizes_m"]
                n=len(sizes)
                if poses.shape!=(T+1,n,7) or linear.shape!=(T+1,n,3) or angular.shape!=(T+1,n,3):
                    raise ValueError(f"Misaligned trace: {row['trajectory_id']}")
                for key in ("distractor_toppled","distractor_fallen"):
                    if key in data and (data[key].shape!=(T+1,n) or data[key].dtype.kind!="b"):
                        raise ValueError(f"Misaligned event flags: {row['trajectory_id']}:{key}")
                if n!=len(row["object_ids"]):raise ValueError("Object order/count mismatch")
                if not np.isfinite(poses).all() or not np.isfinite(linear).all() or not np.isfinite(angular).all():
                    raise ValueError("Non-finite physics measurements")
                if not np.isfinite(sizes).all() or not (sizes>0).all():
                    raise ValueError("Bad object characteristic sizes")
            if row.get("saved_hdf5"):
                filename=row.get("hdf5_file")
                key=row.get("hdf5_trajectory_key")
                if not filename or not key:raise ValueError("Missing HDF5 file/key")
                if filename not in h5_handles:
                    h5_handles[filename]=h5py.File(root/filename,"r")
                f=h5_handles[filename]
                if key not in f:raise ValueError(f"Missing HDF5 group {filename}:{key}")
                g=f[key]
                if len(g["actions"])!=T:raise ValueError(f"HDF5 actions/trace mismatch: {key}")
                if "obs" in g:
                    dataset=_first_dataset(g["obs"])
                    if dataset is not None and len(dataset)!=T+1:
                        raise ValueError(f"HDF5 obs/trace mismatch: {key}")
            validated+=1
    finally:
        for f in h5_handles.values(): f.close()
    return {"num_scenes":len(scenes),"num_trajectories":validated,
            "num_recorded_hdf5":sum(bool(x.get("saved_hdf5")) for x in trajectories)}
