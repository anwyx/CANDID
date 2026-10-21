"""One writer for Scene snapshots, Trajectory truth, and JSONL indices.

All paths in manifests are relative to output root for easy cloud sync.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(x) for x in value]
    if hasattr(value,"detach"):return json_safe(value.detach().cpu().numpy())
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    raise TypeError(f"Cannot serialize {type(value)}")


def freeze_state(value):
    """Clone torch tensors / numpy arrays to preserve initial scene before rollout."""
    if isinstance(value,dict):return {k:freeze_state(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(freeze_state(v) for v in value)
    if hasattr(value,"detach"):return value.detach().clone()
    if isinstance(value,np.ndarray):return value.copy()
    return value


def save_state_npz(path, state):
    """No pickle: flattened dict of numeric arrays in NPZ. Use only with
    a matching ManiSkill version for scene replay. Runner keeps live
    snapshot in memory; this archive is for future reload/reproducibility.
    """
    arrays={}
    def visit(node, trail):
        if isinstance(node,dict):
            for k,v in node.items(): visit(v,trail+[str(k)])
        else:
            val=np.asarray(json_safe(node))
            if val.dtype.kind not in "ifub":
                raise TypeError(f"Non-numeric saved simulator state: {trail}")
            arrays[json.dumps(trail, separators=(',',':'))]=val
    visit(state,[])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)


class DatasetWriter:
    def __init__(self, root):
        self.root=Path(root)
        for p in ("scenes","disturbance_traces"):
            (self.root/p).mkdir(parents=True,exist_ok=True)
        self.scene_manifest=self.root/"scenes.jsonl"
        self.trajectory_manifest=self.root/"trajectories.jsonl"

    def _append(self,path,record):
        with path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(json_safe(record),ensure_ascii=False)+"\n")

    def save_scene(self,scene_id,raw_state,metadata):
        rel=Path("scenes")/f"{scene_id}.npz"
        save_state_npz(self.root/rel,raw_state)
        self._append(self.scene_manifest,{"scene_id":scene_id,
                                        "initial_state":str(rel), **metadata})
        return str(rel)

    def save_trajectory(self,trajectory_id,scene_id,trace,metadata):
        data=trace.arrays()
        rel=Path("disturbance_traces")/f"{trajectory_id}.npz"
        np.savez_compressed(self.root/rel,**data)
        self._append(self.trajectory_manifest,{
            "trajectory_id":trajectory_id,"scene_id":scene_id,
            "disturbance_trace":str(rel),
            "num_steps":trace.steps,
            "object_ids":list(trace.names),
            "disturbance":trace.summary(), **metadata,
        })
        return str(rel)


def load_state_npz(path, device="cpu"):
    """Reconstruct a nested tensor state_dict saved by save_state_npz.

    Intended for team-owned data and SAME ManiSkill/asset configurations.
    Do not assume scene snapshots are portable across simulator versions.
    """
    import torch
    state={}
    with np.load(path,allow_pickle=False) as archive:
        for key in archive.files:
            trail=json.loads(key)
            if not trail:
                raise ValueError("Empty simulator state path")
            cursor=state
            for name in trail[:-1]:
                cursor=cursor.setdefault(name,{})
            cursor[trail[-1]]=torch.as_tensor(archive[key].copy(),device=device)
    return state
