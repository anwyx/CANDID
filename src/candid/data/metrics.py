"""Pure NumPy disturbance metrics shared by episode evaluation and chunk labels."""
from __future__ import annotations
import numpy as np


def angle_between_wxyz(q0, q1):
    q0=np.asarray(q0, dtype=float)
    q1=np.asarray(q1, dtype=float)
    q0=q0/np.maximum(np.linalg.norm(q0,axis=-1,keepdims=True),1e-12)
    q1=q1/np.maximum(np.linalg.norm(q1,axis=-1,keepdims=True),1e-12)
    return 2*np.arccos(np.clip(np.abs(np.sum(q0*q1,axis=-1)),0,1))


def event_flags_from_poses(poses, kinds=None, topple_angle_deg=45.0,
                           table_height_m=0.0, fall_margin_m=0.01):
    """Per-state topple/fall flags [T+1,N], relative to the settled first pose.

    Topple: the object's local +Z axis tilts past a threshold; spheres have no
    upright orientation. Fall: actor origin passes below the tabletop plane.
    CANDID's TableSceneBuilder has its tabletop surface at world Z=0.
    """
    poses=np.asarray(poses,dtype=float)
    if poses.ndim!=3 or poses.shape[-1]!=7 or not np.isfinite(poses).all():
        raise ValueError("Expected finite pose trace [T+1,N,7]")
    if not 0<topple_angle_deg<=180 or fall_margin_m<0:
        raise ValueError("Invalid topple angle or fall margin")
    n=poses.shape[1]
    if kinds is not None and len(kinds)!=n:
        raise ValueError("Object kind count mismatch")
    q=poses[:,:,3:]
    norms=np.linalg.norm(q,axis=-1,keepdims=True)
    if np.any(norms<1e-12):
        raise ValueError("Invalid zero quaternion")
    q=q/norms
    w,x,y,z=np.moveaxis(q,-1,0)
    up=np.stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)),axis=-1)
    cos_tilt=np.sum(up*up[0:1],axis=-1)
    toppled=cos_tilt<np.cos(np.deg2rad(topple_angle_deg))
    if kinds is not None:
        toppled[:,np.asarray(kinds)=="sphere"]=False
    fallen=poses[:,:,2]<(table_height_m-fall_margin_m)
    return {"distractor_toppled":toppled,"distractor_fallen":fallen}


def disturbance_from_poses(poses, sizes_m, start=0, stop=None, w_translation=0.7,
                           w_rotation=0.3, topple_flags=None, fall_flags=None):
    """Affects [start,stop] state interval inclusive; one score per object.

    NOTE: peak translation/rotation reference is the start state, not the
    previous state. Computing chunk labels does not require re-simulation.
    Event counts include only newly occurring false-to-true transitions.
    """
    poses=np.asarray(poses,dtype=float)
    if poses.ndim!=3 or poses.shape[-1]!=7:
        raise ValueError("Expected pose trace [T+1,N,7]")
    n=poses.shape[1]
    sizes=np.asarray(sizes_m,dtype=float)
    if sizes.shape!=(n,) or np.any(sizes<=0) or not np.isfinite(sizes).all():
        raise ValueError("Require positive characteristic length for each object")
    if w_translation<0 or w_rotation<0:
        raise ValueError("Metric weights cannot be negative")
    stop=poses.shape[0]-1 if stop is None else stop
    if not (0<=start<=stop<poses.shape[0]):
        raise ValueError("Invalid inclusive state interval")
    segment=poses[start:stop+1]
    reference=poses[start]
    translation=np.linalg.norm(segment[:,:,:3]-reference[None,:,:3],axis=-1)
    rotation=angle_between_wxyz(reference[None,:,3:],segment[:,:,3:])
    peak_trans=translation.max(axis=0)
    peak_rot=rotation.max(axis=0)
    per_obj=w_translation*peak_trans/sizes+w_rotation*peak_rot/np.pi
    result={
        "peak_translation_m":peak_trans.tolist(),
        "residual_translation_m":translation[-1].tolist(),
        "peak_rotation_rad":peak_rot.tolist(),
        "residual_rotation_rad":rotation[-1].tolist(),
        "per_object_dr":per_obj.tolist(),
        "dr_mean":float(per_obj.mean()) if n else 0.0,
        "dr_max":float(per_obj.max()) if n else 0.0,
        "continuous_dr":float(per_obj.mean()) if n else 0.0,
    }
    for flag,name in ((topple_flags,"topple"),(fall_flags,"fall")):
        if flag is None:
            continue
        flag=np.asarray(flag)
        if flag.shape!=poses.shape[:2] or flag.dtype.kind!="b":
            raise ValueError(f"{name} flags must be boolean [T+1,N]")
        new_event=np.any(~flag[start:stop] & flag[start+1:stop+1],axis=0)
        result[f"per_object_{name}"]=new_event.tolist()
        result[f"{name}_count"]=int(new_event.sum())
        result[f"any_{name}"]=bool(new_event.any())
    return result
