import numpy as np

def validate_xyz_offset(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be three finite XYZ offsets in metres")
    if float(np.linalg.norm(arr)) > 0.25:
        raise ValueError(f"{name} exceeds the diagnostic 25 cm bound")
    return arr

def waypoint_between(start, end, offset, *, min_z=None):
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    delta = validate_xyz_offset(offset, "waypoint offset")
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("start/end must be XYZ vectors")
    pos = 0.5 * (a + b) + delta
    if min_z is not None:
        pos[2] = max(float(pos[2]), float(min_z))
    return pos
