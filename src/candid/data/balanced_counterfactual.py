"""Balanced expert-state probe selection for physical counterfactual replay.

Only archived expert states supply probe locations. Physical labels for new
actions are NEVER inferred from this sampler: collect() re-simulates them.
Category definitions are transparent observable proxies, not contact labels.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

import h5py
import numpy as np

from candid.data.counterfactual import Probe
from candid.data.metrics import disturbance_from_poses


STRATUM_WEIGHTS = {
    "near_contact": .30,
    "low_dr": .25,
    "pre_topple": .20,
    "slight_motion": .15,
    "high_dr": .10,
}
STRATUM_NAMES = tuple(STRATUM_WEIGHTS)


def stratum_quotas(count):
    """Largest-remainder integer allocation with stable, exact total."""
    if count < 1:
        raise ValueError("Probe count must be positive")
    exact = {k: count * v for k, v in STRATUM_WEIGHTS.items()}
    counts = {k: int(np.floor(v)) for k, v in exact.items()}
    remainder = count - sum(counts.values())
    order = sorted(STRATUM_NAMES, key=lambda k: (- (exact[k]-counts[k]), STRATUM_NAMES.index(k)))
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(x) for x in handle if x.strip()]


def _tcp_poses(root, row, expected):
    """Read named expert TCP positions (not fitted labels) for clearance proxy."""
    with h5py.File(root / row["hdf5_file"], "r") as handle:
        tcp = np.asarray(
            handle[row["hdf5_trajectory_key"]]["obs"]["extra"]["tcp_pose"],
            dtype=np.float64,
        )
    if tcp.size != expected * 7:
        raise ValueError(f"Unexpected TCP pose shape in {row['trajectory_id']}: {tcp.shape}")
    return tcp.reshape(expected, 7)[:, :3]


def _probe_options(root, row, rng, *, action_horizon, obs_horizon, max_prefix_steps):
    t_min = obs_horizon - 1
    t_max = int(row["num_steps"]) - action_horizon
    if max_prefix_steps is not None:
        t_max = min(t_max, max_prefix_steps)
    if t_max < t_min:
        return {}

    with np.load(root / row["disturbance_trace"], allow_pickle=False) as archive:
        poses = np.asarray(archive["distractor_poses"], dtype=np.float64)
        sizes = np.asarray(archive["object_sizes_m"], dtype=np.float64)
        topple = np.asarray(archive["distractor_toppled"], dtype=bool)
        fall = np.asarray(archive["distractor_fallen"], dtype=bool)

    if poses.shape[0] != int(row["num_steps"]) + 1:
        raise ValueError(f"Trace length mismatch: {row['trajectory_id']}")
    if topple.shape[0] != len(poses) or fall.shape[0] != len(poses):
        raise ValueError(f"Event trace length mismatch: {row['trajectory_id']}")
    tcp = _tcp_poses(root, row, len(poses))
    n = poses.shape[1]
    if n < 1:
        return {}

    # Planar separation from an object's bounding sphere projected to XY.
    # It is NOT exact gripper-arm clearance or collision geometry.
    with h5py.File(root / row["hdf5_file"], "r") as handle:
        dims = np.asarray(
            handle[row["hdf5_trajectory_key"]]["obs"]["extra"]["distractor_dimensions"][0],
            dtype=np.float64,
        ).reshape(n, 3)
    bounding_radius_xy = .5 * np.linalg.norm(dims[:, :2], axis=1)

    ts = rng.choice(
        np.arange(t_min, t_max + 1),
        size=min(48, t_max - t_min + 1), replace=False,
    )
    options = {}
    best = {
        "near_contact": (float("inf"), None),
        "low_dr": (float("inf"), None),
        "slight_motion": (float("-inf"), None),
        "high_dr": (float("-inf"), None),
    }

    for ti in ts:
        t = int(ti)
        if not np.isfinite(tcp[t]).all():
            continue
        dist_xy = np.linalg.norm(
            poses[t, :, :2] - tcp[t, None, :2], axis=-1) - bounding_radius_xy
        dz = np.abs(poses[t, :, 2] - tcp[t, 2])
        clearance = float(np.min(np.where(dz <= .14, dist_xy, np.inf)))
        dr = disturbance_from_poses(
            poses, sizes, t, t + action_horizon,
            topple_flags=topple, fall_flags=fall,
        )["dr_mean"]
        quiet_start = not (topple[t].any() or fall[t].any())
        quiet_window = not (topple[t:t + action_horizon + 1].any()
                            or fall[t:t + action_horizon + 1].any())
        displacement = float(np.linalg.norm(
            poses[t, :, :3] - poses[t-1, :, :3], axis=-1).max())

        if quiet_window and dr <= .02 and clearance <= .10:
            score = (max(0., clearance), t)
            if score[0] < best["near_contact"][0]:
                best["near_contact"] = score
        if quiet_window and dr <= .005:
            # Prefer low physical disturbance; this category need not be far.
            if dr < best["low_dr"][0]:
                best["low_dr"] = (float(dr), t)
        if quiet_start and .0005 <= displacement <= .01:
            # Motion threshold is metres PER logged action, not m/s.
            if displacement > best["slight_motion"][0]:
                best["slight_motion"] = (displacement, t)
        if quiet_start and dr >= .02:
            if dr > best["high_dr"][0]:
                best["high_dr"] = (float(dr), t)

    for category, (_, t) in best.items():
        if t is not None:
            options[category] = Probe(row["scene_id"], row["trajectory_id"],
                                      t, category, row)
    onset = np.flatnonzero(np.any(~topple[:-1] & topple[1:], axis=1)) + 1
    for instant in onset:
        t = max(t_min, int(instant) - action_horizon)
        if t <= t_max and not topple[t].any() and not fall[t].any():
            options["pre_topple"] = Probe(
                row["scene_id"], row["trajectory_id"], t, "pre_topple", row)
            break
    return options


def choose_balanced_probes(dataset, allowed_scene_ids, *, count, seed=0,
                           action_horizon=8, obs_horizon=2, max_per_scene=10,
                           max_prefix_steps=None, reserve_count=0):
    """Return quota-controlled primary probes followed by stratum-matched spares.

    At most one probe per expert trajectory across BOTH primary and spares.
    Global per-scene cap includes reserves. When eligible supply cannot cover
    quotas, fail BEFORE expensive physical rollouts rather than silently bias.
    Collector additionally enforces quotas on physically accepted states.
    """
    if min(count, action_horizon, obs_horizon, max_per_scene) < 1 or reserve_count < 0:
        raise ValueError("Invalid counts/horizons/cap/reserves")
    root = Path(dataset)
    allowed = set(allowed_scene_ids)
    if not allowed:
        raise ValueError("No allowed training scenes")
    rows = [r for r in _read_jsonl(root / "trajectories.jsonl")
            if r["scene_id"] in allowed and r.get("saved_hdf5")
            and int(r["num_steps"]) >= obs_horizon - 1 + action_horizon]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    buckets = defaultdict(list)
    for row in rows:
        available = _probe_options(
            root, row, rng, action_horizon=action_horizon,
            obs_horizon=obs_horizon, max_prefix_steps=max_prefix_steps,
        )
        for category, probe in available.items():
            buckets[category].append(probe)

    quotas = stratum_quotas(count)
    selected = []
    used_trajectories = set()
    scene_counts = defaultdict(int)
    pointers = {k: 0 for k in STRATUM_NAMES}

    def take(category, n):
        added = 0
        options = buckets[category]
        while added < n and pointers[category] < len(options):
            probe = options[pointers[category]]
            pointers[category] += 1
            if (probe.trajectory_id in used_trajectories
                    or scene_counts[probe.scene_id] >= max_per_scene):
                continue
            selected.append(probe)
            used_trajectories.add(probe.trajectory_id)
            scene_counts[probe.scene_id] += 1
            added += 1
        return added

    # Prioritize scarce/semantically distinct categories, not pre-topple.
    # The accepted-state quotas are checked independently in collect().
    for category in ("near_contact", "slight_motion", "low_dr", "pre_topple", "high_dr"):
        added = take(category, quotas[category])
        if added < quotas[category]:
            raise ValueError(
                f"Balanced sampling cannot fill {category}: "
                f"needed {quotas[category]}, found {added}. "
                f"Eligible trajectories for this stratum: {len(buckets[category])}. "
                "Reduce --states, raise --max-per-scene, or inspect proxy thresholds."
            )

    # Reserve distribution follows the SAME categorical ratios. Exhausted
    # categories do not silently turn into extra pre-topple samples.
    extra_targets = stratum_quotas(reserve_count) if reserve_count else {
        k: 0 for k in STRATUM_NAMES}
    for category in ("near_contact", "slight_motion", "low_dr", "pre_topple", "high_dr"):
        take(category, extra_targets[category])
    return selected
