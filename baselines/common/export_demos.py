"""Export frozen TRAIN scenes to ManiSkill state-based ACT/BC HDF5 format.

CANDID stores state_dict HDF5 groups + trajectories.jsonl. Official ACT/BC
load flat obs under traj_N/obs and require a sibling .json manifest.
Do not include validation or test trajectories in the imitation dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from candid.data.adapter import canonical_observation, observation_vector
from candid.data.splits import load_or_create_scene_split


def export_demos(dataset_root, output_path, success_only=True, seed=0, max_trajectories=None):
    root = Path(dataset_root)
    out = Path(output_path)
    if out.suffix != ".h5":
        raise ValueError("Output path must end in .h5")
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError("Refusing to overwrite an existing exported dataset")
    split = load_or_create_scene_split(root, seed=seed)
    allowed = set(split["train"])
    rows = [json.loads(x) for x in (root / "trajectories.jsonl").read_text().splitlines() if x.strip()]
    scenes = {x["scene_id"]: x for x in
              (json.loads(s) for s in (root / "scenes.jsonl").read_text().splitlines() if s.strip())}
    rows = [row for row in rows if row["scene_id"] in allowed
            and row.get("saved_hdf5") and (row.get("task_success") or not success_only)]
    if max_trajectories is not None:
        if max_trajectories < 1:
            raise ValueError("--max-trajectories must be positive")
        rows = rows[:max_trajectories]
    if not rows:
        raise ValueError("No recorded training trajectories satisfy the filter")
    out.parent.mkdir(parents=True, exist_ok=True)
    episodes = []
    control = "pd_joint_pos"
    observation_dim = action_dim = None
    source_files = {}
    try:
        with h5py.File(out, "w") as target:
            for number, row in enumerate(rows):
                if row.get("control_mode", control) != control or row.get("obs_mode", "state_dict") != "state_dict":
                    raise ValueError(f"Wrong control or observation mode: {row['trajectory_id']}")
                filename = row["hdf5_file"]
                if filename not in source_files:
                    source_files[filename] = h5py.File(root / filename, "r")
                src = source_files[filename][row["hdf5_trajectory_key"]]
                actions = np.asarray(src["actions"], dtype=np.float32)
                n = len(row["object_ids"])
                obs = observation_vector(canonical_observation(src["obs"], n))
                if obs.shape[0] != actions.shape[0] + 1:
                    raise ValueError(f"State/action alignment failed: {row['trajectory_id']}")
                if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                    raise ValueError(f"Nonfinite demonstration: {row['trajectory_id']}")
                if observation_dim is None:
                    observation_dim = obs.shape[1]
                    action_dim = actions.shape[1]
                if obs.shape[1] != observation_dim or actions.shape[1] != action_dim:
                    raise ValueError("All training trajectories must share observation/action dimensions")
                grp = target.create_group(f"traj_{number}")
                grp.create_dataset("obs", data=obs, compression="gzip")
                grp.create_dataset("actions", data=actions, compression="gzip")
                # Official BC reads success[t] as an auxiliary field.
                grp.create_dataset("success", data=np.full(len(actions),
                    int(bool(row["task_success"])), dtype=np.bool_))
                episodes.append({"episode_id": number, "source_trajectory_id": row["trajectory_id"],
                                 "source_scene_id": row["scene_id"], "control_mode": control})
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    finally:
        for handle in source_files.values():
            handle.close()
    first_scene = scenes[rows[0]["scene_id"]]
    source_environments = {}
    for row in rows:
        source = scenes[row["scene_id"]]
        source_key = (source["env_id"], source["env_module"])
        source_environments[source_key] = source_environments.get(source_key, 0) + 1
    metadata = {
        "env_info": {
            "env_id": first_scene["env_id"],
            "env_kwargs": {"control_mode": control, "obs_mode": "state_dict"},
        },
        "episodes": episodes,
        "candid": {
            "source_dataset": str(root), "split": "train",
            "split_sha256": split["dataset_sha256"],
            "source_environments": [
                {"env_id": env_id, "env_module": env_module,
                 "num_trajectories": count}
                for (env_id, env_module), count in sorted(source_environments.items())
            ],
            "env_info_is_representative": True,
            "success_only": bool(success_only),
            "max_trajectories": max_trajectories,
            "observation_dim": observation_dim, "action_dim": action_dim,
            "observation_fields": "candid.data.adapter.OBSERVATION_FIELDS",
        },
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"path": str(out), "trajectories": len(episodes),
            "observation_dim": observation_dim, "action_dim": action_dim}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-trajectories", type=int, default=None,
                        help="Export only the first N eligible TRAIN trajectories for quick CPU smoke tests")
    args = parser.parse_args()
    print(export_demos(args.dataset, args.output,
                       success_only=not args.include_failures, seed=args.seed,
                       max_trajectories=args.max_trajectories))


if __name__ == "__main__":
    main()
