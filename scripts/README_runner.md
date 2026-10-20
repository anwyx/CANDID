# CANDID — reusable environment runner and disturbance metrics

These files refactor the uploaded `scripts/run_baseline.py`. They **do not** change the baseline environment or `planners/execute_baseline.py`.

## 1. Install files into the Git repository

```text
CANDID/
├── envs/
│   ├── __init__.py
│   └── candid_baseline_env.py       # your existing environment
├── planners/
│   ├── __init__.py
│   └── execute_baseline.py         # your existing planner
├── metrics/
│   ├── __init__.py                 # NEW
│   └── disturbance_monitor.py      # NEW
└── scripts/
    └── run_baseline.py            # REPLACE old runner
```

Run commands **from the CANDID project root** inside your ManiSkill Docker container.

## 2. Basic usage

```bash
# Existing baseline: default environment, default planner
python scripts/run_baseline.py --episodes 2 --seed 0 --obs-mode none --save-video

# Collect observations suitable for state-based training *only after your
# environment's _get_obs_extra includes the required object states*.
python scripts/run_baseline.py --episodes 20 --obs-mode state \
  --output-dir demos/baseline_state_test

# Keep the original fixed baseline scene (if your env supports the flag).
python scripts/run_baseline.py --episodes 1 --fixed-scene

# Baseline planner waypoint variation
python scripts/run_baseline.py --episodes 1 --seed 42 --lateral-offset 0.04
```

The default planner still uses `pd_joint_pos` and the xArm6 Robotiq robot.
Using `--obs-mode none` is good for quick debugging, but **does not create policy training observations**.

## 3. Run an environment created by a teammate

Suppose someone adds `envs/primitive_env.py`:

```python
from mani_skill.utils.registration import register_env
from envs.candid_baseline_env import CANDIDBaselineEnv

@register_env("CANDIDPrimitive-v0", max_episode_steps=500)
class CANDIDPrimitiveEnv(CANDIDBaselineEnv):
    # Override _load_scene() / _initialize_episode() and appropriate
    # geometry metadata as needed. This is only a registration example.
    pass
```

Use the **same runner**:

```bash
python scripts/run_baseline.py \
  --env-module envs.primitive_env \
  --env-id CANDIDPrimitive-v0 \
  --episodes 20 --seed 100 \
  --obs-mode none --save-video \
  --output-dir demos/primitive_test
```

`--env-module` is the **Python import path** (not a filesystem filename).
Importing that module registers the ID before `gym.make()` runs.

If an environment requires extra constructor settings, pass them using JSON:

```bash
python scripts/run_baseline.py \
  --env-module envs.custom_env --env-id CANDIDCustom-v0 \
  --env-kwargs '{"num_distractors": 6, "randomize_layout": true}' \
  --episodes 5
```

All keys in `--env-kwargs` must be supported by that environment's constructor.
The `--fixed-scene` shortcut passes `randomize_layout=False` and should only
be used for environments accepting that parameter. It is not universal.

## 4. Shared interface for environments and planners

A teammate can use the **default** `planners.execute_baseline:solve` if their
environment still supports the baseline Pick-and-Place interface:

- `env.unwrapped.cube`: the grasp target object;
- `env.unwrapped.goal_site`: placement goal;
- `env.unwrapped.cube_half_size`: target half-height;
- `env.unwrapped.agent`: xArm6 Robotiq agent;
- `env.unwrapped.evaluate()["success"]`: task success;
- `control_mode="pd_joint_pos"`.

If these differ (e.g. the target is no longer a cube, or the gripper/robot is
different), specify a **different planner**. The runner does **not** magically
make the old xArm6 cube solver work on every task.

```bash
python scripts/run_baseline.py \
  --env-module envs.ycb_env --env-id CANDIDYCB-v0 \
  --planner planners.ycb_planner:solve \
  --planner-config-class planners.ycb_planner:PlanConfig \
  --plan-kwargs '{"lift_height": 0.15}' \
  --episodes 10 --obs-mode state
```

Required planner API:

```python
def solve(env, seed, config=None, vis=False, verbose=False):
    env.reset(seed=seed)  # required, or the recorder will not know episode start
    # Execute actions by calling env.step(actual_action)
    return {"success": True, "stage": "completed", "steps": 150}
```

The solver can return a dict or a dataclass/object (e.g. `PlanResult`) with
these fields. For planners not using a config class, pass
`--planner-config-class none` (and no `--plan-kwargs`).

**Each planner must explicitly reset exactly once per attempt.** During
execution it must step the *wrapped* `env` passed into `solve`, not only
`env.unwrapped`, so that actions are recorded and the monitor runs.

## 5. Shared disturbance interface for new object types

`metrics/disturbance_monitor.py` is **independent** of any particular env.
For accurate metrics, implement these on each new environment:

```python
def get_distractor_poses_for_logging(self):
    # [1, N, 7] tensor or [N, 7] NumPy array; quaternion order [w,x,y,z].
    # Do NOT include these ground-truth poses in policy observations by accident.
    return torch.stack([obj.pose.raw_pose for obj in self.clutter_cubes], dim=1)

def get_distractor_sizes_for_logging(self):
    # [N] positive characteristic sizes in metres, in the SAME object order.
    # For cubes, side length; for irregular objects, select/document a
    # comparable geometric scale (e.g. bounding-box max dimension).
    return [0.04, 0.08, 0.06, 0.10]

def get_scene_layout_for_logging(self):
    # Optional reproducible scene config; do not use these truths in DP inputs.
    return {"objects": "...", "seed_info": "..."}
```

The monitor also falls back to `clutter_cubes` and `DISTRACTOR_HALF_SIZE` for
old cube environments. **Do not rely on `DISTRACTOR_HALF_SIZE` when shapes or
sizes vary:** implement `get_distractor_sizes_for_logging()` to avoid silently
mis-normalizing the continuous disturbance.

If no pose interface exists, the runner still records task actions/observations;
`continuous_dr` is **null** and no pose trace is saved. Missing labels are
never treated as zero disturbance.

Continuous DR is currently the average per-object **peak** consequence:

```text
D_i = w_p * peak_translation_i / characteristic_size_i
    + w_R * peak_rotation_i / pi
continuous_dr = mean(D_i)
d_max = max(D_i)
```

Default weights: `w_p=0.7, w_R=0.3`. Override using
`--translation-weight 0.7 --rotation-weight 0.3`.
The peak is measured relative to the **episode's initial object pose**.
Topple/Fall are separate heuristic flags; adjust definitions before using them
for YCB objects, tilted initial states, or different table heights.

The pure `continuous_disturbance_score()` function is defined in the same
`metrics/disturbance_monitor.py` so future **chunk-level** preprocessing can
reuse exactly the same normalisation. Current `continuous_dr` is episode-level;
it is **not** yet a chunk-level label.

## 6. Outputs and dataset handling

Each run creates files under its `--output-dir`:

```text
demos/primitive_test/
├── trajectories_<timestamp>.h5
├── trajectories_<timestamp>.json
├── episode_metrics.jsonl
├── label_traces/
│   ├── run_<timestamp>_attempt_00000.npz
│   └── ...
└── *.mp4                 # only with --save-video
```

- HDF5: executed actions and observations, not raw simulator truth as env state.
- `.npz`: `[T+1,N,7]` distractor pose trace, plus object normalisation sizes.
- JSONL: success/failure, env/planner/config/seed/commit, metric summaries,
  HDF5 trajectory key and the matching label trace file.
- New `hdf5_file` and `label_trace` fields are **relative to output-dir** so a
  completed Run can be moved to shared storage without breaking its references.
- Keep separate `--output-dir` values for runs generated with different
  object geometries or observation formats; do not mix them in one training set
  without checking compatibility.
- Use `--save-success-only` for a DP imitation dataset, but for the Predictor
  generally retain valid failed rollouts too (default).
- `--no-record --save-video` saves videos/metrics without large HDF5 files.

Finally, multiple trajectories for **the same scene** require reusing the same
seed and changing explicit planner parameters; `--episodes 10` normally means
*ten different seeds*, not ten trajectory variants of the same scene.

## 7. Suggested sanity checks

```bash
python -m py_compile scripts/run_baseline.py metrics/disturbance_monitor.py
python scripts/run_baseline.py --help
python scripts/run_baseline.py --episodes 2 --obs-mode none --save-video
```

If you see `continuous_dr=N/A`, check the environment's pose and size callbacks.
This refactor was syntax-checked and exercised with simulated (mock) environment
interfaces; full ManiSkill/MPlib runtime testing must be done in your Docker.
