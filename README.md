# CANDID — Baseline Scene, Task Development, and Data Collection Guide

> **Project:** Consequence-Aware Non-target Disturbance-Informed Diffusion  
> **Simulator:** ManiSkill 3 | **Robot:** xArm6 + Robotiq parallel gripper  
> **Task:** One target-object pick-and-place in movable clutter  

This document explains how to start from the **shared, tested baseline**, create new tabletop scene variants, run the motion-planning demonstration, inspect videos and failure modes, and collect reproducible data for the **Vanilla Diffusion Policy (DP)** and **chunk-level Disturbance Predictor**.

### Start here

If this is your first time using the repository: **§2** verifies imports, **§4** runs a smoke test, **§5** explains videos, **§7** explains new scene environments, and **§9–10** explain data collection and splits.

**Important:** The current three baseline files implement **one fixed scene, one nominal planning recipe, and a basic disturbance logger**. They do **not** yet implement YCB assets, randomized scene generation, an 800-demonstration DP dataset, a complete chunk-level predictor dataset, or a trained CANDID policy. The sections below distinguish **what works now** from **what contributors need to implement next**.

---

## 1. Research task and non-negotiable conventions

Every variant must preserve the **same task**:

1. Move the gripper toward the designated target object.
2. Grasp and lift **that target only**.
3. Carry the target to the goal region.
4. Place and release the target.
5. Minimize unintended movement of all **non-target movable objects** (distractors).

**Allowed:** incidental contact involving the arm, gripper, or carried target, including a small amount of resulting distractor motion.  
**Not allowed:** adding a separate action to deliberately pick up, remove, or rearrange a distractor before completing the target task.

We use **task success** and **disturbance** as *separate* outcomes. A successful place can still produce substantial disturbance; that is an important CANDID sample, not an automatically failed task.

Do **not** divide the project into different "grasp-side" and "transfer-side" tasks. Each scene is one complete pick-and-place setup with distractors anywhere in the shared pickup/transfer workspace. We distinguish scene variants by **object geometry and composition**: primitives, custom geometric objects, selected YCB objects, and mixed scenes.

### Data and sim-to-real rule

**Only real-robot-obtainable information is permitted as DP/Predictor input.** For the initial implementation this means **RGB-D / camera-derived point clouds, robot proprioception, a calibrated goal definition, and candidate control actions**. Simulator ground-truth object states are allowed **for creating teacher demonstrations, calculating supervision labels, and evaluating results**, but **must never leak into the model's input observations**.

The baseline currently uses the simulator's true target pose inside the **expert motion planner** to plan demonstrations. This is acceptable as privileged **teacher information**, but it does not mean the learned policy may use it. Real-robot deployability still depends on camera calibration, perception quality, and compatible action/control execution.

---

## 2. Project structure and current source files

Run commands from the **`CANDID/` project root**, not from `scripts/`.

```text
workspace/
├── CANDID/                            # Git repository root
│   ├── README.md                       # This file
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── candid_baseline_env.py       # Fixed reference scene
│   │   └── ...                         # New registered scene variants
│   ├── planners/
│   │   ├── __init__.py
│   │   └── execute_baseline.py         # xArm6 / MPlib demonstration
│   ├── scripts/
│   │   └── run_baseline.py             # CLI, recording, monitoring
│   └── demos/                          # Generated data; do not commit
│       └── candid_baseline/
└── videos/                             # Optional shared video folder
```

| File | Responsibility | Relevant public interfaces |
|---|---|---|
| `envs/candid_baseline_env.py` | Scene construction, object reset, camera, robot base, success/reward | `CANDIDBaselineEnv`, `_load_scene`, `_initialize_episode`, `_get_obs_extra`, `evaluate`, `get_distractor_poses_for_logging` |
| `planners/execute_baseline.py` | Generate/execute one pick-and-place attempt | `BaselinePlanConfig`, `solve(env, seed, config, vis, verbose)` |
| `scripts/run_baseline.py` | Create the environment, record HDF5/videos, calculate disturbance metrics | `build_env`, `DisturbanceMonitor`, `main` |


Run from `CANDID/` so the script's `PROJECT_ROOT` path setup can resolve `envs` and `planners`. If the `envs` package has not been initialized, run `touch envs/__init__.py` once. If the currently pushed runner already imports from `envs`, **do not change it again**.

**Quick import verification:**

```bash
cd /workspace/CANDID
python -c "from envs import candid_baseline_env; print('CANDID environment import OK')"
```

### Environment and package requirements

Use the shared ManiSkill 3 environment (the same one in which the baseline already ran), with at least:

- Python, ManiSkill, Gymnasium, NumPy, PyTorch, SAPIEN
- `mplib` (for the expert planner)
- `h5py` (for saved demonstrations)
- A working rendering backend **only if** producing RGB-D observations or MP4 videos

In the team's CPU-only Docker setup, ManiSkill can warn that CUDA rendering is unavailable. A **successful `--obs-mode none` run is not proof that RGB-D or MP4 capture works**. Check camera capture separately.

---

## 3. Shared baseline configuration: do not silently change

The fixed baseline was used to validate the whole pick-and-place execution. Its layout is the reference for subsequent randomization.

| Configuration | Baseline value | Comment |
|---|---|---|
| Gym environment ID | `CANDIDBaseline-v0` | Registered via `@register_env` |
| Robot | `xarm6_robotiq` | Fixed model and gripper |
| Robot base position | `(-0.522, 0.0, 0.0)` m | World frame |
| Robot initialization noise | `0.0` | Deterministic rest pose |
| Target object | Red cube, side length `0.04` m | `self.cube` |
| Target initial center | `(-0.12, 0.0, 0.02)` m | XY fixed; bottom rests on table |
| Goal center XY | `(0.10, 0.0)` m | Goal **marker** lies on table |
| Goal marker radius | `0.060` m | Visual marker, not a collision obstacle |
| Goal success XY tolerance | `0.045` m | Used in `evaluate()` |
| Goal center-height tolerance | `0.025` m | Cube should rest on table |
| Number of baseline distractors | `4` | All are movable, independently tracked |
| Baseline distractor center XY | `(-0.155, 0.078)`, `(-0.035, 0.071)`, `(-0.035, -0.077)`, `(-0.155, -0.090)` | Four separate 4-cm cubes |
| Shared distractor center X range | `[-0.22, 0.03]` m | Intended randomization workspace |
| Shared distractor center Y range | `[-0.16, 0.16]` m | Intended randomization workspace |
| Baseline minimum surface gap | `0.010` m | Additional anti-overlap clearance |
| Goal-marker free gap | `0.030` m | Additional clearance beyond marker/footprint |
| Sensor camera | `(0.23, 0.35, 0.65)` m | Looks at `(-0.06, 0.0, 0.04)` |
| Sensor resolution | `256 × 256` | RGB-D camera source |
| Human video camera | `(0.52, 0.65, 0.58)` m | Looks at `(-0.06, 0.0, 0.12)` |

**Coordinates:** meters, ManiSkill world frame, table surface `z=0`; robot is at negative X; +X generally points toward the goal. Object positions denote **centers**, not bottom faces. Orientations in logged actor pose arrays are quaternions ordered `(qw, qx, qy, qz)`.

**Do not** modify the shared baseline to experiment with a new shape. Create another registered environment class, keep its scene/configuration identifiable, and run separate experiments.

---

## 4. Run the working baseline first

From the project root:

```bash
cd /workspace/CANDID

# Smoke test: minimal observation, no video or large RGB-D collection
python scripts/run_baseline.py --episodes 1 --obs-mode none --verbose
```

Example result from a previously successful run (illustrative; your results may differ):

```text
success=True  stage=completed  steps=194  peak_clutter_translation=0.0000 m
```

This confirms that the script executed a valid baseline pick-and-place. It does **not** establish low-disturbance performance in random clutter, because the scene and nominal path were both fixed.

### Run with visual data and a debugging video

```bash
python scripts/run_baseline.py \
  --episodes 1 \
  --obs-mode rgbd \
  --save-video \
  --verbose \
  --output-dir demos/candid_baseline_visual
```

To record several runs:

```bash
python scripts/run_baseline.py \
  --episodes 5 \
  --seed 0 \
  --obs-mode rgbd \
  --save-video \
  --output-dir demos/candid_baseline_repeat
```

**Note:** In the fixed baseline, varying the reset seed **does not** randomize the object layout. Five repeated trials are **not** five different scenes and should not be counted as independent training geometries.

### Test an alternative transfer route

The planner exposes one simple control for baseline trajectory variation: an optional intermediate waypoint offset in world Y.

```bash
# Nominal/direct transfer
python scripts/run_baseline.py --episodes 1 --lateral-offset 0.00 \
  --obs-mode rgbd --save-video --output-dir demos/route_center

# Waypoint shifted to +Y by 4 cm
python scripts/run_baseline.py --episodes 1 --lateral-offset 0.04 \
  --obs-mode rgbd --save-video --output-dir demos/route_left

# Waypoint shifted to -Y by 4 cm
python scripts/run_baseline.py --episodes 1 --lateral-offset -0.04 \
  --obs-mode rgbd --save-video --output-dir demos/route_right
```

The sign is defined in the **world Y axis**, not the robot's camera-left/right directions. The CLI currently allows offsets within `[-0.15, 0.15]` m; this is an *input bound*, **not** a guarantee of reachability or safe execution. Every result must be validated in simulation.

### Useful CLI options (already implemented)

| Argument | Meaning | Notes |
|---|---|---|
| `--episodes N` | Number of attempts | Not unique scenes |
| `--seed N` | Initial reset seed | Incremented once per attempt |
| `--obs-mode none\|rgbd\|pointcloud` | Saved observation modality | Use `rgbd` for primary data; `none` for debugging |
| `--lateral-offset X` | Transfer-waypoint Y offset, in meters | `0` means nominal direct path |
| `--sim-backend cpu\|gpu\|auto` | Simulation backend | Current script defaults to CPU |
| `--max-episode-steps N` | Episode time limit | Runner defaults to `500` |
| `--output-dir PATH` | Trajectory/metric/label output location | Keep each experiment separate |
| `--traj-name NAME` | HDF5/JSON file basename | Avoid colliding output names |
| `--save-video` | Record MP4 video | Needs rendering support |
| `--vis` | Open interactive viewer | Usually omit in headless Docker |
| `--verbose` | Print planner phases | Good for debugging |
| `--save-success-only` | Save only successful HDF5 demonstrations | **Do not** enable for full predictor-data collection |
| `--no-record` | Disable HDF5 recorder | Still writes JSONL and label traces |

**Not implemented in the current script:** `--env-id`, `--video-dir`, a general random-scene sampler, YCB selection, and full chunk-label preprocessing. Do not use those flags until the relevant code has been added.

---

## 5. Find, copy, and inspect videos

With `--save-video`, ManiSkill's `RecordEpisode` writes MP4 files under the chosen `--output-dir` (alongside trajectory outputs). From the root:

```bash
find demos/candid_baseline_visual -type f -name '*.mp4' -print
```

To keep videos in a folder **next to** `CANDID/`, as agreed:

```bash
mkdir -p ../videos
find demos/candid_baseline_visual -type f -name '*.mp4' \
  -exec cp -v {} ../videos/ \;
ls -lh ../videos/
```

At present, this **copies** the videos; it does not move them or change `run_baseline.py`'s output behavior. Choose a distinct output directory or filename per experiment to avoid overwriting videos with similar names.

**Docker caveat:** `/workspace/videos/` may only exist **inside the container** unless it is mounted from the host. Either add a Docker bind mount for that sibling directory or copy out from the **Mac host terminal**, e.g.:

```bash
docker cp maniskill-dev:/workspace/videos ./videos
```

Run the last command from the host directory in which you want `videos/` created. Adjust the container name if yours differs.

### Manual video QA checklist

For each new scene type, inspect a few **successful**, **failed**, **low-disturbance**, and **high-disturbance** rollouts. Check:

1. Can the target and goal be seen clearly? Are any objects outside the table/camera field of view?
2. Is there initial interpenetration or an object wobbling/moving **before** the robot starts?
3. Does the gripper approach the **target** and grasp it securely without starting a distractor-removal subtask?
4. Does the target stay held during transfer? Does the planner take unexpected overhead detours?
5. Does the gripper release at the goal, retreat, and leave the target resting there?
6. Which object was touched: the arm, gripper, or carried target? Did the distractor slide, rotate, topple, or fall?
7. Do the video's physical events agree with the saved success/metrics report?

A single video is a **debugging aid**, not quantitative evidence. Use HDF5 + label traces for metrics and comparisons. If `--obs-mode none` works but RGB-D/video fails, diagnose rendering separately from motion planning.

---

## 6. What the planner does, and what it does not do

`planners/execute_baseline.py` uses ManiSkill's official `XArm6RobotiqMotionPlanningSolver` (MPlib), and follows:

```text
Reset → Open gripper → Pregrasp → Approach → Close gripper
      → Lift + verify target is grasped
      → [optional lateral transfer waypoint]
      → Transfer above goal → Lower → Release
      → Retreat → Settle → Evaluate success
```

The current motion parameters are defined by `BaselinePlanConfig` in the planner file:

| Planner parameter | Baseline default | Meaning |
|---|---:|---|
| `approach_height` | 0.065 m | Clearance for pregrasp |
| `lift_height` | 0.120 m | Lift distance after grasp |
| `transfer_clearance` | 0.080 m | Extra height above the place pose |
| `retreat_height` | 0.100 m | Retreat after release |
| `gripper_close_steps` | 10 | Hold closed gripper command |
| `gripper_open_steps` | 12 | Hold open gripper command |
| `settle_steps` | 12 | Stabilization before final success evaluation |
| `lateral_offset` | 0.0 m | Optional transfer midpoint Y offset |

The planner combines MPlib RRT* and screw-style motion planning. **Movable distractors are not supplied as hard point-cloud obstacles** to MPlib in this version, but the ManiSkill simulator still resolves actual physical collisions. MPlib planning success is **not** the same as task success or a guarantee of zero disturbance.

The baseline planner uses **`control_mode="pd_joint_pos"`**. Saved demonstration actions are **joint-position targets plus the gripper command**, not EE delta poses. A future DP must use this action convention **or** run a validated, consistent conversion process before training. Do not change action representations silently between teammates.

**Current planner limitation:** It uses a single top-down grasp of `env.unwrapped.cube` and expects `env.unwrapped.goal_site` and `cube_half_size`. This works for the **standard target cube**. Switching the **target** to an arbitrary YCB object will require a new grasp generator and validation; adding YCB **distractors** is the simpler first extension.

---

## 7. Create a new task environment (one task, different objects)

### Step A — Agree on a scene category

Use the same pick-and-place task with these **object-composition tags**:

| Category tag | Distractor asset examples | Suggested first implementation |
|---|---|---|
| `primitive` | Cubes, cuboids, cylinders | ManiSkill primitive builders |
| `custom_geometry` | Long/thin/flat/rounded/chamfered blocks, L-shapes | Parameterized primitive/mesh assets |
| `ycb` | Selected boxes, cans, bottles from YCB | Known ManiSkill YCB actors |
| `mixed` | Combination of above | A composition sampler |

These are **asset categories**, **not four separate manipulation skills**. Distractors may appear near the grasp, along the transport region, or in both places. Initially **keep the red 4-cm cube as target** while changing the distractor objects. Generalize the target after the baseline DP has reliable success.

### Step B — Create a separate environment module

For a **layout-only** variant that still uses four cube distractors, the smallest change is to subclass the baseline and override the position constants:

```python
# envs/candid_primitive_env.py
from mani_skill.utils.registration import register_env
from envs.candid_baseline_env import CANDIDBaselineEnv

@register_env("CANDIDPrimitive-v0", max_episode_steps=250)
class CANDIDPrimitiveEnv(CANDIDBaselineEnv):
    # Same robot, target, goal, camera, object count, and reward.
    # Only the distractor XY centers change here.
    DISTRACTOR_XY = (
        (-0.18,  0.10),
        (-0.04,  0.105),
        (-0.04, -0.11),
        (-0.18, -0.11),
    )
```

This example **only changes the fixed layout**. It does **not** introduce randomized shapes or positions. The inherited `_validate_baseline_layout()` checks this particular square-footprint layout. Validate all variants in 3D and in the physics simulator too.

For a **true object-geometry** variant, modify/override at least:

- `_load_scene()` to construct the desired movable distractor actors, including collision shapes and persistent identifiers;
- `_initialize_episode()` to reset **all** selected objects' poses and velocities;
- `_validate_baseline_layout()` or a suitable replacement to check size-dependent overlap and goal clearance;
- `get_distractor_poses_for_logging()` (or maintain the same shared actor-list interface) so the logger can track every distractor in a stable order.

**Keep the following compatibility names/API:** `self.cube` (the target), `self.goal_site`, `self.cube_half_size`, `self.clutter_cubes` (list of movable non-target actors), `evaluate()`, and `get_distractor_poses_for_logging()`. The present planner/logger depend on them. To use 2–6 distractors, the returned logging tensor must have shape `(num_envs, N, 7)` where `N` is the actual count.

**Note:** `DisturbanceMonitor` can aggregate a variable number of actors if `get_distractor_poses_for_logging()` returns a consistent tensor, but its current topple/fall heuristics are designed as **baseline approximations**. Tall bottles, irregular YCB items, and actors with different upright local axes require geometry-aware topple/fall checks.

### Step C — Register and select the new environment

Simply adding a file to `envs/` is not enough: Gymnasium registration occurs only when the file is **imported**, and the current runner hard-codes `CANDIDBaseline-v0`.

Recommended one-time extension to `scripts/run_baseline.py`:

```python
# 1. Import/register all environment classes you intend to launch:
from envs import candid_baseline_env, candid_primitive_env

# 2. Add inside parse_args():
parser.add_argument("--env-id", default="CANDIDBaseline-v0")

# 3. Replace the gym.make(...) expression inside build_env() with:
env = gym.make(
    args.env_id,
    obs_mode=args.obs_mode,
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    robot_uids="xarm6_robotiq",
    reward_mode="dense",
    sim_backend=args.sim_backend,
    max_episode_steps=args.max_episode_steps,
)

# 4. When creating the episode-metrics row, use:
"environment": args.env_id,  # instead of the hard-coded baseline ID
```

Remember to update experiment labels such as `source_desc` in `RecordEpisode` if they still say "fixed-scene baseline" when running randomized scene variants.

After this extension, an example call is:

```bash
python scripts/run_baseline.py \
  --env-id CANDIDPrimitive-v0 \
  --episodes 1 --obs-mode rgbd --save-video --verbose \
  --output-dir demos/primitive_smoke_test
```

**Until that extension is implemented**, the command above is **not supported**. Alternatively, run a dedicated variant-specific script that imports and creates its own registered environment.

### Step D — Add domain randomization (separate from fixed baseline)

Introduce a **scene configuration and reproducible sampler**, rather than putting untracked `np.random` / `torch.rand` calls in arbitrary parts of the code. Every generated scene should have a unique and saved `scene_id`.

Recommended sampled fields:

```yaml
scene_id: train_000123
category: mixed
seed: 123
robot: xarm6_robotiq
robot_base_xyz: [-0.522, 0.0, 0.0]
target:
  asset: standard_red_cube
  xyz: [-0.12, 0.0, 0.02]
goal_xy: [0.10, 0.0]
distractors:
  - id: distractor_0
    asset: custom_thin_cuboid
    xyz: [-0.17, 0.09, 0.04]
    yaw: 0.0
  - id: distractor_1
    asset: ycb_object_id_here
    xyz: [-0.02, -0.10, 0.05]
    yaw: 0.3
```

This is a **schema example, not a scene already generated by the current runner**. Replace placeholder asset identifiers and derive each object's Z placement from its actual collision shape/support geometry. Add any asset-specific bounding size, mesh scale, and fixture identity necessary to replay a scene exactly.

For the initial geometry-only CANDID experiments, randomize **observable geometry and arrangement**, not hidden mass/friction/center-of-mass. The physics simulator still needs consistent, physically sensible contact parameters; keep them fixed or governed by a documented deterministic asset rule.

Before accepting a sampled scene: (a) no object penetrates the tabletop, robot, target, or another distractor at reset; (b) no object lies in the protected goal landing region; (c) all relevant objects are visible/reasonably reconstructable; (d) the target and goal are kinematically reachable; (e) at least one complete no-rearrangement pick-and-place attempt succeeds in physical simulation. **IK feasibility alone is not proof of task feasibility.**

For a fixed `scene_id`, *all candidate trajectories must start from the exact same initial state*. Save the full scene configuration, object ordering, and seed in a manifest. This is essential for fair low-/high-disturbance comparisons.

**Example of a controlled trajectory comparison:** run three separate **one-episode** commands with `--seed 42` each time and `--lateral-offset` set to `0.0`, `+0.04`, and `-0.04`, storing them in three separate output directories. The baseline remains geometrically identical. For future random scenes, **replay the same saved scene configuration** for all candidate trajectories instead of relying on unrelated random resets.

---

## 8. Read the results (success, disturbance, and failure causes)

By default the runner creates (names vary with timestamps and recording options):

```text
demos/<experiment>/
├── baseline_YYYYMMDD_HHMMSS_microseconds.h5   # ManiSkill RecordEpisode
├── baseline_YYYYMMDD_HHMMSS_microseconds.json # Companion metadata
├── episode_metrics.jsonl                       # One JSON record per attempt
├── label_traces/
│   └── run_..._attempt_0000.npz                 # Distractor pose trace
└── *.mp4                                       # If --save-video
```

The exact HDF5 groups depend on ManiSkill version and selected `obs_mode`. Use `h5py` to inspect them rather than assuming a fixed key layout:

```bash
python - <<'PY'
from pathlib import Path
import h5py
files = sorted(Path('demos/candid_baseline').glob('*.h5'))
if not files:
    raise SystemExit('No HDF5 file: check --output-dir and --no-record')
with h5py.File(files[-1], 'r') as f:
    print('HDF5:', files[-1]); print('Episodes:', list(f.keys()))
    f.visititems(lambda name, obj: print(name, obj.shape) if isinstance(obj, h5py.Dataset) else None)
PY
```

Inspect per-attempt metadata:

```bash
python - <<'PY'
from pathlib import Path
import json
path = Path('demos/candid_baseline/episode_metrics.jsonl')
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
print('Attempts:', len(rows))
print('Successes:', sum(bool(row['task_success']) for row in rows))
for row in rows[-5:]:
    d = row['disturbance']
    print('seed=', row['seed'],
          'success=', row['task_success'],
          'stage=', row['last_stage'],
          'steps=', row['planner_steps'],
          'max_peak_translation_m=', d.get('max_peak_translation_m'))
PY
```

The `label_traces/*.npz` file contains **actual executed** distractor poses, with shape `(T+1, N, 7)` and first frame after reset. The current monitor also computes per-object **peak/residual translation and rotation**, approximate `ever_toppled`, and approximate `ever_fell_below_table` fields.

**Metric limitations:** the current code does **not** yet compute the proposal's normalized `D_max` / `D_mean`, calibrated Disturbance Rate (DR), Safe Success Rate (SSR), or proper off-table fall detection. `max_peak_translation_m` is a raw displacement, **not** the final CANDID disturbance metric. For tall bottles/YCB shapes, update the topple/fall logic before treating those labels as ground truth.

When analyzing, report at least: task success rate, stage/failure reason counts, peak translation (metres), residual translation, rotation, and severe events; later add `D_max`, `D_mean`, DR/SSR, and latency. Compare different planners/policies on **matched initial scene configurations**, not unrelated random seeds.

---

## 9. Data collection: DP demonstrations vs predictor supervision

We use **one physical simulator** but **different sample-selection rules**.

| Data pool | Sources | Which rollouts to keep? | Training purpose |
|---|---|---|---|
| **A. Vanilla DP demonstrations** | Successful planner/IK pick-and-place executions | **Successful complete episodes only**, including low- and moderate-/high-disturbance successes | Learn task competence (imitation learning) |
| **B. Disturbance predictor data** | Nominal + waypoint-perturbed planner executions | **All physically executed, correctly labeled chunks**, from successes and failures | Learn continuous consequences of candidate actions |
| **C. Frozen Vanilla DP rollouts** | Actual rollouts of the trained, frozen task-only DP | Success and failure chunks | Adapt predictor to policy-generated action distribution |

**Do not** train DP on motion-planning failures, missed grasps, or dropped-target episodes. **Do not** filter predictor data to successes only. Planning failures before any simulation action cannot provide valid physical action-consequence labels.

### Pool A — Task-only DP demonstrations

To collect **only successful** HDF5 demonstrations from the current fixed baseline:

```bash
python scripts/run_baseline.py \
  --episodes 10 --seed 0 \
  --obs-mode rgbd --save-success-only \
  --output-dir demos/dp_baseline_debug
```

This is a **recording sanity check only**. The ten identical baseline scenes are **not** ten diverse layouts. The fixed planner is allowed to use exact simulator target pose to plan *teacher actions*; saved student observations must still be camera/proprio-only.

When randomized scenes exist, collect several distinct **successful** trajectories per scene by varying grasp approaches, lift heights and feasible transport waypoints. Avoid keeping only near-identical demonstrations. Do not add a disturbance penalty to the **Vanilla** DP imitation loss.

### Pool B — Continuous disturbance data

Collect all executed attempts, including failed ones (do **not** pass `--save-success-only`):

```bash
python scripts/run_baseline.py \
  --episodes 10 --seed 0 \
  --obs-mode rgbd \
  --output-dir demos/predictor_baseline_debug
```

The monitor writes per-control-step object pose traces to separate `.npz` files. For the actual predictor dataset, also record an unambiguous `scene_id`, source trajectory ID, action/control mode, asset manifest, and timestamp alignment. **The current runner does not yet write all of this metadata automatically.**

Use controlled perturbations of a **reliable nominal trajectory**, not arbitrary independent noise added to every timestep. The present `--lateral-offset` is just **one example** of a planned trajectory variation. A later trajectory generator should vary valid grasp poses, approach, lift height, transfer waypoints, and carried-target orientation while keeping the task unchanged.

A good scene has *multiple trajectories with different disturbance outcomes*; especially valuable are pairs where **both complete the pick-and-place successfully** but one causes more disturbance.

### Pool C — Frozen DP rollouts

After training a competent Vanilla DP, **freeze its weights**, execute it on **training scenes only**, save its realized actions and observations, and append these action-consequence samples to the predictor dataset. **This functionality is not yet implemented by `run_baseline.py`**; it requires a separate policy rollout collector.

Do not use test-scene rollouts for training the DP or predictor.

### Chunk-level predictor labels (target design)

At policy time `t`, form an input from past camera/robot observations and a proposed future `H`-step action chunk; label it with **the consequence of those executed actions**, not the whole episode's final disturbance:

```text
Input : observations[t-history+1 : t+1] + actions[t : t+H]
Label : resulting distractor movement/rotation and new topple/fall events
```

Use **`H=12` as an initial design reference**, following the chosen chunk-level setup; choose history length and actual action execution frequency **consistently across the DP and predictor**. The current MPlib trajectory files record `pd_joint_pos` actions; they are not automatically DP-ready 12-step training windows.

The logger's `pose_trace` records the environment **after each control step**, so labels can be derived from intervals `[t, t+H]`, while respecting episode boundaries. For a monotone running-peak metric, one possible chunk target is `running_peak(t+H) - running_peak(t)`. Also retain raw per-object translation, rotation and severe-event labels. **Do not confuse a final episode score with an individual chunk target.**

Real-robot deployable inputs must not include simulator actor poses, exact collision forces, hidden friction/mass, or future realized states. Those are **labels/diagnostics**, not predictor inputs.

---

## 10. Suggested collection schedule and data splits

These are **planning targets** agreed for a first research pass, **not a dataset already produced** or a guarantee of model performance.

| Stage | Planned scope | Primary objective |
|---|---|---|
| **Pilot** | `20` geometry scenes × `10` candidate attempts = **200 attempts** | Test reset, planning feasibility, video QA, label correctness and disturbance diversity |
| **DP training pool** | Aim for ~**800 successful, diverse demonstrations** | Train and freeze a competent Vanilla DP |
| **Predictor initial training pool** | `400` training geometry scenes × `10` candidate attempts = **4,000 attempts** | Cover zero/low/high disturbance and failures |
| **Predictor validation** | `50` unseen scene geometries × `8` attempts = **400 attempts** | Tune/pick the predictor without test leakage |
| **Frozen DP data** | ~**800–1,200 policy rollouts** on training scenes | Cover the real policy's action distribution |
| **Final test** | **100 held-out geometry scenes** | Evaluate Vanilla DP, guided CANDID and matched baselines |

The 400 training scenes can initially be distributed by **object-composition category**, not by grasp/transfer stage:

| Scene composition | Suggested training scenes |
|---|---:|
| Primitive shapes | 120 |
| Custom geometry | 120 |
| Selected YCB objects | 80 |
| Mixed objects | 80 |
| **Total** | **400** |

For each `scene_id`, try several meaningful trajectory variants. Stop or adjust if almost every attempted trajectory fails before contacting the environment, or if successful trajectories have indistinguishable disturbance. The pilot should decide whether `10` attempts per scene is adequate.

**Split by `scene_id`/geometry layout before generating trajectories.** Do not put different trajectories from the same initial geometry in train and validation/test. For a genuine **unseen-object OOD** split, hold out selected custom shapes or YCB object identities from *all* training scenes, including mixed scenes. Keep the baseline scene separate from the held-out test allocation.

Store at least these manifest fields with each scene/episode: `split`, `scene_id`, category, asset identifiers, object count and stable ordering, dimensions and initial poses, seed, goal, camera configuration, control mode, trajectory generator/parameters, action timestamp resolution, success, disturbance trace path, and code/asset version.

---

## 11. Troubleshooting and common mistakes

| Symptom | Check / fix |
|---|---|
| `ModuleNotFoundError: candid_baseline_env` | File now lives under `envs/`. Use `from envs import candid_baseline_env` and run from project root. |
| `NameNotFound` / unknown Gym ID | Import the module that defines `@register_env(...)` before `gym.make()`. The current runner only creates `CANDIDBaseline-v0` until extended. |
| Planner fails at `pregrasp`, `approach`, `transfer`, or `lower` | Inspect `last_stage` in JSONL; check IK reachability, target/goal pose, collision layout and waypoint selection. Try nominal route first. |
| `success=False` but target looks near goal | Check actual cube XY, height, release state and robot stillness; these are all evaluated. |
| High planner success, zero disturbance everywhere | Baseline may simply avoid all contacts. Introduce controlled geometry/trajectory variation; do not fabricate labels. |
| `--episodes 10` gives nearly identical trajectories | The reference scene/route are intentionally fixed. A different seed alone does not create scene diversity. |
| `--obs-mode none` works but MP4 fails | Check CPU/Vulkan/rendering availability independently; do not assume planner or HDF5 failure. |
| No MP4 appears | Re-run with `--save-video`, inspect the specified `--output-dir`, and check rendering logs. |
| `videos/` appears only inside Docker | Bind-mount `/workspace/videos` or use `docker cp` from the host. |
| Logger fails after changing number of objects | Keep `clutter_cubes` and `get_distractor_poses_for_logging()` consistent; stable object ordering is mandatory. |
| YCB object incorrectly reported as toppled/fallen | Current orientation/height heuristics are for basic upright objects. Add object-specific support geometry and true off-table tests. |
| HDF5 observations accidentally contain object ground truth | Do not add simulator `cube.pose`/distractor poses to `_get_obs_extra`; use `record_env_state=False`. Inspect saved keys before training. |
| DP cannot execute saved motion-planner actions | Check action/control representation. Baseline HDF5 actions use **`pd_joint_pos`**, not EE delta pose. |
| Train/validation appears unrealistically easy | Check that different trajectories from the same scene or identical YCB instances are not split across datasets. |

---

## 12. Team handoff and review checklist

A new scene variant is ready for other contributors **only when**:

- [ ] It uses the shared robot, coordinate frame, baseline task success definition and documented control mode.
- [ ] It has a distinct environment ID and registered module; running it does not overwrite the baseline.
- [ ] Its scene configuration has a stable `scene_id`, repeatable reset, valid tabletop contacts and no initial overlap.
- [ ] It preserves `cube`, `goal_site`, `clutter_cubes` and disturbance-logging interfaces, or updates their consumers together.
- [ ] It passes a one-episode `--obs-mode none` smoke test **and** a rendered RGB-D/video check.
- [ ] One can visually confirm grasp, transfer, place, release and distractor consequences.
- [ ] Both success/failure metadata and per-step pose traces are retained for debugging.
- [ ] No ground-truth simulator object state enters the deployable DP/Predictor observations.
- [ ] At least a few genuinely different trajectories are executed from **identical** initial scene states.
- [ ] Data, seeds, asset IDs, code version and metrics can be traced back to a reproducible experiment.
- [ ] Raw videos/HDF5 datasets are kept out of Git unless the team deliberately adopts a data hosting solution.

**Recommended collaboration workflow:** keep `envs/candid_baseline_env.py`, `planners/execute_baseline.py` and the base runner stable as shared references. Develop new shapes/scenes in separate environment modules; add trajectory-generation and collection features via small, reviewed interfaces. If a shared API must change (object count, observation keys, action mode, metric definition), agree on it before generating large datasets.

---

## 13. Implementation references

- [ManiSkill official repository](https://github.com/mani-skill/ManiSkill)
- [ManiSkill xArm6 PickCube planner example](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/examples/motionplanning/xarm6/solutions/pick_cube.py)
- [ManiSkill xArm6 motion planner](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/examples/motionplanning/xarm6/motionplanner.py)
- [ManiSkill `RecordEpisode` implementation](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/utils/wrappers/record.py)
- [GRITS: Spillage-Aware Guided Diffusion Policy](https://github.com/HCIS-Lab/GRITS) — reference for separating task demonstrations from consequence-prediction data; **CANDID is still a different task with continuous non-target disturbance labels**.

**Bottom line:** first confirm a runnable, reproducible scene; then compare *actual executed* alternative trajectories in that same scene; save camera/proprio observations as policy data and simulator object motion as separate supervision. Only scale up after the pilot demonstrates useful variation in disturbance outcomes.
