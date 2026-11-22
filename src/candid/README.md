# CANDID Core Library (`src/candid/`)

This directory contains the shared simulation, physical-data, data-preparation and closed-loop evaluation code for **CANDID: Consequence-Aware Non-target Disturbance-Informed Diffusion**. It is the reusable implementation layer below the repository's CLI wrappers; the neural-network architectures and training code live separately in `model/`.

For **installation and the complete end-to-end workflow**, start at the [root README](../../README.md). For **all command-line entrypoints**, see the [scripts guide](../../scripts/README.md).

> **Version:** This document describes the current `main` branch, including physical counterfactual replay and frozen multi-environment evaluation—not the earlier collection-only v1 implementation. Runtime data, checkpoints and simulation assets are not distributed in this directory.

## 1. Architectural overview

~~~text
envs/*.py + SceneSpec                     (environment definitions)
          |
          v
src/candid/sim/                           (SAPIEN/ManiSkill physical world)
          |
          +----> planners/execute_baseline.py
          |      + src/candid/runner/       (teacher demonstration selection)
          |                 |
          |                 v
          +---------> src/candid/data/      (HDF5, trace NPZ, labels, splits)
                             |
                  +----------+-----------------------------+
                  |                                        |
                  v                                        v
          model/vanilla_dp/                       model/predictor/
          (state DDPM policy)                     (ordinal risk predictor)
                  |                                        ^
                  |                            physical counterfactual replay
                  +------------------------+               |
                                           v               |
                                 model/guidance/  <---------+
                                 (frozen DP + risk gradients)
                                           |
                                           v
                                  src/candid/eval/
                                  (paired physical rollouts)
~~~

The boundaries matter:

- `sim/` defines scene construction, physics reset, object settling, robot state, observations and task success. It does **not** train neural networks.
- `runner/` orchestrates teacher trajectories from a common saved scene and chooses alternative motion-planner configurations.
- `data/` owns physical traces, disturbance/event labels, data serialisation, canonical observations, training windows, and leakage-resistant scene splits.
- `eval/` restores frozen physical states, calls a supplied policy, executes actions and computes comparable episode-level results.
- `model/` (at repository root) owns the trained Diffusion Policy, Predictor, Guidance sampling and reranking code. It is intentionally separate from `src/candid/`.

## 2. Package layout

~~~text
src/candid/
├── __init__.py
├── sim/
│   ├── spec.py                  # XYRegion, ObjectSpec, SceneSpec
│   ├── actors.py                # Primitive / YCB actor construction and geometry
│   ├── layout.py                # Seeded non-overlapping tabletop placement
│   ├── distractor_selection.py  # Select deterministic distractor subsets
│   ├── settling.py              # Physical scene-settling checks
│   └── environment.py           # CANDIDTabletopEnv, xArm6 task
├── runner/
│   ├── collect.py               # General / legacy planner collection helper
│   ├── adaptive_sampling.py     # Deterministic warmup and candidate generation
│   ├── feedback_sampler.py      # History-informed teacher alternatives
│   ├── production_sampler.py    # Default feedback/uniform and route samplers
│   ├── route_intents.py         # Direct / left / right / over TCP waypoints
│   └── variants.py              # Planner-configuration utilities
├── data/
│   ├── adapter.py               # Named state_dict -> canonical 151-D vector
│   ├── windows.py               # Model-neutral observation / action windows
│   ├── monitor.py               # PhysicsTrace of target/distractor motion
│   ├── metrics.py               # Disturbance, topple/fall and angle calculations
│   ├── storage.py               # DatasetWriter, scene snapshots, JSONL/NPZ
│   ├── trajectories.py          # Read expert trajectories from manifests/HDF5
│   ├── splits.py                # Frozen, fingerprinted scene-level split.json
│   ├── validate.py              # Dataset and T/T+1 consistency verification
│   ├── counterfactual.py        # Expert probes and bounded action perturbations
│   ├── balanced_counterfactual.py  # Stratified CF state selection
│   ├── replay_diagnostics.py    # Archive-versus-physics replay error analysis
│   └── synthetic.py            # Test / synthetic-data helpers
└── eval/
    ├── policy_api.py            # Minimal policy interaction contract
    ├── metrics.py               # SafetyLimits and episode aggregation
    ├── replay_env.py            # Restore saved scene's registered environment
    ├── rollout.py               # Closed-loop episodes; shared policy evaluator
    ├── local_dr.py              # Local disturbance diagnostics
    ├── local_dr_repeat.py       # Repeated local-risk diagnostics
    ├── generate_suite.py        # Generate new frozen evaluation-only scenes
    └── scene_suite.py           # Suite fingerprint and integrity checks
~~~

## 3. Simulation: `sim/`

### Scene definitions

A scenario is a `SceneSpec` containing a target `ObjectSpec`, a set of distractor specifications and a goal region. A thin environment module in **repository-root `envs/`** registers that scene with ManiSkill.

For an example of an actual supported definition, inspect [`envs/candid_baseline_env.py`](../../envs/candid_baseline_env.py); to add a new scene, follow the same pattern:

~~~python
from mani_skill.utils.registration import register_env
from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import SceneSpec, ObjectSpec, XYRegion

TARGET = XYRegion((-0.145, -0.095), (-0.035, 0.035))
CLUTTER = XYRegion((-0.180, -0.040), (-0.095, 0.095))
GOAL = XYRegion((0.080, 0.120), (-0.040, 0.040))

SCENE = SceneSpec(
    name="example_cubes",
    target=ObjectSpec("target_cube", "box", TARGET, (-0.12, 0.0),
                      size=(0.04, 0.04, 0.04)),
    distractors=(
        ObjectSpec("box_a", "box", CLUTTER, (-0.15, 0.07),
                   size=(0.04, 0.04, 0.04)),
    ),
    goal_region=GOAL,
    goal_fixed_xy=(0.10, 0.0),
)

@register_env("CANDIDExampleCubes-v1", max_episode_steps=500)
class CANDIDExampleCubesEnv(CANDIDTabletopEnv):
    SCENE = SCENE
~~~

After adding `envs/candid_example_cubes_env.py`, the runtime needs both `--env-id CANDIDExampleCubes-v1` and `--env-module envs.candid_example_cubes_env` to register the environment.

`CANDIDTabletopEnv` uses xArm6 + Robotiq and is designed around a **single CPU simulation environment** with `pd_joint_pos` actions and `state_dict` observations. It builds primitives/YCB actors, seeds and settles physical layouts, exposes robot/target/distractor states and evaluates pick-and-place success. The standard cube-target motion planner is **not** guaranteed to grasp every complex YCB object.

### Scene reproducibility and settling

The production collector resets each scene once, lets it settle, saves the complete simulator snapshot and reuses that snapshot for all teacher alternatives. The environment's `SceneSpec`, actor dimensions, initial poses, controller configuration and Python module are part of the replay contract. Reproducing the random seed alone is insufficient if the environment module or physics implementation changes.

`distractor_selection.py` permits reproducible subsets of the declared scene's distractors. `settling.py` and the environment's reset checks prevent silently recording substantial initial motion. **Do not loosen settling or archive-replay tolerances just to make an invalid scene pass.**

## 4. Physical data: `data/`

### Recorded dataset contract

A collection is indexed by `scenes.jsonl` and `trajectories.jsonl` and references:

~~~text
demos/example/
├── scenes.jsonl
├── trajectories.jsonl
├── scenes/<scene_id>.npz
├── disturbance_traces/<trajectory_id>.npz
└── trajectories_<run_id>.h5
~~~

The HDF5 recorder stores state observations and executed actions; a physical trace stores object poses, sizes, velocities and event flags. For `T` actions and `N` distractors:

| Trace field | Shape |
|---|---|
| `distractor_poses` | `[T+1, N, 7]`, pose `[x,y,z,qw,qx,qy,qz]` |
| `distractor_linear_velocities` | `[T+1, N, 3]` |
| `distractor_angular_velocities` | `[T+1, N, 3]` |
| `object_sizes_m` | `[N]` |
| `distractor_toppled` / `distractor_fallen` | `[T+1, N]` Boolean |

The first trace frame is taken after reset/settling, before the first action. The data validator checks reference integrity, dimensions and temporal alignment; it does **not** prove the recorded motions are optimal or safe.

### Disturbance and event definitions

For object `i`, starting at a chosen reference frame:

~~~text
DR_i = 0.7 * peak_translation_i / characteristic_size_i
     + 0.3 * peak_rotation_angle_i / pi

DR_mean = mean_i DR_i
DR_max  = max_i  DR_i
~~~

`metrics.py` supports both episode-level DR (relative to the initial state) and **chunk-local** DR (relative to the candidate chunk's first frame). Topple and fall are geometry-based event indicators. The model's local eight-step risk estimate must not be confused with its final **whole-episode** safety metrics.

### Canonical observations and window lengths

`adapter.py` reads named `state_dict` fields (robot qpos/qvel, TCP pose, target/distractor states, masks and goal position) and converts them to a **151-D** state vector for the supported six-distractor xArm6 task. Policies operate on a **7-D** absolute joint-position/gripper action vector.

- Diffusion Policy: observation horizon **2**, prediction horizon **16**, action horizon **8**.
- Ordinal and CF Predictors: two observation frames, eight candidate actions.
- `windows.py` is a **generic utility** whose historical default action horizon is **12**. Production Predictor loaders explicitly request horizon **8**; the Vanilla DP loader implements its own horizon-16 windows. Do not infer the trained model's horizon from the generic utility default.

### Scene-level data splits

`splits.py` freezes one manifest-fingerprinted `split.json` per dataset; default fractions are **70% train / 15% validation / 15% test**. Multiple trajectories from the same physical scene/layout must never be split independently into training and test.

~~~bash
# Execute from the repository root with PYTHONPATH configured:
python - <<'PY'
from candid.data.splits import load_or_create_scene_split
split = load_or_create_scene_split("demos/example")
print({name: len(split[name]) for name in ("train", "validation", "test")})
PY
~~~

Subsequent loads validate the original dataset fingerprint. Intentional dataset changes require a **new experiment** and a new split, not silent reshuffling.

### Physical counterfactual data

`counterfactual.py` and `balanced_counterfactual.py` choose candidate states **from the source training scenes**, create bounded action perturbations and balance reference-state categories. The CLI `scripts/generate_counterfactuals.py` performs the actual, separate SAPIEN replays. Each candidate must be physically relabelled after resetting to the archived initial scene and replaying the expert action prefix; do **not** substitute expert labels for unexecuted perturbed actions.

`replay_diagnostics.py` analyses state/pose discrepancies when archived and replayed physics disagree. It does not waive consistency requirements. See [counterfactual collection](../../docs/COUNTERFACTUAL_DATA_V1.md) and [CF Predictor training](../../docs/COUNTERFACTUAL_PREDICTOR_TRAINING.md).

## 5. Teacher sampling: `runner/`

The **canonical** production entrypoint is [`scripts/collect.py`](../../scripts/collect.py). `runner/collect.py` is the older general-purpose runner used by `scripts/run_baseline.py`, not the canonical feedback/route-intent sampler.

There are two different collection modes:

| Mode | Implementation | Description |
|---|---|---|
| Default | `ProductionSampler`, `adaptive_sampling.py`, `feedback_sampler.py` | 13 initial warmups, then feedback/uniform scene-local teacher alternatives |
| `--route-intents` | `RouteIntentSampler`, `route_intents.py` | Four transfer-route intentions: **direct**, **left**, **right**, **over** |

Each teacher proposal must be executed and observed before the next proposal is chosen. Under either mode, all attempts from one scene restart from the same physically saved snapshot. The route-intent sampler operates on TCP waypoints; it is **not** a whole-arm collision guarantee.

~~~bash
python scripts/collect.py \
  --env-id CANDIDDenseClutter-v1 \
  --env-module envs.candid_dense_clutter_env_random \
  --scenes 2 --trajectories-per-scene 24 \
  --route-intents --no-collision-aware \
  --output-dir demos/route_demo

python scripts/validate_dataset.py demos/route_demo
~~~

The optional collision-aware reference is an **extra** planner attempt; `--no-collision-aware` disables that attempt. Dataset manifests retain failed proposals where possible, including zero-action failures that have no HDF5 trajectory group.

## 6. Closed-loop evaluation: `eval/`

`generate_suite.py` creates a set of **evaluation-only** random scenes from a chosen registered environment; `scene_suite.py` fingerprints and validates its saved metadata and snapshots. Independent policies should evaluate **the same immutable suite**.

`rollout.py` is policy-agnostic: it restores a saved scene, builds canonical observations, asks the policy for a chunk, clips actions to the environment's actual bounds, executes the requested subset and repeats until termination or the episode limit. It logs individual episodes and summarises task success, safety, DR and topple/fall through `eval/metrics.py`.

`replay_env.py` resolves the original environment module and can reconstruct certain simple SceneSpec-only environments for legacy datasets. **Frozen random evaluation suites require the original module**, with a matching SceneSpec; do not use a fallback that changes simulation physics.

~~~bash
# Create a frozen scene suite.
python scripts/generate_eval_scenes.py \
  --env-id CANDIDDenseClutter-v1 \
  --env-module envs.candid_dense_clutter_env_random \
  --num-scenes 10 --seed 900001 \
  --output-dir eval/dense10

# Replay a compatible trained DP; replace the checkpoint path with yours.
python scripts/eval_dp.py \
  --eval-suite eval/dense10 \
  --checkpoint runs/dp/checkpoints/best_val.pt \
  --output-dir runs/vanilla_eval \
  --safe-dr-mean-max 0.2 \
  --max-steps 300 --exec-horizon 4 --no-video
~~~

To evaluate learned CANDID guidance instead, use `scripts/eval_dp_guidance.py` with both DP and Predictor checkpoints and explicitly set the guidance schedule. The [root README](../../README.md#8-run-inference-and-compare-policies) contains the complete SNR-Gaussian Capped command and a matching Best-of-N baseline.

### Evaluation outputs and integrity

- `summary.json`: success, safe success, safety, mean physical DR and event counts.
- `episodes.jsonl` and `episodes.csv`: per-scene/per-seed outcomes.
- Per-episode trajectory trace and optional visualisation / policy diagnostics.
- `eval_suite_sha256`: identifies the exact frozen evaluation scene set.

`scripts/compare_eval_runs.py` checks that paired evaluations agree on suite checksum, safety limits, scene/seed identities, horizon and other key protocol settings. Do not compare unrelated initial scenes as if they were matched trials.

## 7. Extending the library safely

1. **New scene or distractor types:** normally add a new thin `envs/*.py` module with a unique registered Gym ID and a valid `SceneSpec`. Change `sim/actors.py` only for genuinely new actor types.
2. **New collection strategy:** work in `runner/` and keep the dataset contract unchanged; avoid duplicating physics or DR calculations in a script.
3. **New risk metric:** update `data/metrics.py` and the corresponding evaluator/CF consumers deliberately; distinguish local labels from episode outcomes.
4. **New model:** add it under repository-root `model/`, keep actions and observation normalisation compatible, and implement the evaluator's policy interface.
5. **New evaluation scene:** use an immutable separate suite and preserve the defining environment module, seed and SceneSpec. Never evaluate on the CF training scenes and call the result OOD.
6. **New output format:** update `storage.py` and `validate.py` together and add a data-contract test.
7. **Before merging:** run focused tests and then physical simulator checks against the environment, dataset and policy whose behaviour changed.

~~~bash
export PYTHONPATH="$PWD:$PWD/src"
python -m pytest \
  tests/test_contract.py tests/test_disturbance_events.py \
  tests/test_scene_splits.py tests/test_counterfactual.py \
  tests/test_predictor_counterfactual.py tests/test_random_eval_suite.py \
  tests/test_guidance.py tests/test_guidance_snr.py -q
~~~

**Important:** a passing Python unit test is not a substitute for a successful physical SAPIEN replay. Conversely, successful scene replay is not evidence that CANDID improves safety across unseen distributions.

Additional guides: [scripts](../../scripts/README.md), [root quick start](../../README.md), [frozen evaluation](../../docs/RANDOM_EVAL.md), [paired visualisation](../../docs/paired_trajectory_visualization.md).
