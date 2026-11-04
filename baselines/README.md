# ManiSkill ACT / BC / PPO / SAC baselines for CANDID

These are **optional state-based comparison methods**. They do not alter or
replace `model/vanilla_dp`, the disturbance predictor, or `src/candid`.
The base implementations are from [ManiSkill](https://github.com/haosulab/ManiSkill)
at commit `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` (Apache-2.0; see `baselines/LICENSE-APACHE-2.0`).
Network architectures, learning objectives and optimization are kept from the
corresponding official `examples/baselines`. CANDID-specific changes cover
environment registration, canonical observations, CPU compatibility, HDF5
export, ACT normalization/EMA, checkpoints and common evaluation.

## Contracts and requirements

- Robot: xArm6 + Robotiq; actions: 7-D absolute joint position targets
  (`pd_joint_pos`).
- Environment observations: `state_dict`. All four policies receive the
  same flat CANDID feature order defined by
  `candid.data.adapter.OBSERVATION_FIELDS`.
- BC and ACT: imitation learning from successful demonstrations in **train
  scenes only**. PPO and SAC: **online** training using the environment's
  standard dense reward (not the HDF5 demonstrations).
- Evaluation: `baselines/common/eval_baseline.py` invokes the existing
  `candid.eval.rollout` evaluator and reports success, safety, disturbance,
  topple and fall on the frozen TEST split.
- No disturbance-aware loss or safety penalty is added to these baseline
  algorithms.

Install the project `requirements.txt` in an already functional ManiSkill /
SAPIEN environment. The baseline-specific direct dependencies are
`tyro`, `tensorboard` and `tqdm`. No CUDA is required for the tested
single-environment CPU path. ACT uses the state-only Transformer; the unused
image backbone is intentionally not included (and `torchvision` is not
required for these state-based baseline runs).

Run commands from the repository root:

```bash
cd /workspace/CANDID
export PYTHONPATH="$PWD:$PWD/src"
```

## Choose a real environment

Use an existing registered CANDID environment and supply both its Gym ID
and the module that registers it. For example, the tracked default is:

```bash
export ENV_ID=CANDIDBaseline-v1
export ENV_MODULE=envs.candid_baseline_env
```

For the local `ring_120` dataset, use the **actual** environment pair
recorded in `demos/ring_120/scenes.jsonl`. The Ring environment used for
the November 2026 tests was `CANDIDRing-v1` /
`envs.candid_ring_env`. **That environment source module is not included
in this baseline-only PR**: it must be separately committed/available on
any machine that trains online or replays Ring scenes. A dataset alone
does not register a Gym environment.

```bash
export ENV_ID=CANDIDRing-v1
export ENV_MODULE=envs.candid_ring_env
python -c "import importlib; importlib.import_module('$ENV_MODULE')"
```

## Convert demonstrations for BC and ACT

The exporter creates a ManiSkill-style `traj_N` HDF5 plus a `.json` manifest.
It exports *only frozen TRAIN scenes*, converts raw observations into the
canonical state vector, and checks state/action alignment and dimensions.
Existing exports are never overwritten.

For an extremely small CPU smoke run:

```bash
python baselines/common/export_demos.py \
  --dataset demos/ring_120 \
  --output demos/ring_smoke_train.h5 \
  --max-trajectories 2
```

For actual imitation training, omit `--max-trajectories` and choose a
new output path. The manifest lists the original source environments:
its `env_info.env_id` is only a representative source, not proof that
a merged dataset came from one Gym environment.

## Minimal CPU train-and-checkpoint test

These commands deliberately do **almost no training** and disable
training-time online evaluation (`--eval-freq 0`). They verify the interface,
one optimizer update and checkpoint saving, not policy quality.

**BC — 1 optimizer update**

```bash
python baselines/bc/bc.py \
  --env-id "$ENV_ID" --env-module "$ENV_MODULE" \
  --demo-path demos/ring_smoke_train.h5 \
  --control-mode pd_joint_pos --sim-backend cpu --no-cuda \
  --num-demos 2 --batch-size 2 --total-iters 1 \
  --max-episode-steps 10 --num-eval-envs 1 \
  --eval-freq 0 --save-freq 1 --log-freq 1 \
  --no-capture-video --exp-name ring_smoke_bc
```

Checkpoint: `runs/ring_smoke_bc/checkpoints/0.pt`.

**ACT — 1 optimizer update (reduced model, smoke only)**

```bash
python baselines/act/train.py \
  --env-id "$ENV_ID" --env-module "$ENV_MODULE" \
  --demo-path demos/ring_smoke_train.h5 \
  --control-mode pd_joint_pos --sim-backend cpu --no-cuda \
  --num-demos 2 --batch-size 2 --total-iters 1 \
  --num-queries 2 --hidden-dim 64 --nheads 2 \
  --enc-layers 1 --dec-layers 1 --dim-feedforward 128 \
  --max-episode-steps 10 --num-eval-envs 1 \
  --eval-freq 0 --save-freq 1 --log-freq 1 \
  --no-capture-video --exp-name ring_smoke_act
```

Checkpoint: `runs/ring_smoke_act/checkpoints/0.pt`. Remove these
tiny-architecture overrides when training the regular ACT baseline.

**PPO — 4 CPU environment steps**

```bash
python baselines/ppo/ppo.py \
  --env-id "$ENV_ID" --env-module "$ENV_MODULE" \
  --control-mode pd_joint_pos --sim-backend cpu --no-cuda \
  --num-envs 1 --num-eval-envs 1 --num-steps 4 \
  --num-minibatches 2 --update-epochs 1 --total-timesteps 4 \
  --eval-freq 0 --no-capture-video --exp-name ring_smoke_ppo
```

Checkpoint: `runs/ring_smoke_ppo/final_ckpt.pt`.

**SAC — 4 CPU environment steps**

```bash
python baselines/sac/sac.py \
  --env-id "$ENV_ID" --env-module "$ENV_MODULE" \
  --control-mode pd_joint_pos --sim-backend cpu --no-cuda \
  --num-envs 1 --num-eval-envs 1 --buffer-device cpu \
  --buffer-size 64 --batch-size 4 --learning-starts 1 \
  --training-freq 4 --utd 0.25 --total-timesteps 4 \
  --eval-freq 0 --log-freq 4 --no-capture-video \
  --exp-name ring_smoke_sac
```

Checkpoint: `runs/ring_smoke_sac/final_ckpt.pt`.

Run `baselines/common/smoke_actions.py` with each method/checkpoint
to load the model and save one finite, 7-dimensional predicted action
from an exported demonstration observation. Example:

```bash
python baselines/common/smoke_actions.py \
  --method bc --demo-path demos/ring_smoke_train.h5 \
  --checkpoint runs/ring_smoke_bc/checkpoints/0.pt \
  --output runs/ring_smoke_bc/actions.npy
```

## Common closed-loop evaluation

Evaluate on the **TEST scenes** in the source dataset, not the policy
training rollouts. Run the following for each of `bc`, `act`, `ppo`, `sac`
using that method's checkpoint.

```bash
python baselines/common/eval_baseline.py \
  --method bc --dataset demos/ring_120 \
  --checkpoint runs/ring_smoke_bc/checkpoints/0.pt \
  --output-dir runs/ring_closedloop_bc \
  --safe-dr-mean-max 0.2 --max-scenes 1 --max-steps 10 --device cpu
```

`0.2` is an **illustrative smoke-test threshold**, not an approved safety
threshold. The evaluator uses the saved scene environment metadata and
simulator snapshot; the source environment modules must be present.
ACT temporal aggregation requests a new action chunk each step;
the common evaluator handles it using a 1-step execution horizon.

## Validation, status and limitations

```bash
python -m pytest tests/test_baselines_contract.py -v
python -m compileall -q baselines
```

On **2026-11-04**, all four methods were exercised in the user's
single-environment CPU Ring setup: optimizer update, checkpoint reload,
finite 7-D action prediction and 10-step closed-loop replay with standard
CANDID metrics. All four smoke rollouts reported **no task success**,
as expected for essentially untrained models. This verifies plumbing,
**not** convergence, safe operation, or comparative performance.

For a fair benchmark, use more trajectories/steps, tune training
parameters, record wall-clock and environment interaction budgets, and
evaluate the same unseen test scenes with a shared DR/safety definition.
In particular, random online PPO/SAC resets are **not constrained to
the recorded train split**; their training geometry must be explicitly
separated from test geometry before claiming unseen-scene generalization.
Built-in ACT/BC training-time evaluations are diagnostics, not the
frozen TEST evaluation, and successful validation-based checkpoint
selection is not yet implemented uniformly across all four methods.
