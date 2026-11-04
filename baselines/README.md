# ManiSkill ACT, BC, PPO and SAC baselines for CANDID

These are **state-based upstream ManiSkill implementations**, adapted to the CANDID
data contract. They are not new algorithms and do not include disturbance-aware
guidance. All final results must use CANDID's **frozen test scenes** and shared
`candid.eval.rollout` metrics.

## Provenance

- Original code: https://github.com/mani-skill/ManiSkill
- Pinned upstream commit: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`
- License: Apache-2.0 (see repository `LICENSE`).
- ACT: `examples/baselines/act/train.py` and `act/detr/*`
- BC: `examples/baselines/bc/bc.py`
- PPO: `examples/baselines/ppo/ppo.py` (CleanRL-style implementation)
- SAC: `examples/baselines/sac/sac.py` (CleanRL-style implementation)

Core losses, networks, and optimization remain from ManiSkill. CANDID changes
include env registration, observation adaptation, control mode, CPU defaults,
dataset export, ACT's broken upstream stats method, BC normalization-at-eval,
action-space-aware SAC random exploration, and a shared test evaluator.
**Do not cite any benchmark numbers until end-to-end ManiSkill simulator tests
have been performed.**

## Shared interface and fairness

- Robot: `xarm6_robotiq` (environment default, verify when collecting).
- Control: `pd_joint_pos`, 7 continuous absolute joint targets.
- Raw env observations: `state_dict`; policies receive a flat vector in the
  exact `candid.data.adapter.OBSERVATION_FIELDS` order.
- Simulator: CPU supported; optional GPU for PPO/SAC after separate verification.
- Evaluation: `baselines/common/eval_baseline.py`, using the same
  `split.json`, DR, topple/fall, success/safety metrics as Vanilla DP.
- BC and ACT imitate successful **train-scene** demonstrations. PPO and SAC are
  trained through online environment interactions with CANDID dense rewards,
  NOT through offline demonstrations.
- Keep reward training as CANDID's standard task reward; do **not** inject
  disturbance penalties into the baselines unless explicitly reporting a
  different ablation.
- CANDID's immutable `split.json` contains held-out *recorded* scenes;
  ordinary online PPO/SAC randomization may still sample similar geometry.
  A rigorous unseen-scene claim requires controlled train-only RL reset
  distributions or an independent held-out scene family, not only the
  evaluation split.

## Dependencies

Use your existing Docker image with ManiSkill 3, SAPIEN, PyTorch, Gymnasium,
`diffusers`, `h5py`, `numpy` and `torchvision`. Extra upstream training
dependencies: `tyro tensorboard tqdm IPython`. For CPU training, don't use
massive GPU-oriented PPO/SAC parallel environment defaults.

## Preparation

```bash
cd /workspace/CANDID
export PYTHONPATH="$PWD:$PWD/src"
python baselines/common/export_demos.py \
  --dataset demos/merged_all \
  --output demos/candid_baselines_train.h5
```

The exporter reads **train scenes only**, and emits a flattened
`traj_N/obs [T+1,D]`, `actions [T,7]`, `success [T]` HDF5 along with
ManiSkill-compatible `.json` metadata. It refuses to overwrite existing
exports. Datasets with distinct scene geometries must keep identical feature
dimensions, and all training trajectories must use `pd_joint_pos`.

## Training examples

### BC (supervised MLP + MSE)

```bash
python baselines/bc/bc.py --demo-path demos/candid_baselines_train.h5 \
  --env-id CANDIDBaseline-v1 --env-module envs.candid_baseline_env \
  --sim-backend cpu --control-mode pd_joint_pos --max-episode-steps 300 \
  --total-iters 10000 --batch-size 64 --num-eval-envs 1 --no-capture-video \
  --exp-name candid_bc
```

### ACT (CVAE Transformer, ACT reconstruction + KL)

```bash
python baselines/act/train.py --demo-path demos/candid_baselines_train.h5 \
  --env-id CANDIDBaseline-v1 --env-module envs.candid_baseline_env \
  --sim-backend cpu --control-mode pd_joint_pos --max-episode-steps 300 \
  --total-iters 30000 --batch-size 64 --num-eval-envs 1 --num-queries 30 \
  --exp-name candid_act --no-capture-video
```

Upstream ACT temporal aggregation is retained for final evaluation. It requires
a new policy query every environment step (`exec_horizon=1`) even though ACT
predicts an action chunk. The upstream evaluation implementation has additional
assumptions about synchronous episode truncation; use the CANDID evaluator for
the final metrics.

### PPO (online RL)

```bash
python baselines/ppo/ppo.py --env-id CANDIDBaseline-v1 \
  --env-module envs.candid_baseline_env \
  --control-mode pd_joint_pos --sim-backend cpu \
  --num-envs 1 --num-eval-envs 1 --num-minibatches 4 \
  --num-steps 64 --total-timesteps 200000 --no-capture-video \
  --exp-name candid_ppo
```

### SAC (online RL)

```bash
python baselines/sac/sac.py --env-id CANDIDBaseline-v1 \
  --env-module envs.candid_baseline_env \
  --control-mode pd_joint_pos --sim-backend cpu \
  --num-envs 1 --num-eval-envs 1 --buffer-device cpu \
  --buffer-size 100000 --training-freq 64 --total-timesteps 200000 \
  --no-capture-video --exp-name candid_sac
```

These are starting configurations for CPU; learning rate, sample budget and
task reward still require proper validation and tuning. ACT/BC `--total-iters`
counts gradient updates, PPO/SAC `--total-timesteps` counts interactions.
Report both training timesteps and wall-clock costs.

## Held-out evaluation (the same protocol for all four)

```bash
python baselines/common/eval_baseline.py \
  --method bc --dataset demos/merged_all \
  --checkpoint runs/candid_bc/checkpoints/best_eval_success_at_end.pt \
  --output-dir runs/eval_bc --safe-dr-mean-max 0.2 \
  --device cpu
```

Change `--method` and checkpoint for `act`, `ppo`, or `sac`.
The `0.2` DR threshold is **illustrative**, not an approved experimental
threshold. `--save-video` enables output videos. A checkpoint may not be
written when official best-success saving never sees a strictly positive
success rate; use checkpoint frequency options or implement validation-based
saving for full training runs.

PPO checkpoints are plain `Agent.state_dict()`; SAC checkpoints are dicts
with an `actor` field. ACT and BC checkpoints include CANDID metadata
needed by the shared evaluator.

## Status and limitations

- Source inspected and adapted; basic import/data shape/parameter tests are in
  `tests/test_baselines_contract.py`.
- End-to-end ManiSkill training/physical rollout **has not run in the code
  authoring environment** (ManiSkill/SAPIEN are not installed there).
- CPU vector wrappers, ACT evaluation aggregation and SAC replay training need
  a real Docker smoke run before calling these production-ready.
- Evaluate only on held-out scenes; the public ACT/BC built-in evaluation at
  training time does NOT enforce CANDID's frozen test split. Treat it as
  debugging, not your test result.
