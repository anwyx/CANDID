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
dataset export, ACT's broken upstream stats method, a standalone diffusers-compatible EMA to avoid unnecessary torchvision imports, BC normalization-at-eval,
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

The exporter reads **train scenes only** (potentially across different source environments), and emits a flattened
`traj_N/obs [T+1,D]`, `actions [T,7]`, `success [T]` HDF5 along with
ManiSkill-compatible `.json` metadata. The exported `env_info.env_id` is a representative first-training-scene environment only, not evidence of a single source environment; `candid.source_environments` lists all source environments. It refuses to overwrite existing
exports. Datasets with distinct scene geometries must keep identical feature
dimensions, and all training trajectories must use `pd_joint_pos`.

## Identify the real environment before training

`demos/merged_all` is a **dataset**, not a Gym environment. Its scenes may come
from multiple teammate environment modules. Inspect its metadata first:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
root = Path("demos/merged_all")
rows = [json.loads(line) for line in (root/"scenes.jsonl").read_text().splitlines() if line.strip()]
split = json.loads((root/"split.json").read_text())
for split_name in ("train", "validation", "test"):
    selected = set(split[split_name])
    counter = Counter((r["env_id"], r["env_module"]) for r in rows if r["scene_id"] in selected)
    print(split_name, dict(counter))
PY
```

Choose a **real, importable** pair from the TRAIN split, for example:

```bash
export ENV_ID='ACTUAL_ENV_ID_FROM_METADATA'
export ENV_MODULE='envs.actual_env_module_from_metadata'
```

BC/ACT may imitate mixed-environment training demonstrations only when their
canonical observation and action contracts are identical. Their built-in online
training-time evaluation picks the single environment provided by `ENV_ID` /
`ENV_MODULE`; this is not the frozen test evaluation.

PPO/SAC train **online**, so a single chosen `ENV_ID`/`ENV_MODULE` does not
automatically cover a merged multi-environment dataset. In particular, do NOT
call a baseline `merged_all`-trained RL agent until an explicit train-scene
sampling mechanism exists. The final held-out evaluator reconstructs each
recorded test scene from its own metadata and snapshot.

## Training examples

### BC (supervised MLP + MSE)

```bash
python baselines/bc/bc.py --demo-path demos/candid_baselines_train.h5 \
  --env-id $ENV_ID --env-module $ENV_MODULE \
  --sim-backend cpu --control-mode pd_joint_pos --max-episode-steps 300 \
  --total-iters 10000 --batch-size 64 --num-eval-envs 1 --no-capture-video \
  --exp-name candid_bc
```

### ACT (CVAE Transformer, ACT reconstruction + KL)

```bash
python baselines/act/train.py --demo-path demos/candid_baselines_train.h5 \
  --env-id $ENV_ID --env-module $ENV_MODULE \
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
python baselines/ppo/ppo.py --env-id $ENV_ID \
  --env-module $ENV_MODULE \
  --control-mode pd_joint_pos --sim-backend cpu \
  --num-envs 1 --num-eval-envs 1 --num-minibatches 4 \
  --num-steps 64 --total-timesteps 200000 --no-capture-video \
  --exp-name candid_ppo
```

### SAC (online RL)

```bash
python baselines/sac/sac.py --env-id $ENV_ID \
  --env-module $ENV_MODULE \
  --control-mode pd_joint_pos --sim-backend cpu \
  --num-envs 1 --num-eval-envs 1 --buffer-device cpu \
  --buffer-size 100000 --training-freq 64 --total-timesteps 200000 \
  --no-capture-video --exp-name candid_sac
```

These are starting configurations for CPU; learning rate, sample budget and
task reward still require proper validation and tuning. ACT/BC `--total-iters`
counts gradient updates, PPO/SAC `--total-timesteps` counts interactions.
Report both training timesteps and wall-clock costs.

## Extremely small CPU smoke test using `ring_120`

This checks **one model update and generation of finite actions**, not success,
convergence, or fairness. `demos/merged_all` is NOT required.

Keep any locally created `envs/candid_ring_env.py` importable. The dataset
alone contains recorded states and does not register the source Gym env.

```bash
cd /workspace/CANDID
export PYTHONPATH="$PWD:$PWD/src"
test -f demos/ring_120/scenes.jsonl
python - <<'PY'
import json
from pathlib import Path
p = Path("demos/ring_120/scenes.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
print("RING ENVIRONMENTS:")
for env_id, module in sorted({(r["env_id"], r["env_module"]) for r in rows}):
    print(env_id, module)
PY
```

Set the actual environment ID/module from the displayed data. For the commonly
used Ring environment these are `CANDIDRing-v1` and
`envs.candid_ring_env`, but **verify the metadata**, and don't assume its
module exists in GitHub if it was only created locally.

```bash
export ENV_ID=CANDIDRing-v1
export ENV_MODULE=envs.candid_ring_env
# fail early if the env module was deleted locally:
python -c "import importlib; importlib.import_module('$ENV_MODULE')"
python baselines/common/export_demos.py \
  --dataset demos/ring_120 \
  --output demos/ring_smoke_train.h5 \
  --max-trajectories 2
```

The exporter reads only successful demonstrations in the TRAIN split. If
`ring_smoke_train.h5` already exists, choose a new output name.

BC: **one optimizer update**, no online evaluation:

```bash
python baselines/bc/bc.py \
  --env-id "$ENV_ID" --env-module "$ENV_MODULE" \
  --demo-path demos/ring_smoke_train.h5 --no-cuda \
  --sim-backend cpu --control-mode pd_joint_pos \
  --num-demos 2 --batch-size 2 --total-iters 1 \
  --max-episode-steps 10 --num-eval-envs 1 \
  --eval-freq 0 --save-freq 1 --log-freq 1 \
  --no-capture-video --exp-name ring_smoke_bc
```

ACT: **one optimizer update**, tiny Transformer for a *smoke test only*
(the formal ACT baseline must restore its original architecture):

```bash
python baselines/act/train.py \
  --env-id "$ENV_ID" --env-module "$ENV_MODULE" \
  --demo-path demos/ring_smoke_train.h5 --no-cuda \
  --sim-backend cpu --control-mode pd_joint_pos \
  --num-demos 2 --batch-size 2 --total-iters 1 \
  --num-queries 2 --hidden-dim 64 --nheads 2 \
  --enc-layers 1 --dec-layers 1 --dim-feedforward 128 \
  --max-episode-steps 10 --num-eval-envs 1 \
  --eval-freq 0 --save-freq 1 --log-freq 1 \
  --no-capture-video --exp-name ring_smoke_act
```

PPO: **four real environment steps**, one update. CANDID currently supports
only one CPU simulation environment:

```bash
python baselines/ppo/ppo.py \
  --env-id "$ENV_ID" --env-module "$ENV_MODULE" \
  --no-cuda --sim-backend cpu --control-mode pd_joint_pos \
  --num-envs 1 --num-eval-envs 1 --num-steps 4 \
  --num-minibatches 2 --update-epochs 1 --total-timesteps 4 \
  --eval-freq 0 --no-capture-video --exp-name ring_smoke_ppo
```

SAC: **four real environment steps**, one Q/Actor update (not just random
collection):

```bash
python baselines/sac/sac.py \
  --env-id "$ENV_ID" --env-module "$ENV_MODULE" \
  --no-cuda --sim-backend cpu --control-mode pd_joint_pos \
  --num-envs 1 --num-eval-envs 1 --buffer-device cpu \
  --buffer-size 64 --batch-size 4 --learning-starts 1 \
  --training-freq 4 --utd 0.25 --total-timesteps 4 \
  --eval-freq 0 --log-freq 4 --no-capture-video \
  --exp-name ring_smoke_sac
```

**Generate actions without another simulator rollout**:

```bash
for method in bc act ppo sac; do
  case "$method" in
    bc)  ckpt=runs/ring_smoke_bc/checkpoints/0.pt ;;
    act) ckpt=runs/ring_smoke_act/checkpoints/0.pt ;;
    ppo) ckpt=runs/ring_smoke_ppo/final_ckpt.pt ;;
    sac) ckpt=runs/ring_smoke_sac/final_ckpt.pt ;;
  esac
  python baselines/common/smoke_actions.py \
    --method "$method" --demo-path demos/ring_smoke_train.h5 \
    --checkpoint "$ckpt" --output "runs/ring_smoke_${method}/actions.npy"
done
```

A `passed: true` for all four confirms weights were saved, reloaded, and
produced finite `[H,7]` actions from a recorded Ring observation. It **does
not** prove those actions solve the task. For a real 1-scene closed-loop
rollout use `baselines/common/eval_baseline.py --dataset demos/ring_120
--max-scenes 1 --max-steps 10`, with an agreed safety threshold, after
verifying the source scene's environment module is available.

### Pitfalls

- `--eval-freq 0` disables slow training-time evaluations in ACT/BC/PPO/SAC;
  checkpoints are still saved by the smoke commands.
- Two demonstrations have **very little training signal** and must never be
  used to compare success rates or publish metrics.
- PPO/SAC randomize Ring environment resets; they do **not** replay the
  `ring_120` dataset, or enforce a held-out train scene set.
- The policy action generator uses a recorded Ring state; it doesn't require
  rendering, state restoration, or another simulation rollout.
- Avoid reusing prior `--exp-name`/output paths if you want to keep old
  checkpoints intact.

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
