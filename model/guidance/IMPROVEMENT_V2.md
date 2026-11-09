# CANDID improvement v2: policy-distribution risk + executed-prefix guidance

## Goal

Improve **actual physical safety** (DR mean/max, topple/fall and safe-success)
rather than minimizing the Predictor's proxy alone. Keep Vanilla DP checkpoint
unchanged for fair comparison. This is a new candidate algorithm, NOT an
empirically confirmed improvement yet.

What changes:
1. Collect Vanilla DP **training-scene** rollouts with canonical states
   `observations [T+1,O]`, executed actions `[T,A]`, and physics truth.
2. Train a **new predictor** on expert windows + 50% sampled DP rollout
   windows. The Predictor still accepts 8 action targets but its DR/topple
   labels represent ONLY the **first four executed steps**.
3. The Predictor network masks the unused action suffix in its normalized
   input so its estimated local consequence is not influenced by never-executed
   actions.
4. The guidance sampler can restrict gradient corrections to the **first
   four executed actions**; it still uses the same frozen DDPM denoiser,
   and the normal strength=0 Vanilla control is unchanged.

**Leakage protection**: `--split train` in the DP evaluator; Predictor
loader rejects rollout directories not tagged as train and any scene not in
the dataset's train split. Expert validation/test scenes remain held out.

Run from repository root on the CUDA server with the SAME ManiSkill build and
the same `demos/scene1/split.json` used for the baseline.

### 1. Update and test

```bash
git pull --ff-only origin feat/predictor-p0-h8-training-20261106
PYTHONPATH="$PWD:$PWD/src" python -m pytest \
  tests/test_guidance.py tests/test_policy_rollout_training.py \
  tests/test_predictor_p0.py tests/test_predictor_p1.py -q
```

### 2. Collect on-policy TRAIN data (not held-out test!)

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python scripts/eval_dp.py \
  --dataset demos/scene1 \
  --checkpoint ckpt/scene1/best_val.pt \
  --output-dir runs/scene1_dp_train_rollouts_v2 \
  --split train --max-scenes 48 \
  --max-steps 300 --exec-horizon 4 \
  --eval-seed 2026 --safe-dr-mean-max 0.2 \
  --device cuda --no-video \
  2>&1 | tee logs/scene1_dp_train_rollouts_v2.log
```

The output `episodes.jsonl` must list train scenes only, and each
`trajectory.npz` must contain `observations`. Do not substitute
`runs/scene1_guidance_diag5`: those are test-scene data.

### 3. Fit new on-policy predictor (DO NOT overwrite old checkpoint)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_predictor.py \
  --dataset demos/scene1 \
  --policy-rollouts runs/scene1_dp_train_rollouts_v2 \
  --label-horizon 4 --action-horizon 8 --obs-horizon 2 \
  --steps 12000 --batch-size 128 --lr 0.0003 \
  --eval-freq 500 --device cuda \
  --output-dir runs/predictor_policy_h4_v2 \
  2>&1 | tee logs/predictor_policy_h4_v2.log
```

`runs/predictor_policy_h4_v2/checkpoints/best_val.pt` has the old
interface (obs2, actions8) but now predicts physical risk from the
first four actions. It is trained from scratch with its own normalizer,
so do not compare its validation loss directly to the old 8-step model
without matching label definitions.

### 4. Evaluate the new predictor on the frozen test split

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_predictor.py \
  --dataset demos/scene1 \
  --checkpoint runs/predictor_policy_h4_v2/checkpoints/best_val.pt \
  --output-dir runs/predictor_policy_h4_v2_test \
  --device cuda
```

### 5. Full held-out 18-scene Guidance evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_dp_guidance.py \
  --dataset demos/scene1 \
  --dp-checkpoint ckpt/scene1/best_val.pt \
  --predictor-checkpoint runs/predictor_policy_h4_v2/checkpoints/best_val.pt \
  --output-dir runs/scene1_guidance_policy_h4_v2 \
  --max-scenes 18 --max-steps 300 --exec-horizon 4 \
  --guided-action-horizon 4 --eval-seed 2026 \
  --safe-dr-mean-max 0.2 \
  --strength 0.02 --start-timestep 30 \
  --dr-mean-weight 1 --dr-max-weight 0.25 \
  --topple-weight 0.1 --device cuda --no-video \
  2>&1 | tee logs/scene1_guidance_policy_h4_v2.log
```

Compare `summary.json` to the frozen Vanilla evaluation with exactly
the same split, seed, physical replay, action horizon, and safety limits.
Primary: **safe-success**, safety, actual DR, topple/fall.
Secondary: success and runtime. Prediction risk decreases alone are
NOT evidence of improved safety.

For strict same-runtime comparisons regenerate the strength=0 control
on the same code/version; the new metadata and trajectory storage do
not change how its actions are selected.
