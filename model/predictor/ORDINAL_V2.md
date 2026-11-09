# Ordinal Predictor V2 — step 2 of the planned CANDID upgrade

This branch leaves Vanilla DP, the scene split, all original expert
demonstrations and the existing regression Predictor checkpoint unchanged.
It adds a **separate** ordinal DR predictor that predicts monotone
P(DR Mean > threshold) and P(DR Max > threshold), plus Topple/Fall scores.

## Data choice

Training-scene 8-step chunk DR distribution (10,000 sampled windows, 2026-11-09):
- train expert trajectories: 2012 (successful 1773 / failed 239)
- train windows: 434181
- DR Mean > 1e-4: 16.47%, > 0.001: 14.35%, > 0.05: 2.68%, > 0.1: 0.83%
- DR Max > 1e-4: 23.06%, > 0.001: 16.07%, > 0.05: 7.73%, > 0.1: 5.35%
- Topple positive: 2.2%; Fall positive: 0% in the sampled windows.

Default thresholds: `0.0001,0.001,0.005,0.02,0.05,0.1`.
These are a first candidate, not validated optimum.
These rates refer to 8-step *chunk* labels, not whole-episode safety DR.

For the ordinal model, targets are PHYSICAL DR units and DR loss is weighted
multi-threshold BCE. Scalar score + ordered cutpoints enforce monotonic
exceedance probabilities. The continuous Guidance score integrates these
probabilities using threshold increments to produce a capped nonnegative
proxy DR (at most 0.1, **not** precise absolute DR). Topple/Fall retain their
binary heads. An old regression checkpoint still loads without changes.
No Ranking Loss or augmented trajectory data is included in this step.

Fall has no observed positive training examples in the sampled training
windows: the fall head cannot be considered reliable. Do not interpret
its logits as calibrated probabilities until training has event coverage.

## 1. Update and test

```bash
git switch feat/predictor-ordinal-v2
git pull --ff-only origin feat/predictor-ordinal-v2
PYTHONPATH="$PWD:$PWD/src" python -m pytest \
  tests/test_predictor_ordinal.py tests/test_predictor_p0.py \
  tests/test_predictor_p1.py tests/test_guidance.py -q
```

## 2. Train ordinal checkpoint on EXISTING experts only

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python scripts/train_predictor.py \
  --dataset demos/scene1 \
  --dr-objective ordinal \
  --dr-thresholds 0.0001,0.001,0.005,0.02,0.05,0.1 \
  --ordinal-pos-weight 5 \
  --steps 10000 --batch-size 128 --lr 0.0003 \
  --eval-freq 500 --device cuda \
  --output-dir runs/predictor_ordinal_expert_v2 \
  2>&1 | tee logs/predictor_ordinal_expert_v2.log
```

## 3. Held-out predictor evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_predictor.py \
  --dataset demos/scene1 \
  --checkpoint runs/predictor_ordinal_expert_v2/checkpoints/best_val.pt \
  --output-dir runs/predictor_ordinal_expert_v2_test \
  --device cuda
```

The summary includes ordinal_thresholds.{dr_mean,dr_max} with per-threshold
positive_rate, AUROC, AUPRC, Brier score, etc. Scalar MAE is reported for
the capped pseudo-DR, which is **NOT directly comparable** with the old
uncapped regression MAE.

## 4. Compare closed-loop physical safety

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_dp_guidance.py \
  --dataset demos/scene1 \
  --dp-checkpoint ckpt/scene1/best_val.pt \
  --predictor-checkpoint runs/predictor_ordinal_expert_v2/checkpoints/best_val.pt \
  --output-dir runs/scene1_guidance_ordinal_expert_v2 \
  --max-scenes 18 --max-steps 300 --exec-horizon 4 \
  --eval-seed 2026 --safe-dr-mean-max 0.2 \
  --strength 0.005 --start-timestep 30 \
  --device cuda --no-video \
  2>&1 | tee logs/scene1_guidance_ordinal_expert_v2.log
```

The rollout output must be compared against exactly the same 18 held-out
scenes and baseline evaluation seed. Primary metrics: success, safety,
safe-success, physical DR and topple/fall. Predictor validation improvements
alone do not establish CANDID effectiveness.
