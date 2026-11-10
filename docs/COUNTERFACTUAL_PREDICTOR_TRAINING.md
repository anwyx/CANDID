# Counterfactual + ordinal Predictor training (first ablation)

This is a **separate training entrypoint** so the original Ordinal Predictor,
Diffusion Policy, Guidance and Rerank paths stay untouched.

## Train-only source contract

- `--dataset`: the original expert dataset with frozen `split.json`.
- `--counterfactual-dataset`: **complete** physical counterfactual collection,
  using that exact expert train split. Every replay candidate has a real
  physics label. Incomplete collections are rejected.
- Source scenes in counterfactual samples must be a subset of original
  `split["train"]`. The training script partitions those counterfactual
  scenes again into **CF training and CF validation scenes**, preventing
  same-scene candidate leakage in the ranking metric.
- The original expert validation and test scenes never train this model.

## 1. Collect useful physical contrasts

Six states are only a functional smoke test, not a sufficient ML training set.
Try a moderate first batch (64 states x 7 candidates) in a fresh output folder:

```bash
PYTHONPATH=src:. python scripts/generate_counterfactuals.py \
  --dataset demos/scene1 \
  --output-dir runs/counterfactual_train_64_v1 \
  --states 64 --candidates 7 --max-per-scene 8 \
  --candidate-mode multiscale --sigma-levels 0.02,0.06,0.12 \
  --max-prefix-steps 160 --seed 2026
```

On interruption resume the **same command** with `--resume`. Check
`summary.json`: `complete=true`, `num_states=64`, and the counts of
states with diverse Topple / DR outcomes. Skipped expert source replays
do not contribute labels. More data may be needed for meaningful generalization.

## 2. Train by fine-tuning the existing Ordinal Predictor

Use the actual path of your frozen Ordinal V2 checkpoint if different:

```bash
PYTHONPATH=src:. python scripts/train_counterfactual_predictor.py \
  --dataset demos/scene1 \
  --counterfactual-dataset runs/counterfactual_train_64_v1 \
  --init-checkpoint runs/predictor_ordinal_expert_v2/checkpoints/best_val.pt \
  --output-dir runs/predictor_ordinal_cf_rank_v1 \
  --device cuda --steps 2000 --batch-size 64 --cf-batch-size 8 \
  --expert-weight 1.0 --cf-weight 0.3 --rank-weight 0.5 \
  --cf-val-ratio 0.2 --eval-freq 250 --seed 2026
```

Each step uses:
1. Expert batch: original Ordinal DR exceedance + Topple/Fall BCE.
2. Independent physically labeled CF state groups: the same supervised loss.
3. Within-state candidate pairs: smooth pairwise ranking from real
   DR Mean/Max and event outcomes; ties are excluded.

The expert sampler now groups windows in small trajectory-local chunks,
then mixes chunks from different trajectories into batches. This changes
only the new CF training entrypoint and helps avoid strongly correlated
one-trajectory-only minibatches.

**Checkpoint selection:** lowest weighted sum of original expert scene
validation loss and held-out CF scene supervised+ranking loss. Rank
accuracy in the training batch is not a held-out validation score.
`summary.json`, `train_log.csv`, `checkpoints/best_val.pt`, and
`checkpoints/last.pt` are emitted. The saved checkpoint format is still
Ordinal V3 and is loadable by the existing Guidance/Rerank loaders.

## Fast CPU contract tests

```bash
PYTHONPATH=src:. pytest -q \
  tests/test_counterfactual.py \
  tests/test_predictor_counterfactual.py \
  tests/test_predictor_ordinal.py
```

Do not conclude physical safety improved from training loss or held-out
ranking alone. Later compare Vanilla, prior Ordinal Predictor Rerank, and
new CF-Ranking Predictor Rerank on **the same independently held-out evaluation
suite**, with identical budgets, action horizons, and safety thresholds.

**Current status:** code added, not yet run on the user's GPU server and
not yet physically benchmarked. No safety improvement is claimed.
