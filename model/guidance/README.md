# CANDID physical-consequence gradient guidance

Independent inference-only guidance of a frozen Vanilla DP by a frozen
disturbance Predictor. **No changes** to the baseline checkpoint/training
or the existing Best-of-N reranking implementation are needed.

At DDPM timestep t, estimate a denoised action sequence x0 from current x_t
and epsilon_theta; detach epsilon_theta to avoid expensive UNet backprop.
Extract the DP's executable window
`[obs_horizon - 1 : obs_horizon - 1 + act_horizon]`, denormalize to
absolute joint targets, then clamp to the environment action space.
The Predictor normalizes its inputs independently.

Define local chunk risk:
`J = w_mean*max(0,DR_mean) + w_max*max(0,DR_max,DR_mean) + w_topple*sigmoid(topple_logit) + w_fall*sigmoid(fall_logit)`.
Its DR outputs are mapped back to original units before computing gradients.
For t <= start_timestep, update the DDPM posterior step using
`x_{t-1} = scheduler.step(epsilon_theta,t,x_t).prev_sample - rho * grad_{x_t} J`,
with per-sample L2 gradient and correction bounds.

This is a CPU-friendly, first-order GRITS-style guidance approximation,
not a claim of bitwise equivalence to GRITS. In particular, the predictor
outputs local **chunk scores**, not calibrated episode safety probabilities,
and does not explicitly model task success. Therefore, compare success,
actual physical disturbance, topple and fall under identical replay splits.
`--strength 0` is an exact Vanilla DP sampling and RNG control (and skips
Predictor evaluation). Do not infer effectiveness without closed-loop tests.

## Evaluation

```bash
python scripts/eval_dp_guidance.py \
  --dataset demos/dense_random_120_no_collision_aware \
  --dp-checkpoint /path/to/your_dp.pt \
  --predictor-checkpoint /path/to/your_predictor.pt \
  --output-dir runs/guidance_eval_18 \
  --safe-dr-mean-max 0.1 --max-scenes 18 --max-steps 700 \
  --exec-horizon 8 --strength 0.005 --start-timestep 30 --no-video
```

Checkpoint locations and the disturbance safety threshold above are
examples: replace them with actual file paths and evaluation threshold.
The dataset must have a valid `split.json` with held-out test scenes.
For a no-op control, repeat with `--strength 0` and a different output dir.
Use the *same* dataset, test scenes, seed, execution horizon, and DR safety
threshold as the Vanilla and Rerank comparisons. For an initial sweep,
try strengths 0.001, 0.005, 0.01; measure both disturbance and task success.
Reports include ordinary episode metrics and per-episode
`policy_diagnostics.json` with guided-step counts and correction norms.

Tests: `python -m pytest tests/test_guidance.py -q`.
