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

## Evaluation (the same scene1 protocol as Vanilla DP / Rerank)

Use `demos/scene1`, the 18 held-out test scenes, max 300 environment
steps, execution horizon 4 (the Predictor still scores 8-step chunks),
eval seed 2026, and DR-mean safety cutoff 0.2.  The DP and Predictor
checkpoints are independent.

```bash
python scripts/eval_dp_guidance.py \
  --dataset demos/scene1 \
  --dp-checkpoint ckpt/scene1/best_val.pt \
  --predictor-checkpoint ckpt_file/best_val.pt \
  --output-dir runs/scene1_guidance_0005 \
  --max-scenes 18 --max-steps 300 --exec-horizon 4 \
  --eval-seed 2026 --safe-dr-mean-max 0.2 \
  --strength 0.005 --start-timestep 30 --device cpu --no-video
```

For the exact sampling control use the same command with
`--strength 0 --output-dir runs/scene1_guidance_zero`.
Output directories must not already contain results. Compare with
`runs/scene1` (Vanilla), `runs/scene1_shadow` and the scene1 Rerank
run using the same dataset split, seed, DR cutoff and execution horizon.

Suggested guidance ablation: strengths 0, 0.001, 0.005, 0.01
with separate output directories. Inspect `summary.json`,
`episodes.jsonl`, and per-episode `policy_diagnostics.json`.
Do not claim safety improvements without checking real disturbance,
topple/fall, success and safe success in closed-loop rollouts.

Tests: `python -m pytest tests/test_guidance.py -q`.

## Diagnostic rerun: five informative scene1 cases

These are the exact held-out scenes from the previous 18-scene run, identified
by unique substrings. The subset retains each scene's **original index-based
evaluation seed**, so the new rollout can be paired with the earlier full
evaluation. The same-noise counterfactual Vanilla action is evaluated at each
Guidance observation; only Guidance actions are sent to the simulator.

```bash
mkdir -p logs
set -o pipefail
/usr/bin/time -f '\nElapsed time: %e seconds\nPeak memory: %M KB' \
python scripts/eval_dp_guidance.py \
  --dataset demos/scene1 \
  --dp-checkpoint ckpt/scene1/best_val.pt \
  --predictor-checkpoint ckpt_file/best_val.pt \
  --output-dir runs/scene1_guidance_diag5 \
  --scene-substrings 145921,162144,153215,172333,155446 \
  --max-steps 300 --exec-horizon 4 --eval-seed 2026 \
  --safe-dr-mean-max 0.2 --strength 0.005 --start-timestep 30 \
  --device cuda --diagnose --profile-timing --no-video \
  2>&1 | tee logs/scene1_guidance_diag5.log
```

Then summarize and compare the true closed-loop outcomes from the original
18-scene `strength=0` run:

```bash
python scripts/analyze_guidance_diagnostics.py \
  runs/scene1_guidance_diag5 \
  --baseline-run runs/scene1_guidance_zero
```

Interpretation:
- `paired.risk_before` vs `risk_after` scores different chunks under the
  **same live observation** using the exact same diffusion sampling RNG.
- `paired.action_abs_mean/max` measure differences in clipped absolute
  environment joint targets, not normalized latent tensors.
- `timing_ms.denoiser_ms`, `predictor_gradient_ms`, `ddpm_update_ms`
  are CUDA-event GPU stream times, while `sampling_wall_ms` and
  `baseline_sampling_wall_ms` are wall-clock time. They do not sum exactly.
- Additional counterfactual sampling and timing both add overhead, so do
  **not** compare diagnostic-run wall time directly to the previous 18-scene
  runtime as a throughput benchmark.
- The gradient step still optimizes predicted local chunk risk, which may
  disagree with actual environment safety.
