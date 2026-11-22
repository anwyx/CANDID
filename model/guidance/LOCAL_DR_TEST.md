# Paired fixed-state physical DR test (Scene 1)

This is **not** an episode-level comparison. It tests local action consequences
under the same simulator state and exactly matched Torch sampling RNG.

At selected steps along a **Vanilla DP reference trajectory**, the evaluator:
1. Saves a complete simulator state and the DP observation history.
2. Generates an 8-step Vanilla action and an 8-step guided action from
   **identical diffusion RNG states**. Neither candidate changes the reference
   trajectory's RNG.
3. Uses a separate simulator instance, initialized from the same scene
   snapshot, and restores the saved state before every candidate replay.
4. Executes each action chunk independently, records DR after the first 4
   and all 8 steps (using the reference state at the probe as DR origin).
5. Scores both 8-step chunks with the frozen Predictor. It also repeats
   the Vanilla replay and checks physical determinism by default.

## 1. One-scene smoke test

```bash
python -m pytest tests/test_local_dr.py -q
mkdir -p logs
set -o pipefail
python scripts/eval_local_dr.py \
  --dataset demos/scene1 \
  --dp-checkpoint ckpt/scene1/best_val.pt \
  --predictor-checkpoint ckpt_file/best_val.pt \
  --output-dir runs/local_dr_smoke \
  --scene-substrings 145921 \
  --probe-steps 40 \
  --max-steps 300 \
  --eval-seed 2026 \
  --strength 0.005 \
  --start-timestep 30 \
  --device cuda \
  2>&1 | tee logs/local_dr_smoke.log
```

## 2. All five scenes, four probes each

```bash
python scripts/eval_local_dr.py \
  --dataset demos/scene1 \
  --dp-checkpoint ckpt/scene1/best_val.pt \
  --predictor-checkpoint ckpt_file/best_val.pt \
  --output-dir runs/local_dr_5scenes \
  --scene-substrings 145921,162144,153215,172333,155446 \
  --probe-steps 40,80,120,160 \
  --max-steps 300 \
  --eval-seed 2026 \
  --strength 0.005 \
  --start-timestep 30 \
  --device cuda \
  2>&1 | tee logs/local_dr_5scenes.log
```

Output: `pairs.jsonl` includes predicted risk, predicted 8-step
DR and measured 4-step / 8-step DR of both candidates, plus deltas.
`summary.json` reports number of valid pairs and average physical DR deltas.

Important: The 8-step Predictor is trained on **expert** action windows,
whereas this experiment probes model-generated actions. This comparison
tests whether local risk *ranking* transfers. It does not require the
Predictor to estimate episode DR or task success. A full 8-step replay is
an independent counterfactual (the usual closed-loop controller replans
after four steps). The first four steps of the two independent replays
are the executed action prefixes and can be compared fairly.

If baseline repeated restoration is not physically deterministic, the script
raises an error instead of reporting a misleading paired comparison.
ManiSkill versions must agree with the saved state snapshots.

The test requires headless CPU physics (`render_backend=none`) and permits
CUDA inference. It does not change training, Guidance, or the episode evaluator.
