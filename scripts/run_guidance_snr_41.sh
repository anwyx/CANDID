#!/usr/bin/env bash
# Run three matched Scene1 SNR Guidance ablations on separate CUDA devices.
# Usage: bash scripts/run_guidance_snr_41.sh 1 2 3
# Override DP_CKPT, PRED_CKPT, EVAL_SUITE and SNR_RUN_PREFIX if needed.
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: bash scripts/run_guidance_snr_41.sh GPU_GAUSSIAN GPU_INCREASING GPU_CAPPED" >&2
  exit 2
fi
if [[ "$1" == "$2" || "$1" == "$3" || "$2" == "$3" ]]; then
  echo "Use distinct physical GPU IDs for the three experiments" >&2
  exit 2
fi

export PYTHONPATH="src:."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DP_CKPT="${DP_CKPT:-ckpt/scene1/best_val.pt}"
PRED_CKPT="${PRED_CKPT:-runs/predictor_ordinal_joint_regularized_v2/checkpoints/best_val.pt}"
EVAL_SUITE="${EVAL_SUITE:-eval/scene1}"
PREFIX="${SNR_RUN_PREFIX:-runs/scene1_guidance_snr_41}"
for entry in "$DP_CKPT" "$PRED_CKPT"; do
  if [[ ! -f "$entry" ]]; then
    echo "Missing checkpoint: $entry" >&2
    exit 2
  fi
done
if [[ ! -e "$EVAL_SUITE" ]]; then
  echo "Missing evaluation suite: $EVAL_SUITE" >&2
  exit 2
fi

modes=(snr_gaussian snr_increasing snr_gaussian_capped)
gpus=("$1" "$2" "$3")
pids=()
for i in 0 1 2; do
  mode="${modes[$i]}"
  gpu="${gpus[$i]}"
  output="${PREFIX}_${mode}"
  if [[ -e "$output" ]]; then
    echo "Output already exists, refusing overwrite: $output" >&2
    exit 2
  fi
done

for i in 0 1 2; do
  mode="${modes[$i]}"
  gpu="${gpus[$i]}"
  output="${PREFIX}_${mode}"
  echo "Starting $mode on physical GPU $gpu -> $output"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/eval_dp_guidance.py \
    --eval-suite "$EVAL_SUITE" \
    --dp-checkpoint "$DP_CKPT" \
    --predictor-checkpoint "$PRED_CKPT" \
    --output-dir "$output" \
    --safe-dr-mean-max 0.2 \
    --max-scenes 41 --max-steps 300 --exec-horizon 4 \
    --eval-seed 2026 --device cuda --strength 0.02 \
    --start-timestep 30 --guidance-parity odd \
    --strength-schedule "$mode" \
    --snr-center-timestep 15 --snr-sigma 1.0 \
    --dr-mean-weight 1.0 --dr-max-weight 0.25 \
    --topple-weight 0.5 --fall-weight 0 \
    --max-grad-norm 1.0 --max-step-norm 0.02 \
    --no-video > "${output}.log" 2>&1 &
  pids+=("$!")
done

echo "Started three jobs. PIDs: ${pids[*]}"
failed=0
for i in 0 1 2; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED: ${modes[$i]} (see ${PREFIX}_${modes[$i]}.log)" >&2
    failed=1
  else
    echo "DONE: ${modes[$i]} (see ${PREFIX}_${modes[$i]}/summary.json)"
  fi
done
exit "$failed"
