#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$(pwd)/datasets}"
OUT_ROOT="${OUT_ROOT:-$(pwd)/output_priority}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-300}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

mkdir -p "${OUT_ROOT}"

run_drive() {
  local model="$1"
  local input_mode="$2"
  local loss_mode="$3"
  python code/train_drive.py \
    --model-name "${model}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/drive/${model}_${input_mode}_${loss_mode}" \
    --input-mode "${input_mode}" \
    --loss-mode "${loss_mode}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}"
}

run_chase() {
  local model="$1"
  local input_mode="$2"
  local loss_mode="$3"
  python code/train_chase.py \
    --model-name "${model}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}/chase/${model}_${input_mode}_${loss_mode}" \
    --input-mode "${input_mode}" \
    --loss-mode "${loss_mode}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}"
}

# Core model comparison under the same preprocessing/loss protocol.
for model in unet sa_unet sa_unetv2; do
  run_drive "${model}" green_clahe bce_dice
  run_chase "${model}" green_clahe bce_dice
done

# Minimal ablations that isolate preprocessing and loss contributions.
run_drive unet green bce_dice
run_drive unet rgb bce_dice
run_drive unet green_clahe bce
run_chase unet green bce_dice
run_chase unet rgb bce_dice
run_chase unet green_clahe bce

python code/summarize_metrics.py --root "${OUT_ROOT}" --output "${OUT_ROOT}/summary.csv"
