#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

while pgrep -af 'code/train_(drive|chase).+output_bias_core' >/dev/null; do
  sleep 300
done

DATA_ROOT="${DATA_ROOT:-datasets}"
OUT_ROOT="${OUT_ROOT:-output_bias_core_direct}"
LOG_DIR="${OUT_ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-/data1/gushengda/anaconda3/envs/hci/bin/python}"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${ROOT_DIR}/.deps${PYTHONPATH:+:${PYTHONPATH}}"
export GFLOW_ALIGN_WEIGHT="${GFLOW_ALIGN_WEIGHT:-0.15}"
export GFLOW_CONS_WEIGHT="${GFLOW_CONS_WEIGHT:-0.01}"
export GFLOW_SPARSE_WEIGHT="${GFLOW_SPARSE_WEIGHT:-0.02}"
export GFLOW_SECOND_WEIGHT="${GFLOW_SECOND_WEIGHT:-0.35}"
export GFLOW_MISSING_SECOND_WEIGHT="${GFLOW_MISSING_SECOND_WEIGHT:-0.35}"

run_drive() {
  local model="$1"
  local bias="$2"
  local tag="$3"
  local gpu="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" code/train_drive.py \
    --model-name "${model}" \
    --data-root "${DATA_ROOT}" \
    --input-mode green_clahe \
    --loss-mode bce_dice \
    --label-bias-mode "${bias}" \
    --epochs 300 \
    --plateau-patience 20 \
    --early-stop-patience 60 \
    --output-dir "../${OUT_ROOT}/drive/${model}_${bias}_${tag}" \
    > "${LOG_DIR}/drive_${model}_${bias}_${tag}.log" 2>&1 &
}

run_chase() {
  local model="$1"
  local bias="$2"
  local tag="$3"
  local gpu="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" code/train_chase.py \
    --model-name "${model}" \
    --data-root "${DATA_ROOT}" \
    --input-mode green_clahe \
    --loss-mode bce_dice \
    --label-bias-mode "${bias}" \
    --epochs 300 \
    --batch-size 24 \
    --patches-per-image 8 \
    --plateau-patience 20 \
    --early-stop-patience 60 \
    --output-dir "../${OUT_ROOT}/chase/${model}_${bias}_${tag}" \
    > "${LOG_DIR}/chase_${model}_${bias}_${tag}.log" 2>&1 &
}

run_drive gflow_unet_direct second direct_sink 0
run_drive gflow_unet_direct random_primary direct_single_bias_control 1
run_chase gflow_unet_direct second direct_sink 2
run_chase gflow_unet_direct random_primary direct_single_bias_control 3
wait

"${PYTHON_BIN}" code/summarize_bias_core.py --root-dir "${OUT_ROOT}" --output-csv "${OUT_ROOT}/comparison.csv"
echo "[done] ${OUT_ROOT}/comparison.csv"
