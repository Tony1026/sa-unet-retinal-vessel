#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-datasets}"
OUT_ROOT="${OUT_ROOT:-output_bias_core}"
LOG_DIR="${OUT_ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-/data1/gushengda/anaconda3/envs/hci/bin/python}"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${ROOT_DIR}/.deps${PYTHONPATH:+:${PYTHONPATH}}"
export GFLOW_ALIGN_WEIGHT="${GFLOW_ALIGN_WEIGHT:-0.15}"
export GFLOW_CONS_WEIGHT="${GFLOW_CONS_WEIGHT:-0.01}"
export GFLOW_SPARSE_WEIGHT="${GFLOW_SPARSE_WEIGHT:-0.02}"
export GFLOW_SECOND_WEIGHT="${GFLOW_SECOND_WEIGHT:-0.35}"
export GFLOW_MISSING_SECOND_WEIGHT="${GFLOW_MISSING_SECOND_WEIGHT:-0.35}"

GPU_IDS=(${GPU_IDS:-0 1 2 3 4 5 6 7})
MAX_JOBS="${MAX_JOBS:-${#GPU_IDS[@]}}"
JOB_INDEX=0

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]; do
    sleep 10
  done
}

run_job() {
  local dataset="$1"
  local model="$2"
  local bias="$3"
  local extra_tag="$4"
  local gpu="${GPU_IDS[$((JOB_INDEX % ${#GPU_IDS[@]}))]}"
  JOB_INDEX=$((JOB_INDEX + 1))

  local output_dir="../${OUT_ROOT}/${dataset}/${model}_${bias}_${extra_tag}"
  local log_path="${LOG_DIR}/${dataset}_${model}_${bias}_${extra_tag}.log"

  wait_for_slot
  echo "[launch] dataset=${dataset} model=${model} bias=${bias} gpu=${gpu} log=${log_path}"
  if [ "${dataset}" = "drive" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" code/train_drive.py \
      --model-name "${model}" \
      --data-root "${DATA_ROOT}" \
      --input-mode green_clahe \
      --loss-mode bce_dice \
      --label-bias-mode "${bias}" \
      --epochs 300 \
      --plateau-patience 20 \
      --early-stop-patience 60 \
      --output-dir "${output_dir}" \
      > "${log_path}" 2>&1 &
  elif [ "${dataset}" = "chase" ]; then
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
      --output-dir "${output_dir}" \
      > "${log_path}" 2>&1 &
  else
    echo "Unsupported dataset: ${dataset}" >&2
    exit 1
  fi
}

for dataset in drive chase; do
  run_job "${dataset}" gflow_unet second full_learned_conserved
  run_job "${dataset}" gflow_unet_no_cons second no_conservation
  run_job "${dataset}" gflow_unet_uniform second fixed_uniform
  run_job "${dataset}" gflow_unet_random second fixed_random
  run_job "${dataset}" gflow_unet_randinit second random_init_learned
  run_job "${dataset}" gflow_sa_unetv2 second sa_backbone_learned
  run_job "${dataset}" unet random_primary single_head_unet
  run_job "${dataset}" sa_unetv2 random_primary single_head_sa_unetv2
done

wait
"${PYTHON_BIN}" code/summarize_bias_core.py --root-dir "${OUT_ROOT}" --output-csv "${OUT_ROOT}/comparison.csv"
echo "[done] ${OUT_ROOT}/comparison.csv"
