#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-datasets}"
OUT_ROOT="${OUT_ROOT:-output_bias_neurips_core}"
LOG_DIR="${OUT_ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-/data1/gushengda/anaconda3/envs/hci/bin/python}"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${ROOT_DIR}/.deps${PYTHONPATH:+:${PYTHONPATH}}"
export GFLOW_ALIGN_WEIGHT="${GFLOW_ALIGN_WEIGHT:-0.15}"
export GFLOW_CONS_WEIGHT="${GFLOW_CONS_WEIGHT:-0.005}"
export GFLOW_SPARSE_WEIGHT="${GFLOW_SPARSE_WEIGHT:-0.0}"
export GFLOW_SECOND_WEIGHT="${GFLOW_SECOND_WEIGHT:-1.0}"
export GFLOW_MISSING_SECOND_WEIGHT="${GFLOW_MISSING_SECOND_WEIGHT:-1.0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_IDS=(${GPU_IDS:-0 1 2 3 4 5 6 7})
SEEDS=(${SEEDS:-42 123 2026})
MAX_JOBS="${MAX_JOBS:-${#GPU_IDS[@]}}"
GPU_BUSY_MAX_USED_MB="${GPU_BUSY_MAX_USED_MB:-2048}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
JOB_INDEX=0
FAILED=0
declare -A GPU_PIDS=()

CORE_CONFIGS=(
  "gflow_sa_unetv2_cond second conditional_learned_conserved_sa_unetv2"
  "gflow_sa_unetv2_cond_no_cons second conditional_sa_no_conservation"
  "gflow_sa_unetv2_cond_uniform second conditional_sa_fixed_uniform"
  "gflow_sa_unetv2_cond_random second conditional_sa_fixed_random"
  "gflow_sa_unetv2_cond_randinit second conditional_sa_random_init_learned"
  "multihead_sa_unetv2 second ordinary_multihead_sa_unetv2"
  "multihead_unet second ordinary_multihead_unet"
  "gflow_sa_unetv2 second static_learned_conserved_sa_unetv2_supplement"
  "unet random_primary single_head_unet_random_primary"
  "sa_unetv2 random_primary single_head_sa_unetv2_random_primary"
)

wait_for_slot() {
  while [ "$(jobs -rp | wc -l || true)" -ge "${MAX_JOBS}" ]; do
    sleep 10
  done
}

gpu_memory_used_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
    | head -n 1 \
    | tr -dc '0-9'
}

wait_for_gpu() {
  local gpu="$1"
  local pid="${GPU_PIDS[${gpu}]:-}"
  local status=0

  if [ -n "${pid}" ]; then
    set +e
    wait "${pid}"
    status=$?
    set -e
    if [ "${status}" -ne 0 ]; then
      FAILED=1
      echo "[warn] previous job on gpu=${gpu} pid=${pid} exited with status=${status}"
    fi
    unset "GPU_PIDS[${gpu}]"
  fi

  local used
  while true; do
    used="$(gpu_memory_used_mb "${gpu}")"
    if [ -z "${used}" ] || [ "${used}" -le "${GPU_BUSY_MAX_USED_MB}" ]; then
      break
    fi
    echo "[wait-gpu] gpu=${gpu} memory_used_mib=${used} threshold=${GPU_BUSY_MAX_USED_MB}"
    sleep "${GPU_POLL_SECONDS}"
  done
}

job_running_for_output() {
  local output_dir="$1"
  pgrep -f -- "${output_dir}" >/dev/null 2>&1
}

run_job() {
  local dataset="$1"
  local model="$2"
  local bias="$3"
  local tag="$4"
  local seed="$5"
  local gpu="${GPU_IDS[$((JOB_INDEX % ${#GPU_IDS[@]}))]}"
  JOB_INDEX=$((JOB_INDEX + 1))

  local output_dir="../${OUT_ROOT}/${dataset}/${model}_${bias}_${tag}_seed${seed}"
  local metrics_path="${OUT_ROOT}/${dataset}/${model}_${bias}_${tag}_seed${seed}/metrics.json"
  local log_path="${LOG_DIR}/${dataset}_${model}_${bias}_${tag}_seed${seed}.log"

  if [ -f "${metrics_path}" ]; then
    echo "[skip] existing metrics ${metrics_path}"
    return
  fi
  if job_running_for_output "${output_dir}"; then
    echo "[skip] already running output_dir=${output_dir}"
    return
  fi

  wait_for_slot
  wait_for_gpu "${gpu}"
  echo "[launch] dataset=${dataset} model=${model} bias=${bias} seed=${seed} gpu=${gpu} log=${log_path}"
  if [ "${dataset}" = "drive" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" code/train_drive.py \
      --model-name "${model}" \
      --data-root "${DATA_ROOT}" \
      --input-mode green_clahe \
      --loss-mode bce_dice \
      --label-bias-mode "${bias}" \
      --seed "${seed}" \
      --epochs 300 \
      --plateau-patience 20 \
      --early-stop-patience 60 \
      --output-dir "${output_dir}" \
      > "${log_path}" 2>&1 &
    GPU_PIDS["${gpu}"]="$!"
  elif [ "${dataset}" = "chase" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" code/train_chase.py \
      --model-name "${model}" \
      --data-root "${DATA_ROOT}" \
      --input-mode green_clahe \
      --loss-mode bce_dice \
      --label-bias-mode "${bias}" \
      --seed "${seed}" \
      --epochs 300 \
      --batch-size 24 \
      --patches-per-image 8 \
      --plateau-patience 20 \
      --early-stop-patience 60 \
      --output-dir "${output_dir}" \
      > "${log_path}" 2>&1 &
    GPU_PIDS["${gpu}"]="$!"
  else
    echo "Unsupported dataset: ${dataset}" >&2
    exit 1
  fi
}

for seed in "${SEEDS[@]}"; do
  for dataset in drive chase; do
    for config in "${CORE_CONFIGS[@]}"; do
      # shellcheck disable=SC2086
      run_job "${dataset}" ${config} "${seed}"
    done
  done
done

set +e
wait
WAIT_STATUS=$?
set -e
if [ "${FAILED}" -ne 0 ] && [ "${WAIT_STATUS}" -eq 0 ]; then
  WAIT_STATUS=1
fi
"${PYTHON_BIN}" code/summarize_bias_core.py --root-dir "${OUT_ROOT}" --output-csv "${OUT_ROOT}/comparison.csv"
echo "[done] ${OUT_ROOT}/comparison.csv and ${OUT_ROOT}/aggregate.csv"
exit "${WAIT_STATUS}"
