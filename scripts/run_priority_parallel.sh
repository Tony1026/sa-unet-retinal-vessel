#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data1/gushengda/retina-sa-unet-codex}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/datasets}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output_priority}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/priority}"
EPOCHS="${EPOCHS:-300}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
MAX_PARALLEL="${MAX_PARALLEL:-7}"
GPU_LIST="${GPU_LIST:-1 2 3 4 5 6 7}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
cd "${ROOT}"

source /data1/gushengda/anaconda3/etc/profile.d/conda.sh
conda activate hci
export PYTHONPATH="${ROOT}/.deps:${PYTHONPATH:-}"

declare -a JOBS=(
  "drive unet green_clahe bce_dice"
  "drive sa_unet green_clahe bce_dice"
  "drive sa_unetv2 green_clahe bce_dice"
  "chase unet green_clahe bce_dice"
  "chase sa_unet green_clahe bce_dice"
  "chase sa_unetv2 green_clahe bce_dice"
  "drive unet green bce_dice"
  "drive unet rgb bce_dice"
  "drive unet green_clahe bce"
  "chase unet green bce_dice"
  "chase unet rgb bce_dice"
  "chase unet green_clahe bce"
)

gpu_for_slot() {
  local slot="$1"
  local i=0
  for gpu in ${GPU_LIST}; do
    if [[ "${i}" -eq "${slot}" ]]; then
      echo "${gpu}"
      return
    fi
    i=$((i + 1))
  done
  echo "1"
}

running=0
slot=0
for job in "${JOBS[@]}"; do
  read -r dataset model input_mode loss_mode <<< "${job}"
  gpu="$(gpu_for_slot "${slot}")"
  name="${dataset}_${model}_${input_mode}_${loss_mode}"
  log_path="${LOG_ROOT}/${name}.log"
  out_dir="${OUT_ROOT}/${dataset}/${model}_${input_mode}_${loss_mode}"

  if [[ "${dataset}" == "drive" ]]; then
    cmd=(python code/train_drive.py)
  else
    cmd=(python code/train_chase.py)
  fi

  echo "[launch] ${name} gpu=${gpu} log=${log_path}"
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" \
      --model-name "${model}" \
      --data-root "${DATA_ROOT}" \
      --output-dir "${out_dir}" \
      --input-mode "${input_mode}" \
      --loss-mode "${loss_mode}" \
      --device cuda \
      --epochs "${EPOCHS}" \
      --num-workers "${NUM_WORKERS}" \
      --seed "${SEED}"
  ) > "${log_path}" 2>&1 &

  running=$((running + 1))
  slot=$(((slot + 1) % MAX_PARALLEL))
  if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
python code/summarize_metrics.py --root "${OUT_ROOT}" --output "${OUT_ROOT}/summary.csv" > "${OUT_ROOT}/summary_stdout.csv"
echo "[done] summary=${OUT_ROOT}/summary.csv"
