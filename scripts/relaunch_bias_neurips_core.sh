#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/gushengda/retina-sa-unet-codex"
cd "${ROOT_DIR}"

tmux kill-session -t bias_neurips_core 2>/dev/null || true

stamp="${1:-$(date +%Y%m%d_%H%M%S)_stalled}"
if [ -d output_bias_neurips_core ]; then
  mv output_bias_neurips_core "output_bias_neurips_core_${stamp}"
fi

mkdir -p output_bias_neurips_core
tmux new-session -d -s bias_neurips_core \
  "cd ${ROOT_DIR} && bash scripts/run_bias_neurips_core.sh > output_bias_neurips_core/launcher.log 2>&1"
tmux list-sessions
