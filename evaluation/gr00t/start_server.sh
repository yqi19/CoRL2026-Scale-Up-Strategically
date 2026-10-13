#!/usr/bin/env bash
set -euo pipefail
# Start a GR00T N1.7 zmq policy server for one fine-tuned checkpoint.
# Runs in the foreground; start it in its own terminal (or background it).
#
# Usage: bash gr00t/start_server.sh <model_path> [port=5555]
# Env:
#   GR00T_REPO    path to a checkout of NVIDIA Isaac-GR00T (required)
#   GR00T_PYTHON  python of the GR00T environment (default: python)
#   HOST          bind address (default 127.0.0.1)
#   CUDA_VISIBLE_DEVICES  GPU to load the model on
MODEL_PATH="${1:?model_path (folder with the fine-tuned *.safetensors)}"
PORT="${2:-5555}"
: "${GR00T_REPO:?set GR00T_REPO to your Isaac-GR00T checkout}"
GR00T_PYTHON="${GR00T_PYTHON:-python}"
HOST="${HOST:-127.0.0.1}"

cd "${GR00T_REPO}"
NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false \
exec "${GR00T_PYTHON}" -m gr00t.eval.run_gr00t_server \
  --model-path "${MODEL_PATH}" \
  --embodiment-tag new_embodiment \
  --device cuda:0 \
  --host "${HOST}" \
  --port "${PORT}"
