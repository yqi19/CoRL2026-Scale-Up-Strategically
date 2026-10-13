#!/usr/bin/env bash
# Fine-tune pi0.5 (pi05_maniskill) or pi0 (pi0_maniskill) on one ManiSkill LeRobot dataset with the OpenPI
# PyTorch trainer: compute normalization stats, then run DDP training on one node.
#
# Usage:
#   OPENPI_DIR=/path/to/openpi \
#   DATA_PATH=/path/to/lerobot_dataset \
#   EXP_NAME=color_spatial_stair_f18 \
#   bash finetune.sh
#
# Variables:
#   OPENPI_DIR       OpenPI checkout at e6b0441 with our patch and overrides applied (required)
#   DATA_PATH        local LeRobot dataset directory or LeRobot repo id (required)
#   EXP_NAME         experiment name (required)
#   CONFIG           pi05_maniskill (default) or pi0_maniskill
#   BASE_CKPT        directory with the PyTorch base weights (model.safetensors);
#                    default: the config's pytorch_weight_path, relative to OPENPI_DIR
#   OUTPUT_DIR       checkpoint root; checkpoints go to $OUTPUT_DIR/$CONFIG/$EXP_NAME/<step> (default: $OPENPI_DIR/checkpoints)
#   NUM_GPUS         GPUs on this node (default 8); the config batch size (256) is the global batch
#   NUM_TRAIN_STEPS  optional override of the config value (18000)
#   SAVE_INTERVAL    optional override of the config value (6000)
#   WANDB_MODE       default offline
set -euo pipefail

: "${OPENPI_DIR:?set OPENPI_DIR to the patched OpenPI checkout}"
: "${DATA_PATH:?set DATA_PATH to a LeRobot dataset directory or repo id}"
: "${EXP_NAME:?set EXP_NAME}"
CONFIG="${CONFIG:-pi05_maniskill}"
NUM_GPUS="${NUM_GPUS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${OPENPI_DIR}/checkpoints}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

# Resolve local paths before changing directory.
if [[ -e "${DATA_PATH}" ]]; then
  DATA_PATH="$(cd "${DATA_PATH}" && pwd)"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

EXTRA_ARGS=()
if [[ -n "${BASE_CKPT:-}" ]]; then
  EXTRA_ARGS+=(--pytorch_weight_path "$(cd "${BASE_CKPT}" && pwd)")
fi
if [[ -n "${NUM_TRAIN_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--num_train_steps "${NUM_TRAIN_STEPS}")
fi
if [[ -n "${SAVE_INTERVAL:-}" ]]; then
  EXTRA_ARGS+=(--save_interval "${SAVE_INTERVAL}")
fi

cd "${OPENPI_DIR}"

echo "Computing norm stats: config=${CONFIG} data=${DATA_PATH}"
uv run scripts/compute_norm_stats.py "${CONFIG}" --data.repo_id "${DATA_PATH}"

echo "Training: config=${CONFIG} exp_name=${EXP_NAME} gpus=${NUM_GPUS} output=${OUTPUT_DIR}"
uv run torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  scripts/train_pytorch.py "${CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --data.repo_id "${DATA_PATH}" \
  --checkpoint_base_dir "${OUTPUT_DIR}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
