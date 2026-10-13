#!/usr/bin/env bash
# GR00T N1.7 fine-tuning on one ManiSkill Panda factor-pair dataset.
# Hyperparameters are the ones recorded in configs/color_object/conf.yaml.
#
# Usage:
#   GR00T_DIR=/path/to/Isaac-GR00T DATA_PATH=/path/to/gr00t_data/color_object \
#   OUTPUT_DIR=/path/to/checkpoints/color_object bash finetune.sh
set -euo pipefail

: "${GR00T_DIR:?set GR00T_DIR to the Isaac-GR00T checkout (commit 4b1dca9, patched)}"
: "${DATA_PATH:?set DATA_PATH to a GR00T LeRobot v2 dataset (see README)}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for checkpoints}"
NUM_GPUS=${NUM_GPUS:-4}                 # paper runs: 4 GPUs, 1 node
BASE_MODEL=${BASE_MODEL:-nvidia/GR00T-N1.7-3B}
MASTER_PORT=${MASTER_PORT:-29500}
# Optional: GR00T_BACKBONE_PATH=<local Qwen3-VL-2B-Instruct snapshot> (see launch_finetune_local.py)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODALITY_CONFIG="$SCRIPT_DIR/gr00t_overrides/scripts/conflict_panda_config.py"
LAUNCHER="$SCRIPT_DIR/gr00t_overrides/scripts/launch_finetune_local.py"

export PYTHONPATH="$GR00T_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export NO_ALBUMENTATIONS_UPDATE=1
export DS_BUILD_OPS=0

mkdir -p "$OUTPUT_DIR"

# Normalization statistics (written to $DATA_PATH/meta/stats.json once)
if [ ! -f "$DATA_PATH/meta/stats.json" ]; then
    python "$GR00T_DIR/gr00t/data/stats.py" \
        --dataset_path "$DATA_PATH" \
        --embodiment_tag NEW_EMBODIMENT \
        --modality_config_path "$MODALITY_CONFIG"
fi

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    "$LAUNCHER" \
    --base_model_path "$BASE_MODEL" \
    --dataset_path "$DATA_PATH" \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path "$MODALITY_CONFIG" \
    --num_gpus "$NUM_GPUS" \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 32 \
    --dataloader_num_workers 4 \
    --shard_size 1024 \
    --num_shards_per_epoch 100000 \
    --episode_sampling_rate 0.1 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --save_only_model
