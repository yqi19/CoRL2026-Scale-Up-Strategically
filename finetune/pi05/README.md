# π0.5 and π0 fine-tuning (OpenPI)

Fine-tunes π0.5 (`pi05_maniskill`) and π0 (`pi0_maniskill`) on one ManiSkill Panda LeRobot dataset with the OpenPI PyTorch trainer (`scripts/train_pytorch.py`). One run per dataset.

## Contents

| Path | What it is |
| --- | --- |
| `finetune.sh` | Computes norm stats, then launches `torchrun` training |
| `patches/openpi-e6b0441.patch` | Changes to five upstream OpenPI files (see below) |
| `openpi_overrides/src/openpi/policies/maniskill_policy.py` | New file: `ManiskillInputs` / `ManiskillOutputs` transforms |

## 1. Set up OpenPI

Our code was based on OpenPI commit `e6b0441` ("Implement gsutil download for GCS URLs (#901)", 2026-03-13). Every upstream file present in our copy matched this commit except the five in the patch.

```bash
git clone https://github.com/Physical-Intelligence/openpi
cd openpi
git checkout e6b044115c075dc64a2e4e23d33ea3a3c293f123
git apply /path/to/finetune/pi05/patches/openpi-e6b0441.patch
cp -r /path/to/finetune/pi05/openpi_overrides/* .

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install datasets==3.0.0
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

The last line is the transformers patch that the upstream PyTorch backend requires. If `av` fails to build from source during `uv sync`, adding `"av>=15.0.0"` to `override-dependencies` in `pyproject.toml` lets uv use prebuilt wheels.

The patch:
- `src/openpi/training/config.py`: `LeRobotManiskillDataConfig` and the `pi05_maniskill` / `pi0_maniskill` train configs. When `repo_id` is a local path, the norm-stats `asset_id` is a hash of the resolved path, so different datasets do not overwrite each other's stats.
- `src/openpi/training/data_loader.py`: load a LeRobot dataset from a local directory (passed as `root=`) when `repo_id` is a path.
- `scripts/compute_norm_stats.py`: accept `--data.repo_id` to override the dataset, and write stats where training reads them (`assets/<config>/<asset_id>`).
- `src/openpi/policies/policy_config.py`: when serving a checkpoint of a config without a fixed dataset (the ManiSkill configs), load the norm stats from the single directory under `<checkpoint>/assets/`.
- `scripts/train_pytorch.py`: when loading the base `model.safetensors`, strip a `model.` key prefix if present and fill a missing tied `embed_tokens` weight from `lm_head`.

## 2. Dataset

Use the LeRobot datasets written by `data_collection/` in this repo (`collection_strategy/convert_maniskill_to_lerobot.py`), e.g. `<record-base>/<experiment>/<segment>/huggingface_data/`. Pass the dataset directory as `DATA_PATH`. Expected features (30 fps, images stored in the parquet files):

| Key | Shape | Meaning |
| --- | --- | --- |
| `image` | 256x256x3 | front camera |
| `wrist_image` | 256x256x3 | wrist camera |
| `state` | 8 float32 | 7 Panda joint positions + gripper |
| `actions` | 8 float32 | 7 absolute joint targets + gripper command |
| `task` | string | language instruction, used as the prompt (`prompt_from_task=True`) |

`LeRobotManiskillDataConfig` repacks these to `observation/image`, `observation/wrist_image`, `observation/state`, `actions`, `prompt`, the same keys the evaluation client sends (`evaluation/pi0`). `ManiskillInputs` maps the front camera to `base_0_rgb`, the wrist camera to `left_wrist_0_rgb`, and masks out `right_wrist_0_rgb`. The 7 joint dims of the action are converted to deltas relative to the current state for training and back to absolute targets at inference; the gripper stays absolute. `ManiskillOutputs` returns the first 8 action dims.

## 3. Base weights

The PyTorch trainer loads `<pytorch_weight_path>/model.safetensors`. Defaults in the configs (relative to the OpenPI root):

- `pi05_maniskill`: `ckpt/pi05-libero`, PyTorch weights of OpenPI's π0.5-LIBERO checkpoint (`gs://openpi-assets/checkpoints/pi05_libero`).
- `pi0_maniskill`: `ckpt/pi0_base_pytorch`, PyTorch weights of OpenPI's π0 base checkpoint (`gs://openpi-assets/checkpoints/pi0_base`).

Convert a downloaded JAX checkpoint with the upstream script:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir <local copy of gs://openpi-assets/checkpoints/pi0_base> \
    --config_name pi0_maniskill \
    --output_path ckpt/pi0_base_pytorch \
    --precision bfloat16
```

Use `--config_name pi05_maniskill --output_path ckpt/pi05-libero` with the `pi05_libero` checkpoint for π0.5, or set `BASE_CKPT` in `finetune.sh` to any directory with `model.safetensors`.

## 4. Fine-tune

```bash
# pi0.5
OPENPI_DIR=/path/to/openpi \
DATA_PATH=/path/to/color_spatial/stair_f18/huggingface_data \
EXP_NAME=color_spatial_stair_f18 \
NUM_GPUS=8 \
bash finetune.sh

# pi0
CONFIG=pi0_maniskill EXP_NAME=color_spatial_stair_f18_pi0 \
OPENPI_DIR=/path/to/openpi DATA_PATH=/path/to/color_spatial/stair_f18/huggingface_data \
bash finetune.sh
```

`finetune.sh` runs, from `OPENPI_DIR`:

```bash
uv run scripts/compute_norm_stats.py $CONFIG --data.repo_id $DATA_PATH
uv run torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS scripts/train_pytorch.py $CONFIG \
    --exp_name $EXP_NAME --data.repo_id $DATA_PATH --checkpoint_base_dir $OUTPUT_DIR
```

Checkpoints are written to `$OUTPUT_DIR/$CONFIG/$EXP_NAME/<step>/` (default `OUTPUT_DIR` is `$OPENPI_DIR/checkpoints`) and include the norm stats, so they can be served directly with `evaluation/pi0/start_server.sh`. W&B runs in offline mode unless `WANDB_MODE` is set. To resume, rerun the `torchrun` command with `--resume`.

## Hyperparameters

From the `pi05_maniskill` / `pi0_maniskill` configs and the run scripts; the two configs differ only in the model and base weights.

| | `pi05_maniskill` | `pi0_maniskill` |
| --- | --- | --- |
| Model | `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)` | `Pi0Config(action_horizon=10, discrete_state_input=False)` |
| PyTorch base weights | π0.5-LIBERO (`ckpt/pi05-libero`) | π0 base (`ckpt/pi0_base_pytorch`) |
| GPUs | 8 (1 node, DDP) | 8 (1 node, DDP) |
| Global batch size | 256 (32 per GPU) | 256 (32 per GPU) |
| Training steps | 18,000 | 18,000 |
| Checkpoint interval | 6,000 steps | 6,000 steps |
| Optimizer | AdamW, gradient clip norm 1.0 (other AdamW fields at OpenPI defaults) | same |
| LR schedule | linear warmup 10,000 steps to 5e-5, then cosine decay to 5e-5 over 1,000,000 steps (constant 5e-5 after warmup) | same |
| Action horizon | 10 | 10 |
| Action / state dim | 8 (padded to the model's 32) | 8 (padded to the model's 32) |
| Precision | bfloat16 (`pytorch_training_precision` default) | same |
| EMA | `ema_decay=0.999` in the config; the PyTorch trainer does not apply EMA | same |
| Prompt | episode `task` string | same |

The configs also set a JAX `weight_loader` (`pi05_base` / `pi0_base` params on `gs://openpi-assets`); it is only used by the JAX trainer `scripts/train.py`, not by the PyTorch runs above.

## Checkpoints

Fine-tuned checkpoints are linked from the [project page](https://yqi19.github.io/scale-up-strategically/).
