# GR00T N1.7 fine-tuning

Fine-tunes `nvidia/GR00T-N1.7-3B` on one ManiSkill Panda factor-pair dataset (e.g. `color_object`). One run per factor pair.

## Contents

| Path | What it is |
| --- | --- |
| `finetune.sh` | Launch script with the exact hyperparameters of the paper runs |
| `gr00t_overrides/scripts/conflict_panda_config.py` | Modality config for the Panda data (`NEW_EMBODIMENT`) |
| `gr00t_overrides/scripts/launch_finetune_local.py` | Wrapper around upstream `gr00t/experiment/launch_finetune.py` (backbone path, pyav video backend, relative actions) |
| `gr00t_overrides/scripts/prepare_conflict_data.py` | Converts our ManiSkill datasets to GR00T LeRobot v2 (PNG frames to MP4, column renames, `modality.json`) |
| `patches/isaac-gr00t-4b1dca9.patch` | Two small changes to upstream files (see below) |
| `configs/modality.json` | `meta/modality.json` written by the converter |
| `configs/color_object/conf.yaml` | Training and data config of the `color_object` run (paths replaced by placeholders) |

## 1. Set up Isaac-GR00T

Our code was based on Isaac-GR00T commit `4b1dca9` ("Readme Updates and Total Task Correction", 2026-04-21, one commit after the `n1.7-release` tag). Every upstream file present in our copy matched this commit except the two in the patch.

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
git checkout 4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e
git apply /path/to/finetune/gr00t/patches/isaac-gr00t-4b1dca9.patch
# then install following the upstream README (uv / flash-attn 2.7.4.post1)
```

The patch:
- `gr00t/model/gr00t_n1d7/setup.py`: load the processor with `Gr00tN1d7Processor.from_pretrained` when `start_from_checkpoint` is a local directory (avoids network calls for the gated backbone repo).
- `gr00t/utils/video_utils.py`: add a `pyav` branch to the frame-by-index decoder (used because `launch_finetune_local.py` sets `video_backend = "pyav"`).

The scripts in `gr00t_overrides/` are run from this folder; `finetune.sh` puts `$GR00T_DIR` on `PYTHONPATH`. You can also copy them into `Isaac-GR00T/scripts/`.

## 2. Dataset

`data_collection/` in this repo writes one LeRobot dataset per factor pair (parquet with embedded PNG images). Convert it to GR00T LeRobot v2:

```bash
python gr00t_overrides/scripts/prepare_conflict_data.py \
    --src_root <data_collection output>/demo_conflict \
    --out_root <DATA_ROOT> \
    --num_demos 300 --fps 30 \
    --categories color_object   # default: all 10 factor pairs
```

It reads `<src_root>/<pair>/<num_demos>/huggingface_data/<pair>/conflict` and writes `<out_root>/<pair>/`. The paper data used `--num_demos 300 --fps 30`. The ten factor pairs are `color_object, color_size, color_spatial, size_object, spatial_object, spatial_size, verb_color, verb_object, verb_size, verb_spatial`.

Format of the result (keys expected by `conflict_panda_config.py`):
- `observation.state`, `action`: float32, 8-dim = 7 Panda joint positions + gripper width. Modality keys `arm` = [0:7], `gripper` = [7:8].
- Videos `observation.images.image` (front camera) and `observation.images.wrist_image` (wrist camera), 256x256 MP4, mapped to video keys `image` and `wrist_image`.
- `annotation.human.task_description`: the task index, resolved to the instruction through `meta/tasks.jsonl`.
- Embodiment tag `NEW_EMBODIMENT`. State: 1 frame. Action: 16 steps (`delta_indices` 0..15); `arm` is RELATIVE (joint deltas), `gripper` is ABSOLUTE.

## 3. Fine-tune

```bash
GR00T_DIR=/path/to/Isaac-GR00T \
DATA_PATH=<DATA_ROOT>/color_object \
OUTPUT_DIR=<OUTPUT_ROOT>/color_object \
NUM_GPUS=4 \
bash finetune.sh
```

`finetune.sh` first computes `meta/stats.json` with `gr00t/data/stats.py` if missing, then runs `torchrun` on `launch_finetune_local.py`. The backbone `model_name` is set to `Qwen/Qwen3-VL-2B-Instruct` (the paper runs used a local snapshot at revision `89644892e4d85e24eaac8bacfd4f463576704203`; set `GR00T_BACKBONE_PATH` to a local snapshot to run offline). Backbone weights come from the GR00T checkpoint.

Hyperparameters (from `configs/color_object/conf.yaml`):

| | |
| --- | --- |
| Base model | `nvidia/GR00T-N1.7-3B` |
| GPUs | 4 (1 node), DeepSpeed stage 2 |
| Global batch size | 32 (grad. accumulation 1) |
| Max steps | 10,000 |
| Optimizer | `adamw_torch`, lr 1e-4, cosine schedule, warmup ratio 0.05, weight decay 1e-5, max grad norm 1.0 |
| Precision | bf16, tf32 |
| Trainable | projector, diffusion action head, VL layernorm; LLM and vision encoder frozen |
| State dropout | 0.2 |
| Color jitter | brightness 0.3, contrast 0.4, saturation 0.5, hue 0.08 |
| Data sampling | shard size 1024, episode sampling rate 0.1, 100,000 shards per epoch, seed 42 |
| Action chunk | 16 steps (`delta_indices` 0..15); model `action_horizon` is 40 |
| Checkpointing | every 1000 steps, keep last 1, model only |
| Dataloader workers | 4 |

Model architecture fields are loaded from the base checkpoint.

## Checkpoints

Fine-tuned checkpoints are linked from the [project page](https://yqi19.github.io/scale-up-strategically/).
