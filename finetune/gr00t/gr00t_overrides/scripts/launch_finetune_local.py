#!/usr/bin/env python3
"""
Wrapper around gr00t/experiment/launch_finetune.py (Isaac-GR00T @ 4b1dca9) that:
  1. Overrides config.model.model_name with a Qwen3-VL-2B-Instruct backbone
     (same architecture as the gated nvidia/Cosmos-Reason2-2B; the actual
     backbone weights are loaded from the GR00T-N1.7-3B checkpoint).
     Set GR00T_BACKBONE_PATH to a local snapshot dir or a HF repo id.
  2. Forces the pyav video backend and relative actions.
  3. Prints the checkpoint output directory at start and end.
"""

import json
import os
import sys
from pathlib import Path

import tyro

# Make sure gr00t is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# The paper runs used a local snapshot of Qwen/Qwen3-VL-2B-Instruct
# (revision 89644892e4d85e24eaac8bacfd4f463576704203).
LOCAL_BACKBONE = os.environ.get("GR00T_BACKBONE_PATH", "Qwen/Qwen3-VL-2B-Instruct")


def load_modality_config(modality_config_path: str):
    import importlib
    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.insert(0, str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config not found: {modality_config_path}")


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(FinetuneConfig)

    from gr00t.data.embodiment_tags import EmbodimentTag
    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False

    # Use local Qwen3-VL-2B-Instruct as backbone (same architecture as Cosmos-Reason2-2B)
    # Actual backbone weights come from the GR00T-N1.7-3B checkpoint safetensors.
    config.model.model_name = LOCAL_BACKBONE
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project
    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading
    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    # Use pyav backend: fast, pure-Python, no FFmpeg 7.0 soname requirement
    config.data.video_backend = "pyav"
    # Local-only file resolution avoids gated HF repo network calls. The paper
    # runs always set this to True (backbone was a local snapshot directory).
    config.training.transformers_local_files_only = os.path.isdir(LOCAL_BACKBONE)

    output_dir = Path(ft_config.output_dir).resolve()
    print("=" * 64)
    print("  CHECKPOINT OUTPUT DIRECTORY:")
    print(f"  {output_dir}")
    print(f"  Checkpoints saved every {ft_config.save_steps} steps")
    print(f"  Max checkpoints kept: {ft_config.save_total_limit}")
    print("  Final checkpoint will be at:")
    print(f"  {output_dir}/checkpoint-<step>/")
    print("=" * 64)

    run(config)

    print("=" * 64)
    print("  Training complete.")
    print(f"  Checkpoint saved at: {output_dir}")
    ckpts = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
    for ckpt in ckpts:
        print(f"    {ckpt}")
    print("=" * 64)
