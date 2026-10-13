#!/usr/bin/env python3
"""
Conflict (FDR) rollouts for a fine-tuned pi0 / pi0.5 policy served by OpenPI.

Identical to ``gr00t/groot_conflict_eval.py`` (same scene construction, job
format, dual-axis success, video layout and results file); only the policy
boundary differs: the OpenPI websocket policy server
(``scripts/serve_policy.py``) is queried through ``openpi_client``.

Batch mode (used by ../common/run_conflict.sh): one process, all jobs from a
JSON file executed sequentially against one policy server.
"""
from __future__ import annotations

import collections
import dataclasses
import datetime as _dt
import logging
import pathlib
import sys

import gymnasium as gym
import numpy as np
import tqdm
import tyro

# repo roots: ManiSkill fork + collection_strategy (see common/paths.py)
_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from paths import setup_paths  # noqa: E402

setup_paths()

import mani_skill.envs  # noqa: F401 - registers VerbObjectColor-v1

from collection_strategy.lib.pairwise_task_language import VERB_TO_EN
from collection_strategy.lib.training_vocab import (
    THIRD_COLORS_FOR_VERB_OBJECT,
    THIRD_OBJECTS_FOR_VERB_COLOR,
    THIRD_VERBS_FOR_COLOR_OBJECT,
    TRAINING_COLORS,
    TRAINING_SHAPES,
    TRAINING_VERBS,
)
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

COLOR_TO_ID = {c: i for i, c in enumerate(TRAINING_COLORS)}

# size / spatial vocabularies (shared by all policies)
SIZES: tuple[str, ...] = ("small", "large", "smaller", "larger", "smallest", "largest")
SIZE_SCALES: dict[str, float] = {
    "small": 0.72, "large": 1.34,
    "smaller": 0.82, "larger": 1.18,
    "smallest": 0.78, "largest": 1.26,
}
SPATIALS: tuple[str, ...] = ("left", "right", "middle", "front", "behind")
SPATIAL_ANCHORS: dict[str, tuple[float, float]] = {
    "left": (-0.10, 0.00), "right": (0.10, 0.00),
    "middle": (0.00, 0.00), "front": (0.00, 0.10), "behind": (0.00, -0.10),
}
SPATIAL_TO_PHRASE: dict[str, str] = {
    "left": "on the left", "right": "on the right",
    "middle": "in the middle", "front": "in front", "behind": "at the back",
}
_VERB_CAPS: dict[str, str] = {
    "lift": "Lift", "grasp": "Grasp", "push": "Push",
    "pull": "Pull", "rotate": "Rotate", "slide": "Slide",
}


# Args

@dataclasses.dataclass
class Args:
    # OpenPI policy server
    host: str = "127.0.0.1"
    port: int = 8000
    replan_steps: int = 5
    """Execute this many env steps before querying the model again."""

    # OOD pair spec
    experiment: str = "verb_color"
    """verb_color | verb_object | color_object | verb_size | verb_spatial |
    size_object | color_size | color_spatial | spatial_size | spatial_object."""
    pair_i: int = 0
    pair_j: int = 1
    run_type: str = "verb"
    third_seed: int = 0
    third_indices: str = "0,1"
    """Comma-separated indices into THIRD_* list (default '0,1' matches conflict training)."""

    # Episode settings
    num_episodes: int = 20
    max_episode_steps: int = 300
    sim_backend: str = "gpu"
    seed: int = 0
    task_difficulty: float = 1.0
    """VerbObjectColor-v1 difficulty (default 1.0; >1 harder, clamped [0.5,3.0])."""

    # Output
    experiment_root: str = "data/conflict_pi0/experiments"
    experiment_name: str = ""
    save_wrist_video: bool = True

    # Batch mode (server loaded once, all runs executed in-process)
    batch_jobs_file: str = ""
    batch_results_txt: str = ""
    batch_skip_to: int = 0


# Helpers (shared by all policies)

def _to_hwc_uint8(x) -> np.ndarray:
    try:
        import torch
        if torch.is_tensor(x):
            x = x.detach().float().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.0 + 1e-6:
        x = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8)
    return np.ascontiguousarray(x)


def _state8(env: gym.Env) -> np.ndarray:
    qpos = env.unwrapped.agent.robot.get_qpos()
    try:
        import torch
        if torch.is_tensor(qpos):
            qpos = qpos[0].detach().cpu().numpy()
    except Exception:
        pass
    qpos = np.asarray(qpos, dtype=np.float32).ravel()
    out = np.zeros(8, dtype=np.float32)
    out[: min(8, len(qpos))] = qpos[: min(8, len(qpos))]
    return out


def _bool_info(info: dict, key: str) -> bool:
    v = info.get(key, False)
    try:
        import torch
        if torch.is_tensor(v):
            return bool(v.squeeze().item())
    except Exception:
        pass
    return bool(np.asarray(v).squeeze())


def _parse_third_pool(full: tuple, spec: str) -> tuple:
    idxs = [int(x.strip()) for x in spec.split(",") if x.strip()]
    return tuple(full[i] for i in idxs)


# Environment factory (shared by all policies)

def _build_env_and_instruction(args: Args) -> tuple[gym.Env, str]:
    """Create VerbObjectColor-v1 for the given OOD pair. Returns (env, instruction)."""
    import random as _random
    rng = _random.Random(args.third_seed)

    i, j = args.pair_i, args.pair_j
    assert i != j, f"pair_i must differ from pair_j, got ({i}, {j})"

    verb_i = TRAINING_VERBS[i]
    verb_j = TRAINING_VERBS[j]

    _shape_pool = _parse_third_pool(THIRD_OBJECTS_FOR_VERB_COLOR, args.third_indices)
    _color_pool = _parse_third_pool(THIRD_COLORS_FOR_VERB_OBJECT, args.third_indices)
    _verb_pool  = _parse_third_pool(THIRD_VERBS_FOR_COLOR_OBJECT, args.third_indices)

    make_kw: dict = dict(
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend=args.sim_backend,
        render_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps,
        task_difficulty=args.task_difficulty,
    )

    if args.experiment == "verb_color":
        shape = rng.choice(_shape_pool)
        color_i = TRAINING_COLORS[i];  color_j = TRAINING_COLORS[j]
        instruction = VERB_TO_EN[verb_i].format(color=color_j, shape=shape)
        if args.run_type == "verb":
            make_kw.update(
                verb=verb_i, object_shape=shape, object_color_id=COLOR_TO_ID[color_i],
                distractor_max=3, distractor_specs=[(shape, COLOR_TO_ID[color_j]), None, None],
            )
        elif args.run_type == "color":
            make_kw.update(
                verb=verb_j, object_shape=shape, object_color_id=COLOR_TO_ID[color_j],
                distractor_max=3, distractor_specs=[(shape, COLOR_TO_ID[color_i]), None, None],
            )
        else:
            raise ValueError(f"run_type must be 'verb' or 'color' for verb_color, got {args.run_type!r}")

    elif args.experiment == "verb_object":
        color = rng.choice(_color_pool)
        shape_i = TRAINING_SHAPES[i];  shape_j = TRAINING_SHAPES[j]
        instruction = VERB_TO_EN[verb_i].format(color=color, shape=shape_j)
        if args.run_type == "verb":
            make_kw.update(
                verb=verb_i, object_shape=shape_i, object_color_id=COLOR_TO_ID[color],
                distractor_max=3, distractor_specs=[(shape_j, COLOR_TO_ID[color]), None, None],
            )
        elif args.run_type == "shape":
            make_kw.update(
                verb=verb_j, object_shape=shape_j, object_color_id=COLOR_TO_ID[color],
                distractor_max=3, distractor_specs=[(shape_i, COLOR_TO_ID[color]), None, None],
            )
        else:
            raise ValueError(f"run_type must be 'verb' or 'shape' for verb_object, got {args.run_type!r}")

    elif args.experiment == "color_object":
        third_verb = rng.choice(_verb_pool)
        color_i, shape_i = TRAINING_COLORS[i], TRAINING_SHAPES[i]
        color_j, shape_j = TRAINING_COLORS[j], TRAINING_SHAPES[j]
        instruction = VERB_TO_EN[third_verb].format(color=color_i, shape=shape_j)
        if args.run_type == "color":
            make_kw.update(
                verb=third_verb, object_shape=shape_i, object_color_id=COLOR_TO_ID[color_i],
                distractor_max=3, distractor_specs=[(shape_j, COLOR_TO_ID[color_j]), None, None],
            )
        elif args.run_type == "shape":
            make_kw.update(
                verb=third_verb, object_shape=shape_j, object_color_id=COLOR_TO_ID[color_j],
                distractor_max=3, distractor_specs=[(shape_i, COLOR_TO_ID[color_i]), None, None],
            )
        else:
            raise ValueError(f"run_type must be 'color' or 'shape' for color_object, got {args.run_type!r}")

    elif args.experiment == "verb_size":
        size_i = SIZES[i];   size_j = SIZES[j]
        scale_i = SIZE_SCALES[size_i]; scale_j = SIZE_SCALES[size_j]
        _superlative = (i // 2 == 2)
        third_shape = rng.choice(_shape_pool)
        instruction = f"{_VERB_CAPS[verb_i]} the {size_j} {third_shape}."
        if args.run_type == "verb":
            make_kw.update(
                verb=verb_i, object_shape=third_shape, object_color_id=0,
                target_size_scale=scale_i,
                distractor_specs=[(third_shape, 0), (third_shape, 0) if _superlative else None, None],
                distractor_size_scales=[scale_j, 1.00] if _superlative else [scale_j],
                distractor_max=2 if _superlative else 1, object_size_jiggle=0.0,
            )
        elif args.run_type == "size":
            make_kw.update(
                verb=verb_j, object_shape=third_shape, object_color_id=0,
                target_size_scale=scale_j,
                distractor_specs=[(third_shape, 0), (third_shape, 0) if _superlative else None, None],
                distractor_size_scales=[scale_i, 1.00] if _superlative else [scale_i],
                distractor_max=2 if _superlative else 1, object_size_jiggle=0.0,
            )
        else:
            raise ValueError(f"run_type must be 'verb' or 'size' for verb_size, got {args.run_type!r}")

    elif args.experiment == "verb_spatial":
        spatial_i = SPATIALS[i]; spatial_j = SPATIALS[j]
        third_shape = rng.choice(_shape_pool)
        instruction = f"{_VERB_CAPS[verb_i]} the {third_shape} {SPATIAL_TO_PHRASE[spatial_j]}."
        if args.run_type == "verb":
            make_kw.update(
                verb=verb_i, object_shape=third_shape, object_color_id=0,
                target_size_scale=1.0,
                distractor_specs=[(third_shape, 0), None, None],
                distractor_max=1, object_size_jiggle=0.0,
            )
        elif args.run_type == "spatial":
            make_kw.update(
                verb=verb_j, object_shape=third_shape, object_color_id=0,
                target_size_scale=1.0,
                distractor_specs=[(third_shape, 0), None, None],
                distractor_max=1, object_size_jiggle=0.0,
            )
        else:
            raise ValueError(f"run_type must be 'verb' or 'spatial' for verb_spatial, got {args.run_type!r}")

    elif args.experiment == "size_object":
        size_i = SIZES[i];   size_j = SIZES[j]
        shape_i = TRAINING_SHAPES[i]; shape_j = TRAINING_SHAPES[j]
        _superlative = (i // 2 == 2)
        third_verb = rng.choice(_verb_pool)
        instruction = f"{_VERB_CAPS[third_verb]} the {size_i} {shape_j}."
        if args.run_type == "size":
            make_kw.update(
                verb=third_verb, object_shape=shape_i, object_color_id=0,
                target_size_scale=SIZE_SCALES[size_i],
                distractor_specs=[(shape_j, 0), (shape_i, 0) if _superlative else None, None],
                distractor_size_scales=[SIZE_SCALES[size_j], 1.00] if _superlative else [SIZE_SCALES[size_j]],
                distractor_max=2 if _superlative else 1, object_size_jiggle=0.0,
            )
        elif args.run_type == "shape":
            make_kw.update(
                verb=third_verb, object_shape=shape_j, object_color_id=0,
                target_size_scale=SIZE_SCALES[size_j],
                distractor_specs=[(shape_i, 0), (shape_i, 0) if _superlative else None, None],
                distractor_size_scales=[SIZE_SCALES[size_i], 1.00] if _superlative else [SIZE_SCALES[size_i]],
                distractor_max=2 if _superlative else 1, object_size_jiggle=0.0,
            )
        else:
            raise ValueError(f"run_type must be 'size' or 'shape' for size_object, got {args.run_type!r}")

    elif args.experiment == "color_size":
        color_i = TRAINING_COLORS[i]; color_j = TRAINING_COLORS[j]
        size_i = SIZES[i];            size_j = SIZES[j]
        _superlative = (i // 2 == 2)
        third_verb = rng.choice(_verb_pool)
        if _superlative:
            _neutral_color_pool = [c for c in TRAINING_COLORS if c not in (color_i, color_j)]
            _neutral_color = rng.choice(_neutral_color_pool)
        instruction = f"{_VERB_CAPS[third_verb]} the {color_i} {size_j} cube."
        if args.run_type == "color":
            make_kw.update(
                verb=third_verb, object_shape="cube", object_color_id=COLOR_TO_ID[color_i],
                target_size_scale=SIZE_SCALES[size_i],
                distractor_specs=[("cube", COLOR_TO_ID[color_j]),
                                   ("cube", COLOR_TO_ID[_neutral_color]) if _superlative else None, None],
                distractor_size_scales=[SIZE_SCALES[size_j], 1.00] if _superlative else [SIZE_SCALES[size_j]],
                distractor_max=2 if _superlative else 1, object_size_jiggle=0.0,
            )
        elif args.run_type == "size":
            make_kw.update(
                verb=third_verb, object_shape="cube", object_color_id=COLOR_TO_ID[color_j],
                target_size_scale=SIZE_SCALES[size_j],
                distractor_specs=[("cube", COLOR_TO_ID[color_i]),
                                   ("cube", COLOR_TO_ID[_neutral_color]) if _superlative else None, None],
                distractor_size_scales=[SIZE_SCALES[size_i], 1.00] if _superlative else [SIZE_SCALES[size_i]],
                distractor_max=2 if _superlative else 1, object_size_jiggle=0.0,
            )
        else:
            raise ValueError(f"run_type must be 'color' or 'size' for color_size, got {args.run_type!r}")

    elif args.experiment == "color_spatial":
        color_i = TRAINING_COLORS[i]; color_j = TRAINING_COLORS[j]
        third_verb = rng.choice(_verb_pool)
        instruction = f"{_VERB_CAPS[third_verb]} the {color_i} cube {SPATIAL_TO_PHRASE[SPATIALS[j]]}."
        if args.run_type == "color":
            make_kw.update(
                verb=third_verb, object_shape="cube", object_color_id=COLOR_TO_ID[color_i],
                target_size_scale=1.0,
                distractor_specs=[("cube", COLOR_TO_ID[color_j]), None, None],
                distractor_max=1, object_size_jiggle=0.0,
            )
        elif args.run_type == "spatial":
            make_kw.update(
                verb=third_verb, object_shape="cube", object_color_id=COLOR_TO_ID[color_j],
                target_size_scale=1.0,
                distractor_specs=[("cube", COLOR_TO_ID[color_i]), None, None],
                distractor_max=1, object_size_jiggle=0.0,
            )
        else:
            raise ValueError(f"run_type must be 'color' or 'spatial' for color_spatial, got {args.run_type!r}")

    elif args.experiment == "spatial_size":
        size_i = SIZES[i]; size_j = SIZES[j]
        third_shape = rng.choice(_shape_pool)
        instruction = f"Lift the {size_j} {third_shape} {SPATIAL_TO_PHRASE[SPATIALS[i]]}."
        if args.run_type == "spatial":
            make_kw.update(
                verb="lift", object_shape=third_shape, object_color_id=0,
                target_size_scale=SIZE_SCALES[size_i],
                distractor_specs=[(third_shape, 0), None, None],
                distractor_size_scales=[SIZE_SCALES[size_j]],
                distractor_max=1, object_size_jiggle=0.0,
            )
        elif args.run_type == "size":
            make_kw.update(
                verb="lift", object_shape=third_shape, object_color_id=0,
                target_size_scale=SIZE_SCALES[size_j],
                distractor_specs=[(third_shape, 0), None, None],
                distractor_size_scales=[SIZE_SCALES[size_i]],
                distractor_max=1, object_size_jiggle=0.0,
            )
        else:
            raise ValueError(f"run_type must be 'spatial' or 'size' for spatial_size, got {args.run_type!r}")

    elif args.experiment == "spatial_object":
        shape_i = TRAINING_SHAPES[i]; shape_j = TRAINING_SHAPES[j]
        third_verb = rng.choice(_verb_pool)
        instruction = f"{_VERB_CAPS[third_verb]} the {shape_j} {SPATIAL_TO_PHRASE[SPATIALS[i]]}."
        if args.run_type == "spatial":
            make_kw.update(
                verb=third_verb, object_shape=shape_i, object_color_id=0,
                target_size_scale=1.0,
                distractor_specs=[(shape_j, 0), None, None],
                distractor_max=1, object_size_jiggle=0.0,
            )
        elif args.run_type == "shape":
            make_kw.update(
                verb=third_verb, object_shape=shape_j, object_color_id=0,
                target_size_scale=1.0,
                distractor_specs=[(shape_i, 0), None, None],
                distractor_max=1, object_size_jiggle=0.0,
            )
        else:
            raise ValueError(f"run_type must be 'spatial' or 'shape' for spatial_object, got {args.run_type!r}")

    else:
        raise ValueError(f"Unknown experiment {args.experiment!r}")

    env = gym.make("VerbObjectColor-v1", **make_kw)
    return env, instruction


# pi0 policy boundary (openpi websocket) - mirrors
# the OpenPI ManiSkill full-factor client

def _query_pi0(client, img_base: np.ndarray, img_wrist: np.ndarray,
                state8: np.ndarray, instruction: str,
                resize_size: int = 224) -> np.ndarray:
    """Resize cameras, query the openpi pi0 server, return an action chunk
    (T, action_dim); caller slices [:8] for the env."""
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img_base, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img_wrist, resize_size, resize_size))
    chunk = client.infer({
        "observation/image":       img,
        "observation/wrist_image": wrist,
        "observation/state":       state8.astype(np.float32),
        "prompt":                  instruction,
    })["actions"]
    return np.asarray(chunk, dtype=np.float32)


# Batch eval

def batch_eval_conflict(args: Args) -> None:
    import json

    logging.basicConfig(level=logging.INFO, force=True)

    if not args.batch_jobs_file:
        raise ValueError("--batch-jobs-file is required for batch mode")
    if not args.batch_results_txt:
        raise ValueError("--batch-results-txt is required for batch mode")

    all_jobs = json.loads(pathlib.Path(args.batch_jobs_file).read_text())
    if args.batch_skip_to > 0:
        jobs = all_jobs[args.batch_skip_to:]
        logging.info("Resuming from job %d (skipping first %d)",
                     args.batch_skip_to + 1, args.batch_skip_to)
    else:
        jobs = all_jobs
    results_path = pathlib.Path(args.batch_results_txt)
    total = len(all_jobs)
    logging.info("Batch mode: %d/%d jobs remaining, experiment=%s, server=%s:%d",
                 len(jobs), total, args.experiment, args.host, args.port)

    # Connect to the pi0 openpi websocket policy server (loaded once)
    import time as _t
    client = None
    for _ in range(120):
        try:
            client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
            break
        except Exception:
            _t.sleep(2.0)
    if client is None:
        raise RuntimeError(f"pi0 server not reachable at {args.host}:{args.port}")
    logging.info("pi0 server ready at %s:%d", args.host, args.port)

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore

    _run_type_to_label = {
        "verb": "verb_success", "color": "color_success",
        "shape": "shape_success", "size": "size_success", "spatial": "spatial_success",
    }
    _SPATIAL_EXPS = {"verb_spatial", "color_spatial", "spatial_size", "spatial_object"}
    _FIRST_RUN_MAP = {
        "verb_spatial": "verb", "color_spatial": "color",
        "spatial_size": "spatial", "spatial_object": "spatial",
    }
    _run_type_pairs = {
        "verb_color": ("verb", "color"),   "verb_object": ("verb", "shape"),
        "verb_size":  ("verb", "size"),    "verb_spatial": ("verb", "spatial"),
        "color_object": ("color", "shape"), "size_object": ("size", "shape"),
        "color_size": ("color", "size"),   "color_spatial": ("color", "spatial"),
        "spatial_size": ("spatial", "size"), "spatial_object": ("spatial", "shape"),
    }
    _f1_label_map = {
        "verb_color": "verb_success",    "verb_object": "verb_success",
        "verb_size":  "verb_success",    "verb_spatial": "verb_success",
        "color_object": "color_success", "size_object": "size_success",
        "color_size": "color_success",   "color_spatial": "color_success",
        "spatial_size": "spatial_success", "spatial_object": "spatial_success",
    }
    _f2_label_map = {
        "verb_color": "color_success",   "verb_object": "shape_success",
        "verb_size":  "size_success",    "verb_spatial": "spatial_success",
        "color_object": "shape_success", "size_object": "shape_success",
        "color_size": "size_success",    "color_spatial": "spatial_success",
        "spatial_size": "size_success",  "spatial_object": "shape_success",
    }
    first_type = _run_type_pairs.get(args.experiment, ("", ""))[0]

    first_ok = 0;  first_total = 0
    second_ok = 0; second_total = 0
    if args.batch_skip_to > 0 and results_path.exists():
        import re as _re
        for line in results_path.read_text().splitlines():
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            if len(parts) >= 5:
                rt = parts[3]
                m = _re.match(r"(\d+)/(\d+)", parts[4])
                if m:
                    ok, den = int(m.group(1)), int(m.group(2))
                    if rt == first_type:
                        first_ok += ok; first_total += den
                    else:
                        second_ok += ok; second_total += den

    # Loop over jobs
    for job in jobs:
        idx = int(job["index"])

        job_args = dataclasses.replace(
            args,
            pair_i=int(job["pair_i"]),
            pair_j=int(job["pair_j"]),
            run_type=str(job["run_type"]),
            third_seed=int(job["third_seed"]),
            num_episodes=int(job["num_episodes"]),
            seed=int(job["seed"]),
            experiment_name=str(job["experiment_name"]),
        )

        logging.info("[%d/%d] pair=(%d,%d) run_type=%s  experiment=%s",
                     idx, total, job_args.pair_i, job_args.pair_j,
                     job_args.run_type, job_args.experiment)

        env, instruction = _build_env_and_instruction(job_args)
        logging.info("OOD instruction: %r", instruction)

        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (f"{job_args.experiment_name.strip()}_{timestamp}"
                    if job_args.experiment_name.strip()
                    else f"{job_args.experiment}_{job_args.pair_i}_{job_args.pair_j}_{job_args.run_type}_{timestamp}")
        exp_dir   = pathlib.Path(job_args.experiment_root) / run_name
        video_dir = exp_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        if job_args.experiment in _SPATIAL_EXPS:
            _anchor_i = list(SPATIAL_ANCHORS[SPATIALS[job_args.pair_i]])
            _anchor_j = list(SPATIAL_ANCHORS[SPATIALS[job_args.pair_j]])
            if job_args.run_type == _FIRST_RUN_MAP[job_args.experiment]:
                _reset_opts: dict = {"num_distractors": 1, "obj_xy": _anchor_i, "distractor_xy": [_anchor_j]}
            else:
                _reset_opts = {"num_distractors": 1, "obj_xy": _anchor_j, "distractor_xy": [_anchor_i]}
        elif job_args.experiment in (
            "color_object", "verb_object", "verb_color",
            "verb_size", "size_object", "color_size",
        ):
            _reset_opts = {"num_distractors": 1}
        else:
            _reset_opts = {}

        verb_successes = 0
        factor2_successes = 0

        for ep in tqdm.tqdm(range(job_args.num_episodes), desc=f"[{idx}/{total}]"):
            obs, _ = env.reset(seed=job_args.seed + ep, options=_reset_opts)
            client.reset()

            action_plan: collections.deque = collections.deque()
            base_writer  = imageio.get_writer(str(video_dir / f"ep{ep:03d}.mp4"), fps=30)
            wrist_writer = (
                imageio.get_writer(str(video_dir / f"ep{ep:03d}_wrist.mp4"), fps=30)
                if job_args.save_wrist_video else None
            )
            ep_verb_ok    = False
            ep_factor2_ok = False
            done = False

            try:
                while not done:
                    img_base  = _to_hwc_uint8(obs["sensor_data"]["base_camera"]["rgb"])
                    img_wrist = _to_hwc_uint8(obs["sensor_data"]["hand_camera"]["rgb"])

                    base_writer.append_data(img_base)
                    if wrist_writer is not None:
                        wrist_writer.append_data(img_wrist)

                    if not action_plan:
                        state = _state8(env)
                        chunk = _query_pi0(client, img_base, img_wrist, state, instruction)
                        n = min(job_args.replan_steps, len(chunk))
                        if n < 1:
                            break
                        action_plan.extend(chunk[:n])

                    action = np.asarray(action_plan.popleft(), dtype=np.float32).ravel()[:8]
                    obs, _reward, term, trunc, info = env.step(action)

                    if _bool_info(info, "success_first_axis"):
                        ep_verb_ok = True
                    elif _bool_info(info, "success"):
                        ep_verb_ok = True
                    if _bool_info(info, "success_second_axis"):
                        ep_factor2_ok = True

                    done = bool(term or trunc) or (ep_verb_ok and ep_factor2_ok)

            finally:
                base_writer.close()
                if wrist_writer is not None:
                    wrist_writer.close()

            if ep_verb_ok:
                verb_successes += 1
            if ep_factor2_ok:
                factor2_successes += 1

            logging.info("ep=%d  verb_ok=%s  factor2_ok=%s", ep, ep_verb_ok, ep_factor2_ok)

        env.close()
        import gc
        gc.collect()

        n = max(job_args.num_episodes, 1)
        label1 = _run_type_to_label.get(job_args.run_type, f"{job_args.run_type}_success")
        print(f"Success rate ({label1}): {verb_successes} / {n} ({100.0*verb_successes/n:.1f}%)")
        sys.stdout.flush()

        run_name_base = job_args.experiment_name.strip() or (
            f"{job_args.experiment}_{job_args.pair_i}_{job_args.pair_j}_{job_args.run_type}")
        with open(results_path, "a") as f:
            f.write(f"{idx} {job_args.pair_i} {job_args.pair_j} {job_args.run_type} "
                    f"{verb_successes}/{n} {run_name_base}\n")

        summary_lines = [
            f"experiment={job_args.experiment}",
            f"pair=({job_args.pair_i},{job_args.pair_j})",
            f"run_type={job_args.run_type}",
            f"instruction={instruction!r}",
            f"num_episodes={job_args.num_episodes}",
            f"{label1}={verb_successes}/{n} ({100.0*verb_successes/n:.1f}%)",
            f"server={job_args.host}:{job_args.port}",
        ]
        (exp_dir / "success_rate.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        logging.info("Saved results to %s", str(exp_dir))

        if job_args.run_type == first_type:
            first_ok += verb_successes;  first_total += n
        else:
            second_ok += verb_successes; second_total += n

    def _rate(s, n):
        return f"{100.0*s/n:.1f}" if n > 0 else "0.0"

    f1l = _f1_label_map.get(args.experiment, "first_success")
    f2l = _f2_label_map.get(args.experiment, "second_success")
    with open(results_path, "a") as f:
        f.write(f"\noverall_{f1l}={first_ok}/{first_total} ({_rate(first_ok, first_total)}%)\n")
        f.write(f"overall_{f2l}={second_ok}/{second_total} ({_rate(second_ok, second_total)}%)\n")

    print(f"\noverall_{f1l}={first_ok}/{first_total} ({_rate(first_ok, first_total)}%)")
    print(f"overall_{f2l}={second_ok}/{second_total} ({_rate(second_ok, second_total)}%)")
    print(f"\nSaved summary to {results_path}")
    print(f"Done: {total} runs for {args.experiment}")


def eval_conflict(args: Args) -> None:
    """Single-pair debug mode."""
    import json
    import tempfile
    job = [{
        "index": 1, "pair_i": args.pair_i, "pair_j": args.pair_j,
        "run_type": args.run_type, "seed": args.seed,
        "third_seed": args.third_seed, "num_episodes": args.num_episodes,
        "experiment_name": args.experiment_name or
        f"{args.experiment}_{args.pair_i}_{args.pair_j}_{args.run_type}",
    }]
    jf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(job, jf); jf.close()
    rt = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False); rt.close()
    args.batch_jobs_file = jf.name
    args.batch_results_txt = rt.name
    batch_eval_conflict(args)
    print("\n--- results file ---")
    print(pathlib.Path(rt.name).read_text())


def main() -> None:
    args = tyro.cli(Args)
    if args.batch_jobs_file:
        batch_eval_conflict(args)
    else:
        eval_conflict(args)


if __name__ == "__main__":
    main()
