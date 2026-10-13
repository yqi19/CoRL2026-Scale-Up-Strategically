#!/usr/bin/env python3
"""
GR00T N1.7 full-factor eval - aligned 1:1 with the pi0.5 protocol in
the pi0.5 full-factor client (OpenPI fork, examples/maniskill_full_factor/main.py).

The environment, cell vocabulary, instruction format, distractor / size /
spatial logic, success criterion and the `Success rate: X / Y (Z%)` stdout
line match the pi0.5 client, so the numbers are directly comparable.
Only the policy boundary differs is the policy boundary: instead of
the openpi websocket client we drive a fine-tuned GR00T N1.7 checkpoint
served over zmq by gr00t.eval.run_gr00t_server (same wire format the conflict
harness uses - see groot_conflict_eval._query_groot).
"""
from __future__ import annotations

import collections
import dataclasses
import logging
import pathlib
import random

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tqdm
import tyro

from groot_client import GrootClient

# Vocabularies (identical to the pi0.5 client)
TRAINING_VERBS    = ("lift", "grasp", "push", "pull", "rotate", "slide")
TRAINING_COLORS   = ("red", "yellow", "blue", "orange", "green", "black")
TRAINING_SHAPES   = ("cube", "sphere", "cup", "car", "pyramid", "star")
TRAINING_SPATIALS = ("left", "right", "middle", "front", "behind")
TRAINING_SIZES    = ("small", "large", "smaller", "larger")

COLOR_TO_ID = {c: i for i, c in enumerate(TRAINING_COLORS)}

VERB_TO_EN = {
    "lift": "Lift", "grasp": "Grasp", "push": "Push",
    "pull": "Pull", "rotate": "Rotate", "slide": "Slide",
}
SPATIAL_TO_PHRASE = {
    "left": "on the left", "right": "on the right", "middle": "in the middle",
    "front": "in front", "behind": "at the back",
}
SPATIAL_XY_ANCHOR = {
    "left":   (-0.10,  0.00),
    "right":  ( 0.10,  0.00),
    "middle": ( 0.00,  0.00),
    "front":  ( 0.00,  0.10),
    "behind": ( 0.00, -0.10),
}
SIZE_CONFIG = {
    "small":   dict(target_size_scale=0.72, distractor_size_scales=None),
    "large":   dict(target_size_scale=1.34, distractor_size_scales=None),
    "smaller": dict(target_size_scale=0.82, distractor_size_scales=[1.08]),
    "larger":  dict(target_size_scale=1.18, distractor_size_scales=[0.92]),
}


def make_instruction(verb: str, size: str, color: str, shape: str, spatial: str) -> str:
    return f"{VERB_TO_EN[verb]} the {size} {color} {shape} {SPATIAL_TO_PHRASE[spatial]}."


@dataclasses.dataclass
class Args:
    # GR00T zmq policy server
    host: str = "127.0.0.1"
    port: int = 5555
    replan_steps: int = 5

    # Task specification (5 factors)
    verb: str = "lift"
    color: str = "red"
    shape: str = "cube"
    spatial: str = "left"
    size: str = "small"
    prompt: str = ""
    """Override language instruction; if empty, auto-built from the 5 factors."""

    no_distractor_prob: float = 0.70
    """Probability per episode of forcing num_distractors=0 (pi0.5: 0.70)."""

    # ManiSkill
    num_episodes: int = 50
    max_episode_steps: int = 500
    sim_backend: str = "cpu"
    render_backend: str = "cpu"
    obs_mode: str = "rgb"
    render_mode: str | None = None
    seed: int = 0

    # Output
    video_out_path: str = "data/maniskill_full_factor/videos"
    save_wrist_video: bool = True


# Helpers (identical to the pi0.5 client)
def _to_numpy_hwc(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] in (1, 3) and x.shape[-1] != 3 and x.ndim == 3:
        x = np.transpose(x, (1, 2, 0))
    if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.0:
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8)
    return np.ascontiguousarray(x)


def _state8(env: gym.Env) -> np.ndarray:
    qpos = env.unwrapped.agent.robot.get_qpos()
    if torch.is_tensor(qpos):
        qpos = qpos[0].detach().cpu().numpy()
    qpos = np.asarray(qpos, dtype=np.float32).ravel()
    if qpos.size >= 8:
        return qpos[:8].copy()
    out = np.zeros(8, dtype=np.float32)
    out[: qpos.size] = qpos
    return out


def _success(info: dict) -> bool:
    if "success" not in info:
        return False
    s = info["success"]
    if torch.is_tensor(s):
        return bool(s.squeeze().item())
    return bool(np.asarray(s).squeeze())


def _spatial_xy(spatial: str, rng: random.Random) -> list[float]:
    ax, ay = SPATIAL_XY_ANCHOR[spatial]
    return [ax + rng.uniform(-0.012, 0.012), ay + rng.uniform(-0.012, 0.012)]


# GR00T policy boundary (same as groot_conflict_eval._query_groot)
def _query_groot(client: GrootClient, img_base: np.ndarray, img_wrist: np.ndarray,
                  state8: np.ndarray, instruction: str) -> np.ndarray:
    """Build the nested GR00T observation, query the server, return an
    (action_horizon, 8) float32 chunk = [7 joint-pos targets, 1 gripper]."""
    obs = {
        "video": {
            "image":       img_base[None, None, ...],
            "wrist_image": img_wrist[None, None, ...],
        },
        "state": {
            "arm":     state8[:7][None, None, :].astype(np.float32),
            "gripper": state8[7:8][None, None, :].astype(np.float32),
        },
        "language": {
            "annotation.human.task_description": [[instruction]],
        },
    }
    action, _info = client.get_action(obs)
    arm = np.asarray(action["arm"], dtype=np.float32)       # (1, Th, 7)
    grip = np.asarray(action["gripper"], dtype=np.float32)  # (1, Th, 1)
    return np.concatenate([arm[0], grip[0]], axis=-1)       # (Th, 8)


# Main eval loop (env / sampling logic identical to the pi0.5 client)
def eval_full_factor(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    verb    = args.verb.lower().strip()
    color   = args.color.lower().strip()
    shape   = args.shape.lower().strip()
    spatial = args.spatial.lower().strip()
    size    = args.size.lower().strip()

    for val, vocab, name in [
        (verb,    TRAINING_VERBS,    "verb"),
        (color,   TRAINING_COLORS,   "color"),
        (shape,   TRAINING_SHAPES,   "shape"),
        (spatial, TRAINING_SPATIALS, "spatial"),
        (size,    TRAINING_SIZES,    "size"),
    ]:
        if val not in vocab:
            raise ValueError(f"{name}={val!r} not in {vocab}")

    prompt = args.prompt.strip() or make_instruction(verb, size, color, shape, spatial)
    logging.info("prompt=%r", prompt)

    size_cfg = SIZE_CONFIG[size]
    object_color_id = COLOR_TO_ID[color]

    has_comparison = size_cfg["distractor_size_scales"] is not None
    distractor_max = 1 if has_comparison else 0

    make_kw: dict = dict(
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
        verb=verb,
        object_shape=shape,
        object_color_id=object_color_id,
        distractor_max=distractor_max,
        object_size_jiggle=0.0,
    )
    if args.render_mode is not None:
        make_kw["render_mode"] = args.render_mode

    env = gym.make("VerbObjectColor-v1", **make_kw)

    video_out_path = pathlib.Path(args.video_out_path)
    video_out_path.mkdir(parents=True, exist_ok=True)

    client = GrootClient(args.host, args.port)
    rng = random.Random(args.seed)

    successes = 0
    for ep in tqdm.tqdm(range(args.num_episodes)):
        no_distractor = rng.random() < args.no_distractor_prob
        reset_options: dict = {
            "obj_xy": _spatial_xy(spatial, rng),
            "target_size_scale": size_cfg["target_size_scale"],
        }
        if size_cfg["distractor_size_scales"] is not None:
            reset_options["distractor_size_scales"] = size_cfg["distractor_size_scales"]
        if no_distractor:
            reset_options["num_distractors"] = 0

        obs, _ = env.reset(seed=args.seed + ep, options=reset_options)
        client.reset()
        plan: collections.deque = collections.deque()

        base_path  = video_out_path / f"ep{ep:03d}.mp4"
        wrist_path = video_out_path / f"ep{ep:03d}_wrist.mp4"
        writer = imageio.get_writer(base_path, fps=30)
        wrist_writer = imageio.get_writer(wrist_path, fps=30) if args.save_wrist_video else None

        done = False
        ep_success = False
        try:
            while not done:
                rgb_b = _to_numpy_hwc(obs["sensor_data"]["base_camera"]["rgb"])
                rgb_h = _to_numpy_hwc(obs["sensor_data"]["hand_camera"]["rgb"])
                writer.append_data(rgb_b)
                if wrist_writer is not None:
                    wrist_writer.append_data(rgb_h)

                if not plan:
                    st = _state8(env)
                    chunk = _query_groot(client, rgb_b, rgb_h, st, prompt)
                    n = min(args.replan_steps, len(chunk))
                    if n < 1:
                        logging.warning("Empty action chunk from policy")
                        break
                    plan.extend(chunk[:n])

                action = np.asarray(plan.popleft(), dtype=np.float32).ravel()[:8]
                obs, _reward, term, trunc, info = env.step(action)
                if _success(info):
                    ep_success = True
                done = bool(term or trunc) or ep_success
        finally:
            try:
                writer.close()
            finally:
                if wrist_writer is not None:
                    wrist_writer.close()

        if ep_success:
            successes += 1
        logging.info("Episode %d success=%s no_distractor=%s", ep, ep_success, no_distractor)

    env.close()
    rate = successes / max(args.num_episodes, 1)
    logging.info("Success rate: %d / %d (%.1f%%)", successes, args.num_episodes, 100.0 * rate)


def main() -> None:
    eval_full_factor(tyro.cli(Args))


if __name__ == "__main__":
    main()
