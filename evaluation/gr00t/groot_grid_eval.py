#!/usr/bin/env python3
"""
GR00T pairwise-grid eval (HARD protocol) - mirrors pi0_grid_eval.py exactly, with
the policy boundary swapped from openpi-websocket to GR00T zmq (GrootClient).
Same HARD color_size / color_spatial cell construction; same env reset
options; same per-cell scoring & results-txt format; same 3-seed protocol.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from paths import setup_paths  # noqa: E402

setup_paths()

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402,F401  (registers VerbObjectColor-v1)
from groot_client import GrootClient  # noqa: E402
# Env-id / color resolver shared with data collection.
from scripts.run_verb_color_shape_motion_planning import get_env_id_and_color  # noqa: E402

COLORS = ("red", "yellow", "blue", "orange", "green", "black")
SIZES = ("small", "large", "smaller", "larger", "smallest", "largest")
SPATIALS = ("left", "right", "middle", "front", "behind")
SHAPES = ("cube", "sphere", "cup", "car", "pyramid", "star")
VERB_POOL = ("lift", "grasp", "push")
VERB_CAP = {"lift": "Lift", "grasp": "Grasp", "push": "Push",
            "pull": "Pull", "rotate": "Rotate", "slide": "Slide"}
SPATIAL_PHRASE = {"left": "on the left", "right": "on the right",
                  "middle": "in the middle", "front": "in front",
                  "behind": "at the back"}
SPATIAL_ANCHOR = {"left": (-0.10, 0.0), "right": (0.10, 0.0),
                  "middle": (0.0, 0.0), "front": (0.0, 0.10),
                  "behind": (0.0, -0.10)}


def _spatial_xy(spatial, rng):
    ax, ay = SPATIAL_ANCHOR[spatial]
    return [ax + rng.uniform(-0.012, 0.012), ay + rng.uniform(-0.012, 0.012)]


def _to_hwc_uint8(x):
    import torch
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.0 + 1e-6:
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8)
    return np.ascontiguousarray(x)


def _state8(env):
    import torch
    q = env.unwrapped.agent.robot.get_qpos()
    if torch.is_tensor(q):
        q = q[0].detach().cpu().numpy()
    q = np.asarray(q, dtype=np.float32).ravel()
    out = np.zeros(8, dtype=np.float32)
    out[: min(8, q.size)] = q[: min(8, q.size)]
    return out


def _success(info):
    import torch
    s = info.get("success", False)
    if torch.is_tensor(s):
        return bool(s.squeeze().item())
    return bool(np.asarray(s).squeeze())


# HARD protocol: same presets as pi0_grid_eval.HARD_SIZE.
HARD_SIZE = {
    "small":    (0.72, [1.20], 1),
    "large":    (1.34, [0.80], 1),
    "smaller":  (0.82, [1.08], 1),
    "larger":   (1.18, [0.92], 1),
    "smallest": (0.78, [1.00, 1.24], 2),
    "largest":  (1.26, [1.00, 0.80], 2),
}

# EASY protocol: one distractor with about 2x size contrast.
EASY_SIZE = {
    "small":    (0.65, [1.35], 1),
    "large":    (1.40, [0.65], 1),
    "smaller":  (0.70, [1.30], 1),
    "larger":   (1.30, [0.70], 1),
    "smallest": (0.65, [1.40], 1),
    "largest":  (1.40, [0.65], 1),
}


def build_color_size_cell(color, size_label, *, distractor_max_arg, task_difficulty, mode="hard"):
    verb, shape = "lift", "cube"
    table = HARD_SIZE if mode == "hard" else EASY_SIZE
    t_scale, d_scales, n_d = table[size_label]
    env_id, color_id, extra = get_env_id_and_color(
        verb, color, shape, distractor_max=max(int(distractor_max_arg), n_d),
        task_difficulty=task_difficulty)
    make_kw = dict(obs_mode="rgb", control_mode="pd_joint_pos",
                   render_mode="rgb_array")
    if env_id == "VerbObjectColor-v1":
        make_kw.update(extra)
        make_kw["object_size_jiggle"] = 0.0
        make_kw["distractor_specs"] = [("cube", int(color_id))] * n_d + \
            [None] * (3 - n_d)
    else:
        make_kw["object_color_id"] = color_id
    reset_opts = {"num_distractors": int(n_d),
                  "target_size_scale": float(t_scale),
                  "distractor_size_scales": [float(x) for x in d_scales]}
    instruction = f"Lift the {size_label} {color} cube."
    return env_id, make_kw, reset_opts, instruction


def build_color_spatial_cell(color, spatial, *, rng, distractor_max_arg, task_difficulty):
    verb = rng.choice(VERB_POOL)
    shape = "cube"
    others = [s for s in SPATIALS if s != spatial]
    d_spatial = rng.choice(others)
    env_id, color_id, extra = get_env_id_and_color(
        verb, color, shape, distractor_max=1, task_difficulty=task_difficulty)
    make_kw = dict(obs_mode="rgb", control_mode="pd_joint_pos",
                   render_mode="rgb_array")
    if env_id == "VerbObjectColor-v1":
        make_kw.update(extra)
        make_kw["object_size_jiggle"] = 0.0
        make_kw["distractor_specs"] = [("cube", int(color_id)), None, None]
    else:
        make_kw["object_color_id"] = color_id
    reset_opts = {"num_distractors": 1, "target_size_scale": 1.0,
                  "obj_xy": _spatial_xy(spatial, rng),
                  "distractor_xy": [_spatial_xy(d_spatial, rng)],
                  "distractor_size_scales": [1.0]}
    instruction = f"{VERB_CAP[verb]} the {color} cube {SPATIAL_PHRASE[spatial]}."
    return env_id, make_kw, reset_opts, instruction


def build_verb_spatial_cell(verb, spatial, *, rng, distractor_max_arg, task_difficulty):
    """HARD: 2 same-color same-shape(cube) cubes at DIFFERENT spatial anchors.
    Color sampled per-cell (third factor); instruction uses verb + spatial."""
    color = rng.choice(COLORS)
    shape = "cube"
    others = [s for s in SPATIALS if s != spatial]
    d_spatial = rng.choice(others)
    env_id, color_id, extra = get_env_id_and_color(
        verb, color, shape, distractor_max=1, task_difficulty=task_difficulty)
    make_kw = dict(obs_mode="rgb", control_mode="pd_joint_pos",
                   render_mode="rgb_array")
    if env_id == "VerbObjectColor-v1":
        make_kw.update(extra)
        make_kw["object_size_jiggle"] = 0.0
        make_kw["distractor_specs"] = [("cube", int(color_id)), None, None]
    else:
        make_kw["object_color_id"] = color_id
    reset_opts = {"num_distractors": 1, "target_size_scale": 1.0,
                  "obj_xy": _spatial_xy(spatial, rng),
                  "distractor_xy": [_spatial_xy(d_spatial, rng)],
                  "distractor_size_scales": [1.0]}
    instruction = f"{VERB_CAP[verb]} the cube {SPATIAL_PHRASE[spatial]}."
    return env_id, make_kw, reset_opts, instruction


def build_spatial_object_cell(spatial, shape, *, rng, distractor_max_arg, task_difficulty):
    """HARD: target shape at spatial; distractor = DIFFERENT shape, same color,
    at a DIFFERENT spatial.  Instruction uses shape + spatial phrase."""
    color = rng.choice(COLORS)
    verb = rng.choice(VERB_POOL)
    d_shape = rng.choice([s for s in SHAPES if s != shape])
    d_spatial = rng.choice([s for s in SPATIALS if s != spatial])
    env_id, color_id, extra = get_env_id_and_color(
        verb, color, shape, distractor_max=1, task_difficulty=task_difficulty)
    make_kw = dict(obs_mode="rgb", control_mode="pd_joint_pos",
                   render_mode="rgb_array")
    if env_id == "VerbObjectColor-v1":
        make_kw.update(extra)
        make_kw["object_size_jiggle"] = 0.0
        make_kw["distractor_specs"] = [(d_shape, int(color_id)), None, None]
    else:
        make_kw["object_color_id"] = color_id
    reset_opts = {"num_distractors": 1, "target_size_scale": 1.0,
                  "obj_xy": _spatial_xy(spatial, rng),
                  "distractor_xy": [_spatial_xy(d_spatial, rng)],
                  "distractor_size_scales": [1.0]}
    instruction = f"{VERB_CAP[verb]} the {shape} {SPATIAL_PHRASE[spatial]}."
    return env_id, make_kw, reset_opts, instruction


def build_verb_size_cell(verb, size_label, *, rng, distractor_max_arg, task_difficulty):
    """HARD: target cube of size; distractor(s) = same-color cube differing
    ONLY in size.  Instruction uses verb + size word."""
    color = rng.choice(COLORS)
    shape = "cube"
    t_scale, d_scales, n_d = HARD_SIZE[size_label]
    env_id, color_id, extra = get_env_id_and_color(
        verb, color, shape, distractor_max=max(int(distractor_max_arg), n_d),
        task_difficulty=task_difficulty)
    make_kw = dict(obs_mode="rgb", control_mode="pd_joint_pos",
                   render_mode="rgb_array")
    if env_id == "VerbObjectColor-v1":
        make_kw.update(extra)
        make_kw["object_size_jiggle"] = 0.0
        make_kw["distractor_specs"] = [("cube", int(color_id))] * n_d + \
            [None] * (3 - n_d)
    else:
        make_kw["object_color_id"] = color_id
    reset_opts = {"num_distractors": int(n_d),
                  "target_size_scale": float(t_scale),
                  "distractor_size_scales": [float(x) for x in d_scales]}
    instruction = f"{VERB_CAP[verb]} the {size_label} cube."
    return env_id, make_kw, reset_opts, instruction


def build_spatial_size_cell(spatial, size_label, *, rng, distractor_max_arg, task_difficulty):
    """HARD: 2 same-color same-shape(cube) cubes at DIFFERENT spatial AND
    DIFFERENT size.  Instruction uses both spatial phrase + size word."""
    color = rng.choice(COLORS)
    verb = rng.choice(VERB_POOL)
    shape = "cube"
    other_spatials = [s for s in SPATIALS if s != spatial]
    d_spatial = rng.choice(other_spatials)
    t_scale, _, _ = HARD_SIZE[size_label]
    # distractor: opposite-extreme scale
    if size_label in ("small", "smaller", "smallest"):
        d_scale = 1.25
    else:
        d_scale = 0.75
    env_id, color_id, extra = get_env_id_and_color(
        verb, color, shape, distractor_max=1, task_difficulty=task_difficulty)
    make_kw = dict(obs_mode="rgb", control_mode="pd_joint_pos",
                   render_mode="rgb_array")
    if env_id == "VerbObjectColor-v1":
        make_kw.update(extra)
        make_kw["object_size_jiggle"] = 0.0
        make_kw["distractor_specs"] = [("cube", int(color_id)), None, None]
    else:
        make_kw["object_color_id"] = color_id
    reset_opts = {"num_distractors": 1,
                  "target_size_scale": float(t_scale),
                  "obj_xy": _spatial_xy(spatial, rng),
                  "distractor_xy": [_spatial_xy(d_spatial, rng)],
                  "distractor_size_scales": [float(d_scale)]}
    instruction = f"{VERB_CAP[verb]} the {size_label} cube {SPATIAL_PHRASE[spatial]}."
    return env_id, make_kw, reset_opts, instruction


def _groot_action_chunk(client: GrootClient, img_base, img_wrist, state8, instruction):
    """Build GR00T obs, call get_action, return (Th, 8) float32 chunk."""
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
    arm = np.asarray(action["arm"], dtype=np.float32)        # (1, Th, 7) or (Th, 7)
    grip = np.asarray(action["gripper"], dtype=np.float32)   # (1, Th, 1) or (Th, 1)
    if arm.ndim == 3:
        arm = arm[0]
    if grip.ndim == 3:
        grip = grip[0]
    if grip.ndim == 1:
        grip = grip[:, None]
    return np.concatenate([arm, grip], axis=-1).astype(np.float32)  # (Th, 8)


def run_cell(client, env_id, make_kw, reset_opts, instruction, *,
             seed, sim_backend, max_steps, replan):
    mk = dict(make_kw)
    mk["sim_backend"] = sim_backend
    mk["render_backend"] = sim_backend
    env = gym.make(env_id, **mk)
    obs, _ = env.reset(seed=seed, options=reset_opts)
    plan = []
    done = ok = False
    steps = 0
    try:
        while not done and steps < max_steps:
            b = _to_hwc_uint8(obs["sensor_data"]["base_camera"]["rgb"])
            h = _to_hwc_uint8(obs["sensor_data"]["hand_camera"]["rgb"])
            if not plan:
                chunk = _groot_action_chunk(client, b, h, _state8(env), instruction)
                n = min(replan, len(chunk))
                if n < 1:
                    break
                plan = list(chunk[:n])
            act = np.asarray(plan.pop(0), dtype=np.float32).ravel()[:8]
            obs, _r, term, trunc, info = env.step(act)
            steps += 1
            if _success(info):
                ok = True
            done = bool(term or trunc) or ok
    finally:
        env.close()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    choices=["color_size", "color_spatial", "verb_spatial",
                             "spatial_size", "spatial_object", "verb_size"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-txt", required=True)
    ap.add_argument("--sim-backend", default="gpu")
    ap.add_argument("--task-difficulty", type=float, default=1.5)
    ap.add_argument("--max-episode-steps", type=int, default=300)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--distractor-max", type=int, default=2)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--target-episodes", type=int, default=0)
    ap.add_argument("--color-size-mode", choices=["hard", "easy"], default="hard")
    a = ap.parse_args()

    if a.experiment == "color_size":
        f1vals, f1name = COLORS, "color"
        f2vals, f2name = SIZES, "size"
        meta = f"fixed verb=lift shape=cube  {a.color_size_mode.upper()} same-color cube distractor"
    elif a.experiment == "color_spatial":
        f1vals, f1name = COLORS, "color"
        f2vals, f2name = SPATIALS, "spatial"
        meta = "fixed shape=cube; verb~{lift,grasp,push}; HARD same-color cube distractor"
    elif a.experiment == "verb_spatial":
        f1vals, f1name = VERB_POOL, "verb"
        f2vals, f2name = SPATIALS, "spatial"
        meta = "fixed shape=cube; color~{COLORS}; HARD same-color cube distractor at different spatial"
    elif a.experiment == "spatial_size":
        f1vals, f1name = SPATIALS, "spatial"
        f2vals, f2name = SIZES, "size"
        meta = "fixed shape=cube; color&verb~random; HARD same-color cube distractor at different spatial & different size"
    elif a.experiment == "spatial_object":
        f1vals, f1name = SPATIALS, "spatial"
        f2vals, f2name = SHAPES, "shape"
        meta = "color&verb~random; HARD same-color different-shape distractor at different spatial"
    else:  # verb_size
        f1vals, f1name = VERB_POOL, "verb"
        f2vals, f2name = SIZES, "size"
        meta = "fixed shape=cube; color~random; HARD same-color cube distractor differing only in size"
    cells = [(c, x) for c in f1vals for x in f2vals]
    if a.max_cells > 0:
        cells = cells[: a.max_cells]

    client = GrootClient(a.host, a.port)
    # Fail fast if the policy server is not running.
    if not client.ping():
        raise SystemExit(f"GR00T server not reachable at {a.host}:{a.port}")
    rt = pathlib.Path(a.results_txt)
    rt.parent.mkdir(parents=True, exist_ok=True)
    with rt.open("w") as f:
        f.write(f"# gr00t {a.experiment} grid  {meta}  "
                f"seed={a.seed} task_difficulty={a.task_difficulty} "
                f"sim={a.sim_backend} cells={len(cells)}\n")
        f.write(f"idx {f1name} {f2name} success prompt\n")

    import math as _m
    ncells = len(cells)
    reps = max(1, _m.ceil(a.target_episodes / ncells)) if a.target_episodes > 0 else 1
    print(f"cells={ncells} reps/cell={reps} -> {ncells*reps} episodes/seed", flush=True)

    succ = tot = 0
    for idx, (f1, f2) in enumerate(cells, 1):
        for r in range(reps):
            rng = random.Random(a.seed * 100003 + idx * 131 + r)
            try:
                if a.experiment == "color_size":
                    env_id, mk, ro, instr = build_color_size_cell(
                        f1, f2, distractor_max_arg=a.distractor_max,
                        task_difficulty=a.task_difficulty, mode=a.color_size_mode)
                elif a.experiment == "color_spatial":
                    env_id, mk, ro, instr = build_color_spatial_cell(
                        f1, f2, rng=rng, distractor_max_arg=a.distractor_max,
                        task_difficulty=a.task_difficulty)
                elif a.experiment == "verb_spatial":
                    env_id, mk, ro, instr = build_verb_spatial_cell(
                        f1, f2, rng=rng, distractor_max_arg=a.distractor_max,
                        task_difficulty=a.task_difficulty)
                elif a.experiment == "spatial_size":
                    env_id, mk, ro, instr = build_spatial_size_cell(
                        f1, f2, rng=rng, distractor_max_arg=a.distractor_max,
                        task_difficulty=a.task_difficulty)
                elif a.experiment == "spatial_object":
                    env_id, mk, ro, instr = build_spatial_object_cell(
                        f1, f2, rng=rng, distractor_max_arg=a.distractor_max,
                        task_difficulty=a.task_difficulty)
                else:  # verb_size
                    env_id, mk, ro, instr = build_verb_size_cell(
                        f1, f2, rng=rng, distractor_max_arg=a.distractor_max,
                        task_difficulty=a.task_difficulty)
                ok = run_cell(client, env_id, mk, ro, instr,
                              seed=a.seed + idx * 1000 + r,
                              sim_backend=a.sim_backend,
                              max_steps=a.max_episode_steps, replan=a.replan_steps)
            except Exception as e:  # noqa: BLE001
                print(f"[{idx}/{ncells} r{r}] FAIL {f1},{f2}: {e}", flush=True)
                ok, instr = False, f"ERROR:{e}"
            succ += int(ok)
            tot += 1
            with rt.open("a") as f:
                f.write(f'{idx} {f1} {f2} rep{r} {int(ok)} "{instr}"\n')
        print(f"[{idx}/{ncells}] {f1} {f2} {reps}reps -> cum {succ}/{tot}",
              flush=True)

    rate = 100.0 * succ / tot if tot else 0.0
    with rt.open("a") as f:
        f.write(f"\noverall_success={succ}/{tot} ({rate:.1f}%)\n")
    print(f"\nDone {a.experiment}: overall_success={succ}/{tot} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
