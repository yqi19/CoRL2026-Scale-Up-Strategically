#!/usr/bin/env python3
"""
pi0 / pi0.5 pairwise-grid eval (HARD protocol) against an OpenPI websocket
policy server. Same cell construction, reset options, scoring and results
format as ../gr00t/groot_grid_eval.py.

Implemented grids:
  color_size     color(6) x size(6); fixed verb=lift, shape=cube; every cell has
                 same-color cube distractor(s) that differ only in size.
  color_spatial  color(6) x spatial(5); shape=cube, verb ~ {lift, grasp, push};
                 one same-color cube distractor at a different spatial anchor.

Envs are built with ``get_env_id_and_color`` from the ManiSkill fork, so
legacy (verb, shape, color) combinations route to their original env ids.
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
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as _wcp  # noqa: E402
# Env-id / color resolver shared with data collection.
from scripts.run_verb_color_shape_motion_planning import get_env_id_and_color  # noqa: E402

COLORS = ("red", "yellow", "blue", "orange", "green", "black")
SIZES = ("small", "large", "smaller", "larger", "smallest", "largest")
SPATIALS = ("left", "right", "middle", "front", "behind")
VERB_POOL = ("lift", "grasp", "push")  # training_vocab THIRD_VERBS (first 3)
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


# Size presets, copied from collect_pairwise_attribute.py
def _sample_small_large(size_label):
    if size_label == "small":
        return 0.72, None, 0
    if size_label == "large":
        return 1.34, None, 0
    raise ValueError(size_label)


def _size_controls(size_label):
    if size_label == "smaller":
        return 0.82, [1.08], 1
    if size_label == "larger":
        return 1.18, [0.92], 1
    if size_label == "smallest":
        return 0.78, [1.00, 1.24], 2
    if size_label == "largest":
        return 1.26, [1.00, 0.80], 2
    raise ValueError(size_label)


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


# HARD color_size protocol: every cell has same-color cube distractors that differ
# only in size, so the policy must use the size word to find the target.
# Entries: (target_scale, [distractor_scales], n_distractors).
HARD_SIZE = {
    "small":    (0.72, [1.20], 1),
    "large":    (1.34, [0.80], 1),
    "smaller":  (0.82, [1.08], 1),
    "larger":   (1.18, [0.92], 1),
    "smallest": (0.78, [1.00, 1.24], 2),
    "largest":  (1.26, [1.00, 0.80], 2),
}


def build_color_size_cell(color, size_label, *, distractor_max_arg, task_difficulty):
    """HARD: ≥1 same-color cube distractor, size-only difference."""
    verb, shape = "lift", "cube"
    t_scale, d_scales, n_d = HARD_SIZE[size_label]
    env_id, color_id, extra = get_env_id_and_color(
        verb, color, shape, distractor_max=max(int(distractor_max_arg), n_d),
        task_difficulty=task_difficulty)
    make_kw = dict(obs_mode="rgb", control_mode="pd_joint_pos",
                   render_mode="rgb_array")
    if env_id == "VerbObjectColor-v1":
        make_kw.update(extra)
        make_kw["object_size_jiggle"] = 0.0
        # distractor_specs is an env CONSTRUCTOR arg: force cube + target color
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
    """HARD: a same-color cube distractor at a different spatial anchor, so the
    policy must use the spatial phrase to find the target."""
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


def run_cell(client, env_id, make_kw, reset_opts, instruction, *,
             seed, sim_backend, max_steps, replan, resize=224):
    mk = dict(make_kw)
    mk["sim_backend"] = sim_backend
    mk["render_backend"] = sim_backend
    env = gym.make(env_id, **mk)
    obs, _ = env.reset(seed=seed, options=reset_opts)
    plan = []
    done = ok = False
    try:
        while not done:
            b = _to_hwc_uint8(obs["sensor_data"]["base_camera"]["rgb"])
            h = _to_hwc_uint8(obs["sensor_data"]["hand_camera"]["rgb"])
            if not plan:
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(b, resize, resize))
                wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(h, resize, resize))
                chunk = client.infer({
                    "observation/image": img,
                    "observation/wrist_image": wri,
                    "observation/state": _state8(env).astype(np.float32),
                    "prompt": instruction,
                })["actions"]
                chunk = np.asarray(chunk, dtype=np.float32)
                n = min(replan, len(chunk))
                if n < 1:
                    break
                plan = list(chunk[:n])
            act = np.asarray(plan.pop(0), dtype=np.float32).ravel()[:8]
            obs, _r, term, trunc, info = env.step(act)
            if _success(info):
                ok = True
            done = bool(term or trunc) or ok
    finally:
        env.close()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    choices=["color_size", "color_spatial"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-txt", required=True)
    ap.add_argument("--sim-backend", default="cpu")
    ap.add_argument("--task-difficulty", type=float, default=1.3)
    ap.add_argument("--max-episode-steps", type=int, default=150)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--distractor-max", type=int, default=2)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--target-episodes", type=int, default=0,
                    help=">0: run ceil(target/ncells) reps per cell (~target total)")
    a = ap.parse_args()

    if a.experiment == "color_size":
        f2vals, f2name = SIZES, "size"
        meta = "fixed verb=lift shape=cube"
    else:  # color_spatial
        f2vals, f2name = SPATIALS, "spatial"
        meta = "fixed shape=cube; verb~{lift,grasp,push}; no distractor"
    cells = [(c, x) for c in COLORS for x in f2vals]  # full grid
    if a.max_cells > 0:
        cells = cells[: a.max_cells]

    client = _wcp.WebsocketClientPolicy(a.host, a.port)
    rt = pathlib.Path(a.results_txt)
    rt.parent.mkdir(parents=True, exist_ok=True)
    with rt.open("w") as f:
        f.write(f"# pi0 {a.experiment} grid  {meta}  "
                f"seed={a.seed} task_difficulty={a.task_difficulty} "
                f"sim={a.sim_backend} cells={len(cells)}\n")
        f.write(f"idx color {f2name} success prompt\n")

    import math as _m
    ncells = len(cells)
    reps = max(1, _m.ceil(a.target_episodes / ncells)) if a.target_episodes > 0 else 1
    print(f"cells={ncells} reps/cell={reps} -> {ncells*reps} episodes/seed", flush=True)

    succ = tot = 0
    for idx, (color, f2) in enumerate(cells, 1):
        for r in range(reps):
            rng = random.Random(a.seed * 100003 + idx * 131 + r)
            try:
                if a.experiment == "color_size":
                    env_id, mk, ro, instr = build_color_size_cell(
                        color, f2, distractor_max_arg=a.distractor_max,
                        task_difficulty=a.task_difficulty)
                else:
                    env_id, mk, ro, instr = build_color_spatial_cell(
                        color, f2, rng=rng, distractor_max_arg=a.distractor_max,
                        task_difficulty=a.task_difficulty)
                ok = run_cell(client, env_id, mk, ro, instr,
                              seed=a.seed + idx * 1000 + r,
                              sim_backend=a.sim_backend,
                              max_steps=a.max_episode_steps, replan=a.replan_steps)
            except Exception as e:  # noqa: BLE001
                print(f"[{idx}/{ncells} r{r}] FAIL {color},{f2}: {e}", flush=True)
                ok, instr = False, f"ERROR:{e}"
            succ += int(ok)
            tot += 1
            with rt.open("a") as f:
                f.write(f'{idx} {color} {f2} rep{r} {int(ok)} "{instr}"\n')
        print(f"[{idx}/{ncells}] {color} {f2} {reps}reps -> cum {succ}/{tot}",
              flush=True)

    rate = 100.0 * succ / tot if tot else 0.0
    with rt.open("a") as f:
        f.write(f"\noverall_success={succ}/{tot} ({rate:.1f}%)\n")
    print(f"\nDone {a.experiment}: overall_success={succ}/{tot} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
