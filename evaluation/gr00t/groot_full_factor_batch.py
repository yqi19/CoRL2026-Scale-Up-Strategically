#!/usr/bin/env python3
"""
GR00T all-factor (full-factor) eval, single-process batch.

Functionally identical to running groot_full_factor_main.py once per sampled
cell, but all cells run in one Python process: torch/mani_skill import,
GR00T zmq connection and GPU-sim context are initialised once instead of per cell.
All per-cell logic (cell sampling, RNG sequence, env build, success, video,
result-line / header / overall_success format) is reused from
groot_full_factor_main.py, so results match the per-cell script.

Usage:
  groot_full_factor_batch.py --host H --port P --results-txt PATH --video-root DIR \
      [--sample-n 200] [--sample-seed 42] [--seed-base 40] [--total-episodes 200] \
      [--max-episode-steps 500] [--no-distractor-prob 0.70] [--replan-steps 5] \
      [--sim-backend gpu] [--render-backend gpu]
"""
from __future__ import annotations

import argparse
import itertools
import math
import pathlib
import random
import sys

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import numpy as np

# Reuses the per-cell logic of groot_full_factor_main.py.
from groot_full_factor_main import (
    SIZE_CONFIG,
    COLOR_TO_ID,
    _query_groot,
    _spatial_xy,
    _state8,
    _success,
    _to_numpy_hwc,
    make_instruction,
)
from groot_client import GrootClient

# Same ordering as the pi0.5 full-factor cell sampler.
VERBS    = ["lift", "grasp", "push", "pull", "rotate", "slide"]
COLORS   = ["red", "yellow", "blue", "orange", "green", "black"]
SHAPES   = ["cube", "sphere", "cup", "car", "pyramid", "star"]
SPATIALS = ["left", "right", "middle", "front", "behind"]
SIZES    = ["small", "large", "smaller", "larger"]


def sample_cells(sample_n: int, sample_seed: int):
    all_tasks = list(itertools.product(VERBS, COLORS, SHAPES, SPATIALS, SIZES))
    if sample_n > 0:
        rng = random.Random(sample_seed)
        rng.shuffle(all_tasks)
        all_tasks = all_tasks[:sample_n]
    return all_tasks


def run_cell(client, verb, color, shape, spatial, size, cell_seed, n_eps,
             no_distractor_prob, max_steps, replan_steps, sim_backend,
             render_backend, video_dir, save_wrist=True):
    """Mirror groot_full_factor_main.eval_full_factor for ONE cell, 1 process."""
    prompt = make_instruction(verb, size, color, shape, spatial)
    size_cfg = SIZE_CONFIG[size]
    object_color_id = COLOR_TO_ID[color]
    has_comparison = size_cfg["distractor_size_scales"] is not None
    distractor_max = 1 if has_comparison else 0

    make_kw = dict(
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend=sim_backend,
        render_backend=render_backend,
        max_episode_steps=max_steps,
        verb=verb,
        object_shape=shape,
        object_color_id=object_color_id,
        distractor_max=distractor_max,
        object_size_jiggle=0.0,
    )
    env = gym.make("VerbObjectColor-v1", **make_kw)
    video_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cell_seed)  # same as per-cell main.py (rng=Random(args.seed))
    successes = 0
    try:
        for ep in range(n_eps):
            no_distractor = rng.random() < no_distractor_prob
            reset_options = {
                "obj_xy": _spatial_xy(spatial, rng),
                "target_size_scale": size_cfg["target_size_scale"],
            }
            if size_cfg["distractor_size_scales"] is not None:
                reset_options["distractor_size_scales"] = size_cfg["distractor_size_scales"]
            if no_distractor:
                reset_options["num_distractors"] = 0

            obs, _ = env.reset(seed=cell_seed + ep, options=reset_options)
            client.reset()
            plan = []
            base_w = imageio.get_writer(video_dir / f"ep{ep:03d}.mp4", fps=30)
            wrist_w = imageio.get_writer(video_dir / f"ep{ep:03d}_wrist.mp4", fps=30) if save_wrist else None
            done = False
            ep_ok = False
            try:
                while not done:
                    rgb_b = _to_numpy_hwc(obs["sensor_data"]["base_camera"]["rgb"])
                    rgb_h = _to_numpy_hwc(obs["sensor_data"]["hand_camera"]["rgb"])
                    base_w.append_data(rgb_b)
                    if wrist_w is not None:
                        wrist_w.append_data(rgb_h)
                    if not plan:
                        chunk = _query_groot(client, rgb_b, rgb_h, _state8(env), prompt)
                        nn = min(replan_steps, len(chunk))
                        if nn < 1:
                            break
                        plan = list(chunk[:nn])
                    action = np.asarray(plan.pop(0), dtype=np.float32).ravel()[:8]
                    obs, _r, term, trunc, info = env.step(action)
                    if _success(info):
                        ep_ok = True
                    done = bool(term or trunc) or ep_ok
            finally:
                base_w.close()
                if wrist_w is not None:
                    wrist_w.close()
            if ep_ok:
                successes += 1
    finally:
        env.close()
    return successes, n_eps, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--results-txt", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--seed-base", type=int, default=40)
    ap.add_argument("--total-episodes", type=int, default=200)
    ap.add_argument("--max-episode-steps", type=int, default=500)
    ap.add_argument("--no-distractor-prob", type=float, default=0.70)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--sim-backend", default="gpu")
    ap.add_argument("--render-backend", default="gpu")
    a = ap.parse_args()

    cells = sample_cells(a.sample_n, a.sample_seed)
    total_cells = len(cells)
    n_eps = max(1, math.ceil(a.total_episodes / total_cells))

    rt = pathlib.Path(a.results_txt)
    rt.parent.mkdir(parents=True, exist_ok=True)
    with rt.open("w") as f:
        f.write("# Full-factor inference (GR00T N1.7) [single-process batch]\n")
        f.write(f"sample_n={a.sample_n}  sample_seed={a.sample_seed}  total_cells={total_cells}\n")
        f.write(f"total_episodes_target={a.total_episodes}  num_episodes_per_cell={n_eps}\n")
        f.write(f"total_episodes_actual={total_cells * n_eps}\n")
        f.write(f"host={a.host}  port={a.port}\n")
        f.write(f"sim_backend={a.sim_backend}  render_backend={a.render_backend}\n")
        f.write(f"max_episode_steps={a.max_episode_steps}  seed_base={a.seed_base}\n")
        f.write(f"no_distractor_prob={a.no_distractor_prob}  replan_steps={a.replan_steps}\n\n")
        f.write("index verb color shape spatial size prompt successes/total\n")

    client = GrootClient(a.host, a.port)
    tot_s = tot_n = 0
    for i, (verb, color, shape, spatial, size) in enumerate(cells, start=1):
        cell_seed = a.seed_base + i
        vdir = pathlib.Path(a.video_root) / f"{verb}_{size}_{color}_{shape}_{spatial}"
        print(f"[{i}/{total_cells}] {make_instruction(verb,size,color,shape,spatial)}", flush=True)
        try:
            s, n, prompt = run_cell(
                client, verb, color, shape, spatial, size, cell_seed, n_eps,
                a.no_distractor_prob, a.max_episode_steps, a.replan_steps,
                a.sim_backend, a.render_backend, vdir)
            cell_res = f"{s}/{n}"
            tot_s += s
            tot_n += n
        except Exception as e:  # noqa: BLE001
            print(f"  !! cell {i} failed: {e}", flush=True)
            prompt = make_instruction(verb, size, color, shape, spatial)
            cell_res = "NA"
        with rt.open("a") as f:
            f.write(f'{i} {verb} {color} {shape} {spatial} {size} "{prompt}" {cell_res}\n')

    rate = 100.0 * tot_s / tot_n if tot_n else 0.0
    with rt.open("a") as f:
        f.write(f"\noverall_success={tot_s}/{tot_n} ({rate:.1f}%)\n")
    print(f"\nDone: {tot_n} episodes across {total_cells} cells")
    print(f"Overall: {tot_s}/{tot_n} ({rate:.1f}%)")
    print(f"Results: {rt}")


if __name__ == "__main__":
    main()
