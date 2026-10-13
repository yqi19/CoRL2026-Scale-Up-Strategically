#!/usr/bin/env python3
"""
Convert ManiSkill replayed trajectory H5 files to LeRobot format (pairwise / generic).

Options:

- ``--task-map-file`` to load the stem -> instruction JSON from disk.
- ``--repo-id`` for the LeRobot dataset id.

Install::

  pip install -r requirements_lerobot_convert.txt

Usage::

  python collection_strategy/convert_maniskill_to_lerobot.py \\
    --demos-dir /path/to/demos_root \\
    --task-map-file /path/to/task_map.json \\
    --repo-id maniskill_pairwise_verb_color_random \\
    --output-dir /path/to/lerobot_dataset
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_NAME = "maniskill_pairwise"
FPS = 30
IMAGE_SIZE = (256, 256)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "lerobot_dataset"


def load_episodes_from_h5(h5_path: Path, task_name: str):
    """Load all trajectories from one H5. Yields (task_name, rgb_base, rgb_hand, state, actions) per episode."""
    if not h5_path.exists():
        return
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_")[1]))
        if not traj_keys:
            return
        first = f[traj_keys[0]]
        has_state = "obs" in first and "agent" in first["obs"] and "qpos" in first["obs"]["agent"]
        for traj_key in traj_keys:
            traj = f[traj_key]
            actions = np.asarray(traj["actions"][:], dtype=np.float32)
            T = actions.shape[0]
            rgb_base = traj["obs"]["sensor_data"]["base_camera"]["rgb"][:T]
            rgb_hand = traj["obs"]["sensor_data"]["hand_camera"]["rgb"][:T]
            if has_state:
                qpos = np.asarray(traj["obs"]["agent"]["qpos"][:T], dtype=np.float32)
                state = qpos[:, :8] if qpos.shape[1] >= 8 else np.pad(qpos, ((0, 0), (0, 8 - qpos.shape[1])))
            else:
                state = np.zeros((T, 8), dtype=np.float32)
            yield task_name, rgb_base, rgb_hand, state, actions


def add_episode_to_dataset(dataset: LeRobotDataset, task_label: str, rgb_base, rgb_hand, state, actions):
    """Add one episode (one trajectory) to the dataset."""
    from PIL import Image

    n = len(actions)
    if n == 0:
        return
    for i in range(n):
        img = np.asarray(rgb_base[i])
        if img.shape[:2] != IMAGE_SIZE:
            img = np.array(Image.fromarray(img).resize((IMAGE_SIZE[1], IMAGE_SIZE[0])))
        wrist_img = np.asarray(rgb_hand[i])
        if wrist_img.shape[:2] != IMAGE_SIZE:
            wrist_img = np.array(Image.fromarray(wrist_img).resize((IMAGE_SIZE[1], IMAGE_SIZE[0])))
        frame = {
            "task": str(task_label),
            "image": img,
            "wrist_image": wrist_img,
            "state": state[i],
            "actions": actions[i],
        }
        dataset.add_frame(frame)
    dataset.save_episode()


def main():
    import argparse

    p = argparse.ArgumentParser(description="Convert ManiSkill H5 to LeRobot format (LeRobotDataset API)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output dataset root")
    p.add_argument(
        "--demos-dir",
        type=Path,
        default=None,
        help="Root dir of demos; H5 discovered by stem keys in task map",
    )
    p.add_argument("--max-episodes", type=int, default=None, help="Max episodes per H5 (default: all)")
    p.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_NAME,
        help="LeRobot dataset repo_id / name",
    )
    p.add_argument(
        "--task-map",
        type=str,
        default=None,
        help="JSON dict: H5 filename stem -> task label (use --task-map-file for large maps)",
    )
    p.add_argument(
        "--task-map-file",
        type=Path,
        default=None,
        help="Path to JSON file (same format as --task-map). Overrides --task-map when set.",
    )
    args = p.parse_args()

    if args.demos_dir is None:
        raise ValueError("--demos-dir is required for this script")

    demos_dir = Path(args.demos_dir)
    if not demos_dir.is_dir():
        raise FileNotFoundError(f"--demos-dir not a directory: {demos_dir}")

    if args.task_map_file is not None:
        raw = args.task_map_file.read_text(encoding="utf-8")
        task_map_dict = json.loads(raw)
    elif args.task_map:
        task_map_dict = json.loads(args.task_map)
    else:
        raise ValueError("--demos-dir requires --task-map-file or --task-map")

    stem_to_paths: dict[str, list[Path]] = {}
    for pth in demos_dir.rglob("*.h5"):
        stem_to_paths.setdefault(pth.stem, []).append(pth)

    h5_to_task: dict[Path, str] = {}
    for stem, label in task_map_dict.items():
        if stem not in stem_to_paths:
            raise FileNotFoundError(f"No .h5 with stem {stem!r} under {demos_dir}")
        paths = stem_to_paths[stem]
        if len(paths) > 1:
            raise ValueError(f"Multiple .h5 with stem {stem!r} under {demos_dir}: {paths}")
        h5_to_task[paths[0]] = str(label)

    h5_files = sorted(h5_to_task.keys(), key=lambda x: str(x))

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed existing dataset at {output_dir}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_dir,
        robot_type="panda",
        fps=FPS,
        features={
            "image": {"dtype": "image", "shape": (*IMAGE_SIZE, 3)},
            "wrist_image": {"dtype": "image", "shape": (*IMAGE_SIZE, 3)},
            "state": {"dtype": "float32", "shape": (8,)},
            "actions": {"dtype": "float32", "shape": (8,)},
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    for h5_path in h5_files:
        task_name = h5_to_task[h5_path]
        if not h5_path.exists():
            print(f"Skip (not found): {h5_path}")
            continue
        episodes = list(load_episodes_from_h5(h5_path, task_name))
        if args.max_episodes is not None:
            episodes = episodes[: args.max_episodes]
        print(f"Converting {h5_path.name} -> task '{task_name}' ({len(episodes)} episodes)")
        for task_label, rgb_base, rgb_hand, state, actions in tqdm(episodes, desc=task_name):
            add_episode_to_dataset(dataset, task_label, rgb_base, rgb_hand, state, actions)

    print("Conversion complete.")
    print(f"Output saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
