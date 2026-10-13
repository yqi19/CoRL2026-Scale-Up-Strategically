#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Convert conflict_maniskill datasets (parquet + embedded PNG bytes, no videos)
into GR00T-compatible LeRobot v2 format:
  - Extract embedded PNG bytes -> MP4 videos (one per camera per episode)
  - Rename parquet columns to standard names (observation.state, action)
  - Add annotation column for language (annotation.human.task_description)
  - Write modality.json
  - Update info.json with video counts
  - Copy meta files (episodes.jsonl, tasks.jsonl) as-is

Output per category:
  <output_root>/<category>/  <- groot-compatible LeRobot v2 dataset

Usage:
    python prepare_conflict_data.py \
        --src_root <data_collection output>/demo_conflict \
        --out_root <gr00t_data_root> \
        --categories color_object color_size ...
        [--num_demos 300]
"""

import argparse
import io
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


CATEGORIES = [
    "color_object",
    "color_size",
    "color_spatial",
    "size_object",
    "spatial_object",
    "spatial_size",
    "verb_color",
    "verb_object",
    "verb_size",
    "verb_spatial",
]

# Panda arm: 7 joints + 1 gripper = 8-dim state/action
# state dims 0-6: joint positions, dim 7: gripper width
MODALITY_JSON = {
    "state": {
        "arm": {"start": 0, "end": 7},
        "gripper": {"start": 7, "end": 8},
    },
    "action": {
        "arm": {"start": 0, "end": 7},
        "gripper": {"start": 7, "end": 8},
    },
    "video": {
        "image": {"original_key": "observation.images.image"},
        "wrist_image": {"original_key": "observation.images.wrist_image"},
    },
    "annotation": {
        "human.task_description": {"original_key": "annotation.human.task_description"},
    },
}


def decode_image(cell) -> np.ndarray:
    """Decode a parquet image cell (dict with 'bytes' key or raw bytes) -> uint8 HWC numpy."""
    if isinstance(cell, dict):
        raw = cell["bytes"]
    elif isinstance(cell, bytes):
        raw = cell
    else:
        raise ValueError(f"Unknown image cell type: {type(cell)}")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def frames_to_mp4(frames: list[np.ndarray], out_path: Path, fps: int = 30) -> None:
    """Write a list of HWC uint8 numpy frames as an H264 MP4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    for frame_arr in frames:
        frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def prepare_category(
    src_root: Path,
    out_root: Path,
    category: str,
    num_demos: int,
    fps: int = 30,
) -> None:
    src_dataset = src_root / category / str(num_demos) / "huggingface_data" / category / "conflict"
    out_dataset = out_root / category

    print(f"\n{'='*60}")
    print(f"Processing: {category}  ({num_demos} demos)")
    print(f"  src: {src_dataset}")
    print(f"  out: {out_dataset}")

    if not src_dataset.exists():
        print(f"  SKIP: source not found")
        return

    meta_src = src_dataset / "meta"
    data_src = src_dataset / "data"

    # Load metadata
    with open(meta_src / "info.json") as f:
        info = json.load(f)

    with open(meta_src / "tasks.jsonl") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    task_map = {t["task_index"]: t["task"] for t in tasks}

    with open(meta_src / "episodes.jsonl") as f:
        episodes_meta = [json.loads(line) for line in f if line.strip()]

    total_episodes = info["total_episodes"]
    chunk_size = info["chunks_size"]
    data_path_pattern = info["data_path"]

    # Prepare output dirs
    out_meta = out_dataset / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)
    out_data = out_dataset / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    total_videos_created = 0

    # Process each episode
    for ep_meta in tqdm(episodes_meta, desc=category):
        ep_idx = ep_meta["episode_index"]
        chunk_idx = ep_idx // chunk_size

        # Load source parquet
        src_parquet = src_dataset / data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=ep_idx
        )
        df = pd.read_parquet(src_parquet)

        # -- Build converted parquet --
        new_df = pd.DataFrame()
        new_df["observation.state"] = df["state"]
        new_df["action"] = df["actions"]
        new_df["timestamp"] = df["timestamp"]
        new_df["frame_index"] = df["frame_index"]
        new_df["episode_index"] = df["episode_index"]
        new_df["index"] = df["index"]
        new_df["task_index"] = df["task_index"]
        # Add annotation column (task index -> same integer, loader looks up tasks.jsonl)
        new_df["annotation.human.task_description"] = df["task_index"]

        # Save converted parquet (same chunk structure)
        out_chunk_dir = out_data / f"chunk-{chunk_idx:03d}"
        out_chunk_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = out_chunk_dir / f"episode_{ep_idx:06d}.parquet"
        new_df.to_parquet(out_parquet, index=False)

        # -- Extract videos per camera --
        n_frames = len(df)
        for cam_key in ("image", "wrist_image"):
            frames = [decode_image(df[cam_key].iloc[i]) for i in range(n_frames)]
            vid_dir = (
                out_dataset
                / "videos"
                / f"chunk-{chunk_idx:03d}"
                / f"observation.images.{cam_key}"
            )
            vid_path = vid_dir / f"episode_{ep_idx:06d}.mp4"
            frames_to_mp4(frames, vid_path, fps=fps)
            total_videos_created += 1

    print(f"  Created {total_videos_created} MP4 files")

    # -- Write modality.json --
    with open(out_meta / "modality.json", "w") as f:
        json.dump(MODALITY_JSON, f, indent=4)

    # -- Write updated info.json --
    new_info = dict(info)
    new_info["total_videos"] = total_videos_created
    new_info["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    new_info["features"] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": [
                "joint0", "joint1", "joint2", "joint3",
                "joint4", "joint5", "joint6", "gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [8],
            "names": [
                "joint0", "joint1", "joint2", "joint3",
                "joint4", "joint5", "joint6", "gripper",
            ],
        },
        "observation.images.image": {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channels"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "annotation.human.task_description": {"dtype": "int64", "shape": [1], "names": None},
    }
    with open(out_meta / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)

    # -- Copy episodes.jsonl and tasks.jsonl verbatim --
    shutil.copy(meta_src / "episodes.jsonl", out_meta / "episodes.jsonl")
    shutil.copy(meta_src / "tasks.jsonl", out_meta / "tasks.jsonl")

    print(f"  Done: {out_dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--num_demos", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    for cat in args.categories:
        prepare_category(
            src_root=args.src_root,
            out_root=args.out_root,
            category=cat,
            num_demos=args.num_demos,
            fps=args.fps,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
