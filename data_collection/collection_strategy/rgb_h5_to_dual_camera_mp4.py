#!/usr/bin/env python3
"""
Encode ``*_rgb.h5`` replay trajectories into two MP4s per episode:

- ``<stem>_agent_view.mp4`` — ``base_camera`` RGB (third-person / agent view)
- ``<stem>_in_hand.mp4`` — ``hand_camera`` RGB (wrist / in-hand)

Videos are written next to the source ``.h5`` (same resolution as stored in H5,
typically 256×256 for VerbObjectColor-v1 + Panda wrist cam).

Usage::

  python collection_strategy/rgb_h5_to_dual_camera_mp4.py --motion-dir /path/to/replay_data_segment
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mani_skill.utils.visualization.misc import images_to_video


def _frames_from_traj(traj, cam: str) -> list[np.ndarray]:
    g = traj["obs"]["sensor_data"][cam]["rgb"]
    n = int(g.shape[0])
    out: list[np.ndarray] = []
    for i in range(n):
        x = np.asarray(g[i], dtype=np.uint8)
        # (1, H, W, 3) or (1, 1, H, W, 3) from batched sim
        while x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"expected H×W×3 rgb frame, got shape {x.shape}")
        out.append(x)
    return out


def process_h5(h5_path: Path, fps: int, force: bool) -> None:
    stem = h5_path.stem
    out_dir = str(h5_path.parent)
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[1]),
        )
        for tk in traj_keys:
            traj = f[tk]
            suffix = f"_{tk}" if len(traj_keys) > 1 else ""
            name_a = f"{stem}{suffix}_agent_view"
            name_h = f"{stem}{suffix}_in_hand"
            path_a = h5_path.parent / f"{name_a}.mp4"
            path_h = h5_path.parent / f"{name_h}.mp4"
            if not force and path_a.is_file() and path_h.is_file():
                continue
            try:
                base_frames = _frames_from_traj(traj, "base_camera")
                hand_frames = _frames_from_traj(traj, "hand_camera")
            except KeyError as e:
                print(f"Skip {h5_path} {tk}: missing camera data ({e})")
                continue
            if not base_frames:
                continue
            images_to_video(base_frames, out_dir, video_name=name_a, fps=fps, verbose=False)
            images_to_video(hand_frames, out_dir, video_name=name_h, fps=fps, verbose=False)
            print(f"Wrote {path_a.name}  {path_h.name}")


def main() -> None:
    p = argparse.ArgumentParser(description="RGB H5 -> agent_view + in_hand MP4s")
    p.add_argument(
        "--motion-dir",
        type=Path,
        required=True,
        help="Root directory; all *_rgb.h5 under it are processed (recursive)",
    )
    p.add_argument("--fps", type=int, default=30, help="Video FPS (default 30, match RecordEpisode)")
    p.add_argument("--force", action="store_true", help="Overwrite existing mp4")
    p.add_argument(
        "--stems-file",
        type=Path,
        default=None,
        help="Optional: only encode H5s whose stem (filename without .h5) is listed, one per line",
    )
    args = p.parse_args()
    d = args.motion_dir
    if not d.is_dir():
        raise FileNotFoundError(d)
    all_rgb = sorted(d.rglob("*_rgb.h5"))
    if args.stems_file is not None:
        lines = [
            ln.strip()
            for ln in args.stems_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        want = set(lines)
        all_rgb = [p for p in all_rgb if p.stem in want]
    for h5 in all_rgb:
        process_h5(h5, args.fps, args.force)


if __name__ == "__main__":
    main()
