#!/usr/bin/env python3
"""
ManiSkill HDF5 → LeRobot format via ``LeRobotDataset`` (camera streams as **encoded videos**).

Field names follow `AgibotTech/Genie-Envisioner` README (LeRobot layout): use
``observation.images.*`` for cameras, ``observation.state`` for proprioception, and
``action`` (not ``actions``) so configs can set ``state_key`` / ``action_key`` / ``valid_cam``
without rewrites.

- ``FPS`` matches ``convert_maniskill_to_lerobot.py`` (replay / frame-aligned).
- ``IMAGE_SIZE``, ``robot_type="panda"``, video encoding via LeRobot 0.4+ API.

Install in any env with ``lerobot>=0.4,<0.6``::

  pip install -r collection_strategy/requirements_lerobot_video.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Iterable, Iterator, Optional

import h5py
import numpy as np
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# Same constants as convert_maniskill_to_lerobot.py (REPO_ROOT = data_collection/).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_NAME = "lerobot_maniskill_video"
# Match image-based converter / ManiSkill replay timing.
FPS = 30
IMAGE_SIZE = (256, 256)

# Genie-Envisioner README: observation.images.{cam}, observation.state, action
KEY_IMAGE_BASE = "observation.images.base_camera"
KEY_IMAGE_WRIST = "observation.images.hand_camera"
KEY_STATE = "observation.state"
KEY_ACTION = "action"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "lerobot_dataset_video"


def _load_converter_mp4_helpers():
    """Load only ``load_frames_from_mp4`` from ``convert_to_lerobot.py`` (without importing mani_skill)."""
    path = REPO_ROOT.parent / "maniskill" / "mani_skill" / "trajectory" / "convert_to_lerobot.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_convert_to_lerobot_mp4", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_frames_from_mp4


_load_frames_from_mp4 = None


def _resize_rgb(arr: np.ndarray) -> np.ndarray:
    from PIL import Image

    img = np.asarray(arr)
    if img.shape[:2] == IMAGE_SIZE:
        return img
    return np.array(Image.fromarray(img).resize((IMAGE_SIZE[1], IMAGE_SIZE[0])))


def _has_h5_rgb(traj_group: h5py.Group) -> bool:
    try:
        _ = traj_group["obs"]["sensor_data"]["base_camera"]["rgb"]
        _ = traj_group["obs"]["sensor_data"]["hand_camera"]["rgb"]
        return True
    except (KeyError, TypeError):
        return False


def load_episodes_from_h5(
    h5_path: Path,
    task_name: str,
    *,
    video_dir: Optional[Path] = None,
) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per trajectory: ``rgb_base`` and ``rgb_hand`` as ``[T, H, W, 3]`` uint8."""
    if not h5_path.exists():
        return
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_")[1]))
        if not traj_keys:
            return
        first = f[traj_keys[0]]
        use_h5_rgb = _has_h5_rgb(first)
        has_state = "obs" in first and "agent" in first["obs"] and "qpos" in first["obs"]["agent"]

        if not use_h5_rgb:
            if video_dir is None or not video_dir.is_dir():
                raise FileNotFoundError(
                    f"H5 has no embedded RGB cameras; set --video-dir (e.g. parent of .h5 with 0.mp4, 1.mp4, …): {h5_path}"
                )
            global _load_frames_from_mp4
            if _load_frames_from_mp4 is None:
                _load_frames_from_mp4 = _load_converter_mp4_helpers()
            if _load_frames_from_mp4 is None:
                raise RuntimeError("Cannot load MP4 helper from mani_skill/trajectory/convert_to_lerobot.py")

        for ep_idx, traj_key in enumerate(traj_keys):
            traj = f[traj_key]
            actions = np.asarray(traj["actions"][:], dtype=np.float32)
            T = int(actions.shape[0])
            if use_h5_rgb:
                rgb_base = np.asarray(traj["obs"]["sensor_data"]["base_camera"]["rgb"][:T])
                rgb_hand = np.asarray(traj["obs"]["sensor_data"]["hand_camera"]["rgb"][:T])
            else:
                vid_path = video_dir / f"{ep_idx}.mp4"
                if not vid_path.is_file():
                    raise FileNotFoundError(f"Expected episode video missing: {vid_path}")
                frames = _load_frames_from_mp4(vid_path, max_frames=T + 1)
                rgb_base = np.asarray(frames[:T])
                rgb_hand = rgb_base.copy()

            if has_state:
                qpos = np.asarray(traj["obs"]["agent"]["qpos"][:T], dtype=np.float32)
                state = qpos[:, :8] if qpos.shape[1] >= 8 else np.pad(qpos, ((0, 0), (0, 8 - qpos.shape[1])))
            else:
                state = np.zeros((T, 8), dtype=np.float32)
            yield task_name, rgb_base, rgb_hand, state, actions


def _maybe_start_image_writer(dataset: LeRobotDataset) -> None:
    writer = getattr(dataset, "image_writer", None)
    if writer is None and hasattr(dataset, "start_image_writer"):
        dataset.start_image_writer()


def _save_episode(dataset: LeRobotDataset) -> None:
    save = dataset.save_episode
    try:
        save(encode_videos=True)
    except TypeError:
        save()


def _add_frame_compat(dataset: LeRobotDataset, frame: dict, task_label: str) -> None:
    """
    Add one frame across LeRobot API variants.

    - Newer variants may require: add_frame(frame, task=...)
    - Older variants accept:      add_frame(frame)
    """
    add_frame = dataset.add_frame
    frame_no_task = {k: v for k, v in frame.items() if k != "task"}
    try:
        # Newer API variant: task is an explicit argument, not a frame feature.
        add_frame(frame_no_task, task=str(task_label))
        return
    except TypeError:
        # Older API variant: no explicit task argument.
        pass
    add_frame(frame)


def add_episode_to_dataset(
    dataset: LeRobotDataset,
    task_label: str,
    rgb_base: np.ndarray,
    rgb_hand: np.ndarray,
    state: np.ndarray,
    actions: np.ndarray,
) -> None:
    n = len(actions)
    if n == 0:
        return
    for i in range(n):
        frame = {
            "task": str(task_label),
            KEY_IMAGE_BASE: _resize_rgb(rgb_base[i]),
            KEY_IMAGE_WRIST: _resize_rgb(rgb_hand[i]),
            KEY_STATE: state[i],
            KEY_ACTION: actions[i],
        }
        _add_frame_compat(dataset, frame, task_label)
    _save_episode(dataset)


def _iter_h5_files(root: Path, *, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.h5" if recursive else "*.h5"
    yield from sorted(root.glob(pattern))


def _load_task_map_dict(args: argparse.Namespace) -> dict:
    if args.task_map_file is not None:
        raw = Path(args.task_map_file).read_text(encoding="utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid --task-map-file JSON: {e}") from e
    if args.task_map:
        try:
            return json.loads(args.task_map)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid --task-map JSON: {e}") from e
    raise SystemExit("Provide --task-map-file or --task-map")


def _build_h5_to_task(args: argparse.Namespace) -> dict[Path, str]:
    if args.traj_path is not None:
        if not args.task_name:
            raise SystemExit("--task-name is required when using --traj-path")
        p = args.traj_path.resolve()
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")
        return {p: str(args.task_name)}

    if args.demos_dir is not None:
        demos_dir = Path(args.demos_dir).resolve()
        if not demos_dir.is_dir():
            raise SystemExit(f"--demos-dir not a directory: {demos_dir}")
        task_map_dict = _load_task_map_dict(args)
        stem_to_paths: dict[str, list[Path]] = {}
        for p in demos_dir.rglob("*.h5"):
            stem_to_paths.setdefault(p.stem, []).append(p)
        h5_to_task: dict[Path, str] = {}
        for stem, label in task_map_dict.items():
            if stem not in stem_to_paths:
                raise SystemExit(f"No .h5 with stem {stem!r} under {demos_dir}")
            paths = stem_to_paths[stem]
            if len(paths) > 1:
                raise SystemExit(f"Multiple .h5 with stem {stem!r}: {paths}")
            h5_to_task[paths[0]] = str(label)
        return h5_to_task

    if args.h5_root is not None:
        root = args.h5_root.resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        h5_list = list(_iter_h5_files(root, recursive=args.recursive))
        if not h5_list:
            raise SystemExit(f"No .h5 under {root}")
        if args.task_map or args.task_map_file:
            try:
                stem_to_label = _load_task_map_dict(args)
            except SystemExit:
                raise
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid task map JSON: {e}") from e
            out: dict[Path, str] = {}
            for h5 in h5_list:
                if h5.stem not in stem_to_label:
                    raise SystemExit(f"No task label for stem {h5.stem!r} in --task-map")
                out[h5] = str(stem_to_label[h5.stem])
            return out
        return {h5: h5.stem for h5 in h5_list}

    raise SystemExit("Provide one of: --traj-path, --demos-dir, or --h5-root")


def _resolve_video_dir(h5_path: Path, video_dir: Optional[str]) -> Optional[Path]:
    if video_dir is None:
        return None
    if video_dir == "auto":
        return h5_path.parent
    return Path(video_dir).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ManiSkill H5 → LeRobot (LeRobotDataset, use_videos=True)."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--traj-path", type=Path, help="Single ManiSkill .h5 file.")
    src.add_argument("--demos-dir", type=Path, help="Scan demos tree for .h5 (requires --task-map).")
    src.add_argument("--h5-root", type=Path, help="Scan directory for .h5 files; task = stem unless --task-map.")

    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output dataset root.")
    p.add_argument("--task-name", type=str, default=None, help="Task label (required with --traj-path).")
    p.add_argument(
        "--task-map",
        type=str,
        default=None,
        help='JSON: stem -> label for --demos-dir or --h5-root, e.g. \'{"run123":"Pick cube"}\'',
    )
    p.add_argument(
        "--task-map-file",
        type=Path,
        default=None,
        help="JSON file (same format as --task-map). Preferred for large maps.",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_NAME,
        help="LeRobot dataset repo_id stored in meta (Genie: domain / dataset name).",
    )
    p.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help="LeRobot video codec id (see lerobot.datasets.video_utils.VALID_VIDEO_CODECS), "
        "e.g. h264, hevc, auto, h264_nvenc. Not ffmpeg names like libx264.",
    )
    p.add_argument("--max-episodes", type=int, default=None, help="Max episodes per H5 file.")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="With --h5-root, use **/*.h5 (default: only *.h5 in that directory).",
    )
    p.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help='When H5 has no embedded RGB: directory with 0.mp4, 1.mp4, … Use "auto" for .h5 parent.',
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-dir before writing if it exists.",
    )
    p.add_argument(
        "--skip-if-output-complete",
        action="store_true",
        help="If output-dir already has meta/info.json and data/*.parquet, exit 0 without changes.",
    )
    return p


def _output_looks_complete(root: Path) -> bool:
    info = root / "meta" / "info.json"
    if not info.is_file():
        return False
    for p in (root / "data").rglob("*.parquet") if (root / "data").is_dir() else []:
        if p.is_file():
            return True
    return False


def _normalize_vcodec(vcodec: str) -> str:
    """LeRobot >=0.4 uses logical names (h264, hevc, …), not ffmpeg encoder names."""
    legacy = {"libx264": "h264", "libx265": "hevc"}
    if vcodec in legacy:
        print(f"Note: mapping deprecated vcodec {vcodec!r} -> {legacy[vcodec]!r} for LeRobotDataset.")
        return legacy[vcodec]
    return vcodec


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.vcodec = _normalize_vcodec(args.vcodec)
    h5_to_task = _build_h5_to_task(args)
    h5_files = list(h5_to_task.keys())

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and args.skip_if_output_complete and _output_looks_complete(output_dir):
        print(f"Skip conversion (complete dataset already at {output_dir})")
        return 0
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_dir} (pass --overwrite to replace)")
        shutil.rmtree(output_dir)
        print(f"Removed existing dataset at {output_dir}")

    features = {
        KEY_IMAGE_BASE: {"dtype": "video", "shape": (*IMAGE_SIZE, 3)},
        KEY_IMAGE_WRIST: {"dtype": "video", "shape": (*IMAGE_SIZE, 3)},
        KEY_STATE: {"dtype": "float32", "shape": (8,)},
        KEY_ACTION: {"dtype": "float32", "shape": (8,)},
    }
    create_kw: dict = {
        "repo_id": args.repo_id,
        "fps": FPS,
        "features": features,
        "root": output_dir,
        "robot_type": "panda",
        "use_videos": True,
        "image_writer_threads": 8,
        "image_writer_processes": 4,
    }
    try:
        dataset = LeRobotDataset.create(**create_kw, vcodec=args.vcodec)
    except TypeError:
        create_kw.pop("vcodec", None)
        dataset = LeRobotDataset.create(**create_kw)
    _maybe_start_image_writer(dataset)

    for h5_path in h5_files:
        task_name = h5_to_task[h5_path]
        if not h5_path.exists():
            print(f"Skip (not found): {h5_path}")
            continue
        vdir = _resolve_video_dir(h5_path, args.video_dir)
        try:
            episodes = list(load_episodes_from_h5(h5_path, task_name, video_dir=vdir))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Skip {h5_path}: {e}")
            continue
        if args.max_episodes is not None:
            episodes = episodes[: args.max_episodes]
        print(f"Converting {h5_path.name} -> task {task_name!r} ({len(episodes)} episodes)")
        for task_label, rgb_base, rgb_hand, state, actions in tqdm(episodes, desc=task_name):
            add_episode_to_dataset(dataset, task_label, rgb_base, rgb_hand, state, actions)

    print("Conversion complete.")
    print(f"Output saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
