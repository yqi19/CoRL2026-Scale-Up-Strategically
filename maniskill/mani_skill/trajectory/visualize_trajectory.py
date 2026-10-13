"""
Load and visualize ManiSkill trajectory HDF5 files.

Usage:
  python -m mani_skill.trajectory.visualize_trajectory --traj-path demos/PickCube-v1/motionplanning/20260204_154921.rgb+depth+segmentation.pd_joint_pos.physx_cuda.h5
  python -m mani_skill.trajectory.visualize_trajectory --traj-path path/to/file.h5 --traj-idx 0 --save-dir ./frames
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Set headless backend before any import can pull in pyplot (e.g. via mani_skill).
# Avoids "Cannot load backend 'TkAgg'" when running with --save-dir or --summary without a display.
if "--save-dir" in sys.argv or "--summary" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import h5py
import matplotlib
import numpy as np

from mani_skill.utils.io_utils import load_json


def load_h5_data(data: h5py.Group | h5py.Dataset) -> dict[str, Any] | np.ndarray:
    """Recursively load an HDF5 group/dataset into Python dict/ndarray."""
    if isinstance(data, h5py.Dataset):
        return data[:]
    out = {}
    for k in data.keys():
        out[k] = load_h5_data(data[k])
    return out


def load_trajectory_file(traj_path: str):
    """
    Load all trajectory data from a ManiSkill .h5 file and its .json metadata.

    Returns:
        h5_file: open h5py.File (read-only)
        json_data: dict with keys "episodes", "env_info"
        traj_keys: list of trajectory group names, e.g. ["traj_0", "traj_1", ...]
    """
    traj_path = Path(traj_path)
    if not traj_path.suffix == ".h5":
        raise ValueError(f"Expected .h5 file, got {traj_path}")
    json_path = traj_path.with_suffix(".json")
    if not json_path.exists():
        json_path = traj_path.parent / (traj_path.stem.split(".")[0] + ".json")
    if not json_path.exists():
        raise FileNotFoundError(f"No JSON found for {traj_path}")
    json_data = load_json(str(json_path))
    h5_file = h5py.File(str(traj_path), "r")
    traj_keys = sorted([k for k in h5_file.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_")[1]))
    return h5_file, json_data, traj_keys


def get_trajectory_data(h5_file: h5py.File, traj_key: str) -> dict[str, Any]:
    """Load a single trajectory group into a nested dict of numpy arrays."""
    return load_h5_data(h5_file[traj_key])


def get_sensor_data(traj_data: dict, camera_name: str = "base_camera") -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Extract rgb, depth, segmentation from trajectory obs for a given camera.
    Returns (rgb, depth, segmentation). Any can be None if not present.
    """
    try:
        sd = traj_data["obs"]["sensor_data"][camera_name]
    except KeyError:
        return None, None, None
    rgb = sd.get("rgb")
    depth = sd.get("depth")
    seg = sd.get("segmentation")
    if rgb is not None and not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)
    if depth is not None and not isinstance(depth, np.ndarray):
        depth = np.array(depth)
    if seg is not None and not isinstance(seg, np.ndarray):
        seg = np.array(seg)
    return rgb, depth, seg


def depth_to_rgb(depth: np.ndarray, invalid_val: float = 0.0) -> np.ndarray:
    """Convert depth array to RGB for visualization (viridis-like)."""
    import matplotlib.pyplot as plt
    depth = np.squeeze(depth)
    valid = np.isfinite(depth) & (depth > invalid_val)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmin, vmax = np.min(depth[valid]), np.max(depth[valid])
    if vmax <= vmin:
        vmax = vmin + 1
    norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    try:
        cmap = plt.colormaps["viridis"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("viridis")
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def segmentation_to_rgb(seg: np.ndarray) -> np.ndarray:
    """Convert segmentation IDs to RGB using a deterministic colormap."""
    import matplotlib.pyplot as plt
    seg = np.squeeze(seg).astype(np.int64)
    ids = np.unique(seg)
    ids = ids[ids >= 0]
    if len(ids) == 0:
        return np.zeros((*seg.shape, 3), dtype=np.uint8)
    try:
        cmap = plt.colormaps["tab20"].resampled(max(len(ids), 20))
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("tab20", max(len(ids), 20))
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for i, sid in enumerate(ids):
        out[seg == sid] = (np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
    return out


def visualize_frame(
    traj_data: dict,
    step: int,
    camera_name: str = "base_camera",
    ax_rgb=None,
    ax_depth=None,
    ax_seg=None,
    fig=None,
):
    """Plot one timestep: RGB, depth, segmentation."""
    import matplotlib.pyplot as plt
    rgb, depth, seg = get_sensor_data(traj_data, camera_name)
    n_plots = sum([rgb is not None, depth is not None, seg is not None])
    if n_plots == 0:
        if fig is not None:
            fig.suptitle("No sensor_data found for this camera")
        return

    if ax_rgb is None:
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        idx = 0
        if rgb is not None:
            ax_rgb = axes[idx]; idx += 1
        else:
            ax_rgb = None
        if depth is not None:
            ax_depth = axes[idx]; idx += 1
        else:
            ax_depth = None
        if seg is not None:
            ax_seg = axes[idx]
        else:
            ax_seg = None

    if rgb is not None:
        img = rgb[step] if rgb.ndim > 3 else rgb
        ax_rgb.imshow(img)
        ax_rgb.set_title("RGB")
        ax_rgb.axis("off")
    if depth is not None:
        d = depth[step] if depth.ndim > 3 else depth
        drgb = depth_to_rgb(d)
        ax_depth.imshow(drgb)
        ax_depth.set_title("Depth")
        ax_depth.axis("off")
    if seg is not None:
        s = seg[step] if seg.ndim > 3 else seg
        srgb = segmentation_to_rgb(s)
        ax_seg.imshow(srgb)
        ax_seg.set_title("Segmentation")
        ax_seg.axis("off")

    if fig is not None:
        actions = traj_data.get("actions")
        n_steps = actions.shape[0] if actions is not None else 0
        success = traj_data.get("success")
        succ = success[step] if success is not None and step < len(success) else None
        fig.suptitle(f"Step {step} / {n_steps}  |  success@step={succ}")


def run_interactive(traj_path: str, camera_name: str = "base_camera"):
    """Interactive viewer: sliders for trajectory index and timestep."""
    import matplotlib.pyplot as plt
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    h5_file, json_data, traj_keys = load_trajectory_file(traj_path)
    episodes = json_data.get("episodes", [])
    env_info = json_data.get("env_info", {})

    # Cache one trajectory at a time to avoid loading all into memory
    current_traj_key = [traj_keys[0]]
    current_data = [get_trajectory_data(h5_file, current_traj_key[0])]

    def get_steps():
        return current_data[0]["actions"].shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax_rgb, ax_depth, ax_seg = axes

    def update(traj_idx: int, step: int):
        key = traj_keys[traj_idx]
        if key != current_traj_key[0]:
            current_traj_key[0] = key
            current_data[0] = get_trajectory_data(h5_file, key)
        traj_data = current_data[0]
        rgb, depth, seg = get_sensor_data(traj_data, camera_name)
        for ax in axes:
            ax.clear()
        visualize_frame(traj_data, step, camera_name, ax_rgb, ax_depth, ax_seg, fig)
        plt.draw()

    n_trajs = len(traj_keys)
    traj_idx_slider = plt.axes([0.15, 0.02, 0.35, 0.03])
    step_slider = plt.axes([0.6, 0.02, 0.3, 0.03])
    s_traj = plt.Slider(traj_idx_slider, "Traj", 0, n_trajs - 1, valinit=0, valstep=1)
    max_steps = get_steps()
    s_step = plt.Slider(step_slider, "Step", 0, max(0, max_steps - 1), valinit=0, valstep=1)

    def on_traj_change(val):
        idx = int(s_traj.val)
        current_key = traj_keys[idx]
        if current_key != current_traj_key[0]:
            current_traj_key[0] = current_key
            current_data[0] = get_trajectory_data(h5_file, current_key)
        new_max = current_data[0]["actions"].shape[0] - 1
        s_step.valmax = new_max
        s_step.ax.set_xlim(s_step.valmin, s_step.valmax)
        if s_step.val > new_max:
            s_step.set_val(new_max)
        update(idx, int(s_step.val))

    def on_step_change(val):
        update(int(s_traj.val), int(s_step.val))

    s_traj.on_changed(on_traj_change)
    s_step.on_changed(on_step_change)

    plt.subplots_adjust(bottom=0.12, top=0.92)
    fig.suptitle(f"{Path(traj_path).name}  |  {env_info.get('env_id', '')}  |  {n_trajs} trajectories")
    update(0, 0)
    plt.show()
    h5_file.close()


def run_save_frames(traj_path: str, save_dir: str, camera_name: str = "base_camera", max_trajs: int | None = None, max_steps_per_traj: int | None = None):
    """Load all data and save visualization frames to save_dir."""
    import matplotlib.pyplot as plt
    h5_file, json_data, traj_keys = load_trajectory_file(traj_path)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    if max_trajs is not None:
        traj_keys = traj_keys[:max_trajs]
    for ti, key in enumerate(traj_keys):
        traj_data = get_trajectory_data(h5_file, key)
        n_steps = traj_data["actions"].shape[0]
        steps = range(n_steps)
        if max_steps_per_traj is not None:
            steps = list(steps)[:max_steps_per_traj]
        for step in steps:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            visualize_frame(traj_data, step, camera_name, axes[0], axes[1], axes[2], fig)
            out_file = save_path / f"traj{ti}_step{step:04d}.png"
            fig.savefig(out_file, dpi=100, bbox_inches="tight")
            plt.close(fig)
    h5_file.close()
    print(f"Saved frames to {save_path}")


def print_summary(traj_path: str):
    """Print a short summary of the trajectory file without loading full obs."""
    h5_file, json_data, traj_keys = load_trajectory_file(traj_path)
    episodes = json_data.get("episodes", [])
    env_info = json_data.get("env_info", {})
    print(f"File: {traj_path}")
    print(f"Env: {env_info.get('env_id', '')}")
    print(f"Trajectories: {len(traj_keys)}")
    for i, key in enumerate(traj_keys[:5]):
        g = h5_file[key]
        n_actions = g["actions"].shape[0]
        print(f"  {key}: {n_actions} steps")
    if len(traj_keys) > 5:
        print(f"  ... and {len(traj_keys) - 5} more")
    h5_file.close()


def main():
    parser = argparse.ArgumentParser(description="Load and visualize ManiSkill trajectory HDF5")
    parser.add_argument("--traj-path", type=str, required=True, help="Path to .h5 trajectory file")
    parser.add_argument("--camera", type=str, default="base_camera", help="Camera name in sensor_data")
    parser.add_argument("--save-dir", type=str, default=None, help="If set, save frames to this dir instead of interactive")
    parser.add_argument("--max-trajs", type=int, default=None, help="Max trajectories to export (when using --save-dir)")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per trajectory to export (when using --save-dir)")
    parser.add_argument("--summary", action="store_true", help="Only print summary and exit")
    args = parser.parse_args()

    # Set non-GUI backend before any pyplot import when running headless (save-dir or summary).
    # This avoids "Cannot load backend 'TkAgg'" when no display is available.
    if args.save_dir or args.summary:
        matplotlib.use("Agg")

    if args.summary:
        print_summary(args.traj_path)
        return

    if args.save_dir:
        run_save_frames(args.traj_path, args.save_dir, camera_name=args.camera, max_trajs=args.max_trajs, max_steps_per_traj=args.max_steps)
    else:
        run_interactive(args.traj_path, camera_name=args.camera)


if __name__ == "__main__":
    main()
