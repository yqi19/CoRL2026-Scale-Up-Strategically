#!/usr/bin/env python3
import argparse
import shutil

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Create a dummy *_rgb.h5 from a raw ManiSkill trajectory h5."
    )
    parser.add_argument("--src", required=True, help="Source raw trajectory .h5")
    parser.add_argument("--dst", required=True, help="Destination *_rgb.h5")
    parser.add_argument("--h", type=int, default=64, help="Dummy image height")
    parser.add_argument("--w", type=int, default=64, help="Dummy image width")
    args = parser.parse_args()

    shutil.copy2(args.src, args.dst)
    with h5py.File(args.dst, "a") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        for tk in traj_keys:
            traj = f[tk]
            if "actions" not in traj:
                continue
            t = int(traj["actions"].shape[0])
            obs = traj.require_group("obs")
            sensor_data = obs.require_group("sensor_data")
            agent = obs.require_group("agent")
            if "qpos" not in agent:
                agent.create_dataset("qpos", data=np.zeros((t, 8), dtype=np.float32))
            rgb = np.zeros((t, args.h, args.w, 3), dtype=np.uint8)
            for cam in ("base_camera", "hand_camera"):
                cam_group = sensor_data.require_group(cam)
                if "rgb" in cam_group:
                    del cam_group["rgb"]
                cam_group.create_dataset("rgb", data=rgb, compression="gzip")


if __name__ == "__main__":
    main()
