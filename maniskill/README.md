# ManiSkill fork

A vendored fork of [ManiSkill 3](https://github.com/haosulab/ManiSkill) (based on upstream `main`, early 2026) with the
tabletop environments, motion-planning solutions, and trajectory tools used for data collection and
evaluation in *Scale Up Strategically*. Upstream library code is kept unchanged except where listed below.

`VerbObjectColor-v1` is the environment used by both `data_collection/` and `evaluation/`. One instruction
is defined by verb, color, object shape, size, and spatial relation; up to 3 distractor objects are added.

## Install

```bash
pip install -e maniskill   # from the repository root
```

ManiSkill dependencies (SAPIEN, mplib, ...) are installed by `setup.py`. Motion planning needs Linux (`mplib`).

## Assets

Robot meshes, the table scene and textures are included under `mani_skill/`. Assets that ManiSkill downloads at runtime are handled by ManiSkill itself (`python -m mani_skill.utils.download_asset`).

The `cup`, `car` and `can` meshes are in `asset/`. `tabletop_composite_objects.py` finds this folder by searching the parent directories of the package.

## Registered environments (ours)

| Env ID | File |
|---|---|
| `VerbObjectColor-v1` | `envs/tasks/tabletop/verb_object_color_env.py` |
| `PickCubeColor-v1`, `PickSphereColor-v1`, `PickRingColor-v1` | `envs/tasks/tabletop/pick_up_*_color.py` |
| `PushCubeColor-v1`, `PushSphereColor-v1`, `PushRingColor-v1` | `envs/tasks/tabletop/push_*_color.py` |
| `PullCubeColor-v1`, `PullSphereColor-v1`, `PullRingColor-v1` | `envs/tasks/tabletop/pull_*_color.py` |

The `*Color-v1` single-verb envs are earlier versions kept for compatibility. New data uses `VerbObjectColor-v1`.

## Files added or changed vs upstream

All paths are under `mani_skill/`.

New:
- `envs/tasks/tabletop/verb_object_color_env.py`: the `VerbObjectColor-v1` task (verbs `lift`, `grasp`, `push`,
  `pull`, `rotate`, `slide`; 6 shapes, 6 colors; target and distractor size scales; distractor control via
  `distractor_specs` / `distractor_match_target_geometry`; optional `ood_dual_eval` success masks for evaluation).
- `envs/tasks/tabletop/tabletop_composite_objects.py`: builders for cube, sphere, cup, car, pyramid, star, ring.
- `envs/tasks/tabletop/{pick_up,push,pull}_{cube,sphere,ring}_color.py`, `object_sizes_default.txt`.
- `examples/motionplanning/panda/solutions/verb_object_color.py` and `*_color.py`: scripted motion-planning solutions.
- `trajectory/export_maniskill_trajectory.py` (replay implementation), `trajectory/visualize_trajectory.py`.

Modified:
- `envs/tasks/tabletop/__init__.py`: registers the envs above.
- `examples/motionplanning/panda/run.py`, `examples/motionplanning/panda/solutions/__init__.py`: hooks for the
  new solutions. `examples/motionplanning/base_motionplanner/motionplanner.py`: optional debug prints.
- `trajectory/replay_trajectory.py`: thin entry point that calls `export_maniskill_trajectory`.
- `trajectory/convert_to_lerobot.py`: can read frames from MP4 files when the H5 has no RGB.
- `utils/building/actors/common.py`: adds `build_ring`.
- `agents/robots/panda/panda_wristcam.py`: wrist camera at 256x256.
- `envs/utils/system/backend.py`: accepts `pci:<id>` render backends.
- `utils/structs/actor.py`: skips hide/show for actors without a render body.

Other tasks only differ from upstream in their default camera resolution (256x256).

## License

ManiSkill is released under the Apache-2.0 license (`LICENSE`, `LICENSE-3RD-PARTY`).
