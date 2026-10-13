# collection_strategy

Strategy data for compositional generalization. A strategy picks a fixed set of training combinations
(the "support") on a factor grid; episodes are then drawn from that support until `--num-demos` motion-planning
demos succeed. `random` draws uniformly with replacement from its support; the structured strategies cycle through
reshuffled full passes of theirs. The support is seeded by `--seed`.

## Settings and entry points

| Setting | Grid | Python collector | Pipeline script |
|---|---|---|---|
| Pairwise, verb x color / verb x object / color x object | 2 factors from the 6-value vocab; third factor sampled from the first `--third-pool-size` values (default 2) | `collect_pairwise.py --experiment <exp>` | `<exp>/collect_convert.sh` |
| Pairwise, verb x size / verb x spatial / color x size / spatial x size | same index patterns mapped to size / spatial labels; remaining factors fixed (e.g. `red cube`) | `collect_pairwise_attribute.py --experiment <exp>` | `<exp>/collect_convert.sh` |
| Pairwise, color x spatial | 6 colors x spatial labels with the same index patterns; verb `lift`, shape `cube` fixed | `collect_color_spatial.py` | `color_spatial/collect_convert.sh` |
| Pairwise, spatial x object | 5 spatial x 6 shapes grid, `--factor-count 12/20/30` (number of training cells); verb `lift`, color `red` fixed | `collect_spatial_object.py` | `spatial_object/collect_convert.sh` |
| Verb-Color-Object | full (verb, color, shape) triples, `--factor-count 15/20/30` = 3^3 / 4^3 / 6^3 index cube | `collect_verb_color_object.py` | `verb_color_object/collect_convert.sh` |
| All-Factor | verb x color x shape x spatial x size (6x6x6x5x6 = 6480 tuples) | `collect_all_factor.py` | `all_factor/collect_convert.sh`, `all_factor/collect_convert_segmented.sh` |

Each `<exp>/collect.py` is a thin wrapper that calls the collector listed above (with `--experiment <exp>` where needed).
Pairwise `--factor-count` is 6 / 12 / 18 (index grids 2x2 / 4x4 / 6x6; `stair` at f6 uses a 3x3 list).

## Strategies

| Paper | Code | Pairwise (`lib/pairwise_factor_patterns.py`) | Verb-Color-Object (`lib/triple_factor_patterns.py`) | All-Factor (`lib/all_factor_support_f50.py`) |
|---|---|---|---|---|
| V (ours) | `stair` (All-Factor: `stairvss`) | f6 / f12: fixed staircase lists; f18: row 0 + main diagonal (i,i) + sub-diagonal (i,i+1). Spatial-Object: column-first staircase (`spatial_object_stair`) | verb ray (i,0,0) + shape ray (0,0,k) + space diagonal + verb->color->shape zigzag | staircase walk from (0,0,0,0,0) (size->spatial->shape->color->verb) + verb/size/shape rays anchored at color=`black`, spatial=`left` + 1 random cell |
| L | `Lrandom` | f6: 2x2; f12: row 0 + column 0 + (1,1), (1,2); f18: row 0 + column 0 + 3 random interior cells. Spatial-Object: row 0 + column 0 + random interior cells up to the factor count | three axis rays + 1 / 2 / 4 random cells | five single-factor rays through the origin + 4 random cells (f12: verb and color rays + 1 random) |
| Random | `random` | 3 / 6 / 9 uniform random cells for f6 / f12 / f18. Spatial-Object: factor-count uniform random cells | `factor // 3` cells, greedy max-min (L-inf) spread | f12: 12 uniform random cells; f20 / f40: 5 / 10 cells in a sub-grid |
| Complete | not implemented | | | |

Support sizes (computed from the code): pairwise f6/12/18: `stair` 5/10/15, `Lrandom` 4/9/14, `random` 3/6/9;
Spatial-Object f12/20/30: 12/20/30 for all three; Verb-Color-Object f15/20/30: `stair` 10/15/25, `Lrandom` 8/12/20,
`random` 5/6/10; All-Factor: `stairvss` 42 (f50), `Lrandom` 29 (f50), `random` 12 (f12).

## Collector arguments (shared)

| Argument | Default | Meaning |
|---|---|---|
| `--strategy` | required | see above |
| `--factor-count` | pairwise 6 (color x spatial, spatial x object: 12), Verb-Color-Object required, All-Factor 50 | grid size |
| `--num-demos` | 300 | target number of successful motion-planning demos |
| `--record-base`, `--record-segment` | `outputs/demo_gen`, `<strategy>[_f<N>]` | output directory `<record-base>/<experiment>/<record-segment>/` |
| `--seed` | 0 | support sampling, shuffles, third-factor sampling, MP seeds |
| `--only-count-success` / `--max-attempts` | on / 3 | retry motion planning per episode (0 = unlimited) |
| `--num-distractors`, `--distractor-max` | 2, 2 (All-Factor, color x spatial, spatial x object: 2, 3) | distractor objects per episode (max 3) |
| `--third-pool-size` | 2 | pairwise only: third factor drawn from the first N values |
| `--obj-xy-range XMIN XMAX YMIN YMAX` | none | object xy sampling range (pipeline default `-0.1 0.1 -0.1 0.1`) |
| `--size-pool`, `--small-large-distractor-policy` | all six sizes, `bernoulli` | All-Factor only: allowed size labels and distractor policy for `small` / `large` |
| `--sim-backend`, `--render-backend` | `cpu` | ManiSkill backends |
| `--save-video` | off | save motion-planning videos |

## Other files

- `build_pairwise_task_map.py`, `build_attribute_task_map.py`: write `task_map.json` (H5 stem -> instruction).
- `convert_maniskill_to_lerobot.py`: replayed `*_rgb.h5` -> LeRobot dataset (images resized to 256x256).
- `convert_maniskill_to_lerobot_video.py`: optional video-encoded LeRobot dataset (`SKIP_CONVERT_VIDEO=0` in the
  attribute pipelines; requirements in `requirements_lerobot_video.txt`).
- `rgb_h5_to_dual_camera_mp4.py`: optional base / wrist camera MP4s from `*_rgb.h5` (`SAVE_DUAL_MP4=1`).
- `lib/training_vocab.py`: verb / color / shape vocabularies and third-factor lists.
- `lib/all_factor_scene.py`: size labels -> target / distractor scales for All-Factor.
- `lib/pairwise_task_language.py`: stem parsing and instruction templates.
- `lib/vulkan_headless_default.sh`: picks a Vulkan driver for headless CPU rendering.
