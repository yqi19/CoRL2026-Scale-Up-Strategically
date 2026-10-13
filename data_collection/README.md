# Data collection

Scripted demonstration collection in ManiSkill (`VerbObjectColor-v1`) for the two kinds of training data in the paper:

1. **Conflict data for FDR** (`conflict_experiment/`): two factors are fully correlated in training
   (only the index diagonal of the two-factor grid is collected), so the policy can be tested on held-out
   combinations to measure which factor dominates.
2. **Strategy data for compositional generalization** (`collection_strategy/`): the training combinations are
   chosen by a data-collection strategy (V, L, Random) on pairwise grids, the three-factor
   Verb-Color-Object cube, and the All-Factor space.

Every instruction is built from five factors:

| Factor | Values |
|---|---|
| verb | `lift`, `grasp`, `push`, `pull`, `rotate`, `slide` |
| color | `red`, `yellow`, `blue`, `orange`, `green`, `black` |
| object (shape) | `cube`, `sphere`, `cup`, `car`, `pyramid`, `star` |
| size | `small`, `large`, `smaller`, `larger`, `smallest`, `largest` |
| spatial | `left`, `right`, `middle`, `front`, `behind` |

## Install

```bash
# from the repository root
pip install -e maniskill
pip install -r data_collection/requirements_lerobot_convert.txt   # LeRobot conversion
```

Motion planning needs Linux (`mplib`). LeRobot conversion can run in a separate environment: set
`LEROBOT_PYTHON=/path/to/that/env/bin/python`. ManiSkill steps use `PYTHON` (default: `python` on `PATH`).
With `SIM_BACKEND=cpu RENDER_BACKEND=cpu` (the default), `collection_strategy/lib/vulkan_headless_default.sh` selects
the Lavapipe Vulkan driver if it is installed; set `MANISKILL_USE_GPU=1` to use the NVIDIA driver instead.

All commands below are run from `data_collection/`.

## Strategy names

| Paper | Pairwise and Verb-Color-Object | All-Factor |
|---|---|---|
| V (ours) | `stair` | `stairvss` |
| L | `Lrandom` | `Lrandom` |
| Random | `random` | `random` |
| Complete | not implemented as a strategy | not implemented as a strategy |

- `stair` (pairwise): one full row + main diagonal + sub-diagonal (f6 / f12 use fixed staircase lists);
  Spatial-Object uses a column-first staircase on its 5 x 6 grid.
- `stair` (Verb-Color-Object): verb ray + object ray + space diagonal + 3D staircase.
- `stairvss` (All-Factor): staircase walk over all five axes + verb/size/shape rays + 1 random cell.
- `Lrandom`: row 0 + column 0 (axis rays in 3D / 5D) plus a few random cells.
- `random`: uniform random cells (Verb-Color-Object: greedy max-min spread).

See `collection_strategy/README.md` for the exact supports.

`FACTOR_COUNT` selects the grid size: pairwise 6 / 12 / 18 (Spatial-Object: 12 / 20 / 30), Verb-Color-Object
15 / 20 / 30, All-Factor 50 (`random`: 12).

## Pipeline and output layout

Every `collect_convert.sh` script runs three steps:

1. **Collect**: motion planning writes raw trajectories. `MP_RUNS = max(NUM_DEMOS, ceil(NUM_DEMOS * MP_OVERSAMPLE))`
   successful plans are collected (`MP_OVERSAMPLE=1.15` by default).
2. **Replay**: raw files are copied to `replay_data/` and replayed with `--obs-mode rgb` until `NUM_DEMOS` succeed;
   their stems are written to `selected_rgb_stems.txt`.
3. **Convert**: a `task_map.json` (H5 stem to language instruction) is built and the selected episodes are converted
   to a LeRobot dataset (256x256 base and wrist cameras).

```
<PIPELINE_OUT>/<NUM_DEMOS>/
  original_data/<experiment>/<segment>/VerbObjectColor-v1/motionplanning/*.h5|*.json
  replay_data/<experiment>/<segment>/...  *_rgb.h5, selected_rgb_stems.txt, task_map.json
  huggingface_data/<experiment>/<segment>/   LeRobot dataset
```

`<segment>` is `<STRATEGY>_f<FACTOR_COUNT>` for strategy data and `conflict` for conflict data. Trajectory stems are
`<strategy>_<idx>_<verb>_<color>_<shape>` (All-Factor adds `_<spatial>_<size>`). Collection resumes from existing
files; `CLEAN_RECORD=1` (default) deletes the segment first. Use `SKIP_COLLECT`, `SKIP_REPLAY`, `SKIP_CONVERT` to run
single steps.

## Examples

Pairwise grid, Verb x Color, V at f = 12:

```bash
PIPELINE_OUT=./outputs STRATEGY=stair FACTOR_COUNT=12 NUM_DEMOS=300 \
  ./collection_strategy/verb_color/collect_convert.sh
```

Other pairwise grids: `verb_object`, `color_object`, `verb_size`, `verb_spatial`, `color_size`, `spatial_size`,
`color_spatial`, and `spatial_object` (5 spatial x 6 shape grid, `FACTOR_COUNT` 12 / 20 / 30).

Three-factor Verb-Color-Object, V at f = 20:

```bash
PIPELINE_OUT=./outputs STRATEGY=stair FACTOR_COUNT=20 NUM_DEMOS=300 \
  ./collection_strategy/verb_color_object/collect_convert.sh
```

All-Factor, V at f = 50 with 400 demos. These are the settings in the original launch scripts for `stairvss` and
`Lrandom`; `random` was launched with `FACTOR_COUNT=12` and the script defaults for distractors:

```bash
PIPELINE_OUT=./outputs/all_factor_stairvss_f50 STRATEGY=stairvss FACTOR_COUNT=50 NUM_DEMOS=400 \
  NUM_DISTRACTORS=3 DISTRACTOR_MAX=3 \
  SIZE_POOL=small,large,smaller,larger,smallest,largest SMALL_LARGE_DISTRACTOR_POLICY=random_uniform \
  ./collection_strategy/all_factor/collect_convert.sh
```

For `Lrandom` and for `random` at `NUM_DEMOS=800`, the original runs used
`SEGMENT_SIZE=400 ./collection_strategy/all_factor/collect_convert_segmented.sh` with the same variables. It restarts the
collector every 400 demos (a long single process stopped producing successful plans after about 870 episodes).

Conflict data for FDR, Verb x Color:

```bash
PIPELINE_OUT=./outputs COLLECTION_EXPERIMENT=verb_color NUM_DEMOS=300 \
  ./conflict_experiment/collect_convert.sh
PIPELINE_OUT=./outputs COLLECTION_EXPERIMENT=verb_size NUM_DEMOS=300 \
  ./conflict_experiment/collect_convert_attribute.sh
```

## Layout

```
data_collection/
  collection_strategy/   strategy data (pairwise, Verb-Color-Object, All-Factor); see its README
  conflict_experiment/   conflict data for FDR; see its README
  scripts/               run_verb_color_shape_motion_planning.py (motion-planning entry used by all collectors),
                         create_dummy_rgb_h5.py (optional replay fallback in the verb_color / verb_object /
                         color_object scripts, REPLAY_DUMMY_ON_FAILURE=1)
  requirements_lerobot_convert.txt
```
