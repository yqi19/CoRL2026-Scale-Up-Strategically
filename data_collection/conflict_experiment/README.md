# conflict_experiment

Conflict (factor-correlation) data for the Factor Dominance Rate (FDR) experiments. In the training data two
factors are fully correlated: only the index diagonal (k, k) of their grid is collected, for example
`lift`+`red`, `grasp`+`yellow`, ... for verb x color. A policy fine-tuned on this data is then evaluated on
held-out (off-diagonal) combinations to see which factor it follows.

Trajectory stems start with `conflict_` and the output segment is `<experiment>/conflict/`.

## Files

| File | Purpose |
|---|---|
| `collect_conflict.py` | collector for `verb_color`, `verb_object`, `color_object` (diagonal over the 6 training values) |
| `collect_conflict_attribute.py` | collector for `verb_size`, `verb_spatial`, `object_spatial`, `color_spatial`, `color_size`, `spatial_size`, `spatial_object`, `size_object` (diagonal length 5 when spatial is involved) |
| `lib/conflict_sampling.py` | diagonal pairs and third-factor pool parsing |
| `collect_convert.sh` | collect -> replay -> LeRobot for the first three experiments |
| `collect_convert_attribute.sh` | the same pipeline for the attribute experiments |

## Collector arguments

| Argument | Default | Meaning |
|---|---|---|
| `--experiment` | required | see table above |
| `--third-indices` | `0,1` | indices into the third-factor list; the third factor is sampled uniformly from these values |
| `--num-demos` | 300 | target number of successful motion-planning demos |
| `--record-base`, `--record-segment` | `outputs/demo_gen`, `conflict` | output directory `<record-base>/<experiment>/<record-segment>/` |
| `--seed` | 0 | shuffles, third-factor sampling, MP seeds |
| `--only-count-success`, `--max-attempts` | on, 3 | retry motion planning per episode (0 = unlimited) |
| `--num-distractors`, `--distractor-max` | 2, 2 | distractor objects per episode (`collect_conflict.py`) |
| `--sim-backend`, `--render-backend` | `cpu` | ManiSkill backends |
| `--obj-xy-range`, `--save-video` | none, off | object xy range, motion-planning videos |

The third factor is the shape for `verb_color` and `verb_size` / `verb_spatial` / `spatial_size`, the color for
`verb_object`, and the verb for `color_object`, `object_spatial`, `spatial_object`, `size_object`, `color_spatial`,
`color_size`.

## Pipelines

Run from `data_collection/`:

```bash
PIPELINE_OUT=./outputs COLLECTION_EXPERIMENT=verb_object NUM_DEMOS=300 THIRD_INDICES=0,1 \
  ./conflict_experiment/collect_convert.sh

PIPELINE_OUT=./outputs COLLECTION_EXPERIMENT=spatial_size NUM_DEMOS=300 \
  ./conflict_experiment/collect_convert_attribute.sh
```

Output: `<PIPELINE_OUT>/<NUM_DEMOS>/{original_data,replay_data,huggingface_data}/<experiment>/conflict/`, with
the same environment variables as the strategy pipelines (`MP_OVERSAMPLE`, `SKIP_*`, `CLEAN_*`, `PYTHON`,
`LEROBOT_PYTHON`, ...; see the header of each script).
