# common

| File | Purpose |
|---|---|
| `paths.py` | Adds the ManiSkill fork (`MANISKILL_ROOT`, default `../maniskill`) and data-collection code (`DATA_COLLECTION_ROOT`, default `../data_collection`) to `sys.path`, so `collection_strategy` and `scripts.run_verb_color_shape_motion_planning` can be imported. |
| `run_conflict.sh` | Conflict (FDR) rollouts for one checkpoint and one factor pair. Samples held-out pairs, writes the job list, and runs `gr00t/groot_conflict_eval.py` or `pi0/pi0_conflict_eval.py` against a running server. |
| `summarize_seeds.py` | Averages the `overall_*` lines of per-seed result files (used by the grid and All-Factor runners). |
| `pairwise_env.py` | Model-agnostic builder for the verb_color, color_object, verb_object and verb_size grids (36 cells each). Returns the env kwargs, reset options and instruction for a cell. |

```bash
# conflict rollouts: <gr00t|pi0> <experiment> [seed] [total_episodes] [out_dir]
PORT=5555 bash common/run_conflict.sh gr00t color_size 42 200 results/conflict_gr00t

# list the 36 cells of a grid, optionally reset one env
python common/pairwise_env.py verb_color --seed 42 --third-pool-size 2 [--smoke-reset --sim-backend cpu]
```

Conflict output layout (read by `vlm_judge/vlm_eval.py`):
```
<out_dir>/<experiment>/<experiment>_ood_seed<seed>.txt
<out_dir>/<experiment>/experiments/ood_<idx>_<experiment>_<i>_<j>_<run_type>_<timestamp>/
    video/ep000.mp4, video/ep000_wrist.mp4, success_rate.txt
```
