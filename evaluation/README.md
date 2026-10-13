# Evaluation

Evaluation code for "Scale Up Strategically: Learning Compositional Generalization via Bias-Aware Evaluation and Data Collection for Robotic Manipulation" (CoRL 2026).

All evaluations run in ManiSkill on the `VerbObjectColor-v1` environment from the fork in `../maniskill`. Instructions combine five factors: verb, color, object (shape), size and spatial position. Policies run as separate servers (GR00T N1.7 over zmq, pi0 / pi0.5 over the OpenPI websocket), and a lightweight ManiSkill client drives the rollouts.

| Evaluation | What it measures | Entry point |
|---|---|---|
| Conflict / FDR | Which factor a policy follows when trained on factor-correlated data and tested on held-out combinations | `common/run_conflict.sh`, then `vlm_judge/` |
| Pairwise grid | Compositional generalization on held-out cells of a two-factor grid | `gr00t/run_pairwise_grid.sh`, `pi0/run_pairwise_grid.sh` |
| All-Factor | Success when all five factors appear in one instruction | `gr00t/run_all_factor.sh` |

## Layout

```
evaluation/
├── common/      shared env helpers, conflict job runner, seed summarizer
├── gr00t/       GR00T N1.7 zmq client and runners (conflict, grid, all-factor)
├── pi0/         pi0 / pi0.5 OpenPI clients and runners (conflict, grid)
└── vlm_judge/   Gemini-2.5-Flash judge and FDR / FDH computation
```

## Setup

1. ManiSkill client environment:
   ```bash
   pip install -e maniskill            # from the repository root
   pip install -r evaluation/requirements.txt
   ```
   The clients also import `collection_strategy/` and `scripts/run_verb_color_shape_motion_planning.py` from `../data_collection`. `common/paths.py` adds `../maniskill` and `../data_collection` to `sys.path`; override with `MANISKILL_ROOT` / `DATA_COLLECTION_ROOT`.
2. GR00T server: a separate environment with [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (N1.7). Set `GR00T_REPO` (and optionally `GR00T_PYTHON`) for `gr00t/start_server.sh`.
3. pi0 / pi0.5 server: an OpenPI checkout that defines the `pi0_maniskill` / `pi05_maniskill` configs (see `pi0/README.md`). Install `openpi-client` into the client environment.
4. VLM judge: `export GEMINI_API_KEY=...` (or `GOOGLE_API_KEY`).

All scripts take the interpreter from `PYTHON` (default `python`) and the server from `HOST` / `PORT`.

## Checkpoints

No weights are stored in this repository.

Checkpoints are linked from the [project page](https://yqi19.github.io/scale-up-strategically/). The FDR / FDH set has one checkpoint per factor pair: `color_object`, `color_size`, `color_spatial`, `size_object`, `spatial_object`, `spatial_size`, `verb_color`, `verb_object`, `verb_size`, `verb_spatial`.

## Protocol

Values are the defaults of the runner scripts.

| | Conflict / FDR | Pairwise grid (HARD) | All-Factor |
|---|---|---|---|
| Experiments | 10 factor pairs | GR00T: color_size, color_spatial, verb_spatial, spatial_size, spatial_object, verb_size; pi0: color_size, color_spatial | one 5-factor instruction per cell |
| Episodes per seed | 200 sampled held-out pairs x 2 scenes (one per factor) = 400 rollouts, 1 episode each | target 200: `ceil(200 / #cells)` episodes per cell over the full grid | 200 cells sampled (seed 42) from 6x6x6x5x4, 1 episode each |
| Seeds | 42 | 42, 40, 41 | 40, 41, 42 (seed base) |
| Max steps | 300 (GR00T), 150 (pi0) | env time limit (120 for `VerbObjectColor-v1`; GR00T client also caps at 150) | 500 |
| Replan (actions executed per chunk) | 5 (GR00T), 10 (pi0) | 10 | 5 |
| Task difficulty | env default | 1.5 | env default |
| Distractors | 1 object carrying the competing factor value; no object matches both factors | Same-color distractor(s) that differ only in the probed factor (size scale, spatial anchor or shape) | 70% of episodes have none; otherwise up to 1 size-comparison distractor (smaller/larger only) |
| Sim backend | gpu (GR00T), cpu (pi0) | gpu (GR00T), cpu (pi0) | gpu |
| Metric | VLM label per rollout, then FDR / FDH | env success rate, mean over seeds | env success rate, mean over seeds |

Grid cells are built with `get_env_id_and_color` from the ManiSkill fork, which routes push/pull + cube/sphere + red/yellow/blue to the legacy `*Color-v1` envs (50-step limit, no distractor kwargs). Size pairs in the conflict experiments are only swapped within (small, large), (smaller, larger) and (smallest, largest). Spatial anchors are left / right / middle / front / behind at +-0.10 m.

## Commands

Start a policy server in one terminal:
```bash
GR00T_REPO=/path/to/Isaac-GR00T CUDA_VISIBLE_DEVICES=0 \
  bash evaluation/gr00t/start_server.sh /path/to/ckpt/verb_color 5555
```

Conflict rollouts (FDR), then the VLM judge and FDR / FDH:
```bash
PORT=5555 bash evaluation/common/run_conflict.sh gr00t verb_color 42 200 results/conflict_gr00t
python evaluation/vlm_judge/vlm_eval.py --results-root results/conflict_gr00t --label gr00t
python evaluation/vlm_judge/fdr_fdh.py --vlm-dir results/conflict_gr00t/vlm_eval/gr00t
```

Pairwise grid (3 seeds). Checkpoint names follow `<pair>_<strategy>_f<k>`; the V strategy has the code name `stair`:
```bash
PORT=5555 bash evaluation/gr00t/run_pairwise_grid.sh color_spatial results/grid/color_spatial_stair_f18
PORT=8000 bash evaluation/pi0/run_pairwise_grid.sh   color_spatial results/grid_pi0/color_spatial_stair_f18
```

All-Factor (3 seeds):
```bash
PORT=5555 bash evaluation/gr00t/run_all_factor.sh results/all_factor/all_factor_stair_f50_n400
```

## FDR and FDH

For a factor pair (f1, f2), the VLM judge labels each conflict rollout as following f1, f2, or neither. With N_f1 and N_f2 the counts,

    FDR = (N_f1 - N_f2) / (N_f1 + N_f2)

FDR > 0 means f1 dominates. FDH ranks the five factors with a Copeland score over the 10 pairwise FDRs: a factor wins a pair when its FDR margin exceeds tau = 5%, loses when it is below -tau, and ties otherwise; score = wins - losses. `vlm_judge/fdr_fdh.py` implements both. `--env-dir` computes the same FDR from the env's dual-axis success counts instead of the VLM labels.

## Not included

- pi0.5 clients for the 3-factor Verb-Color-Object and All-Factor settings live in the OpenPI fork referenced in `pi0/README.md`.
- Genie-Envisioner runners were dropped: they depend on an unpublished inference package.
- `common/pairwise_env.py` builds the verb_color / color_object / verb_object / verb_size grids but has no policy runner here.
