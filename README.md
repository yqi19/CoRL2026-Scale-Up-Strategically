# Scale Up Strategically

Code for **Scale Up Strategically: Learning Compositional Generalization via Bias-Aware Evaluation and Data Collection for Robotic Manipulation** (CoRL 2026).

[Project page](https://yqi19.github.io/scale-up-strategically/) · [Paper](https://arxiv.org/abs/2607.21582)

Fine-tuned robot policies over-rely on some instruction factors (e.g., color) and ignore others (e.g., verb, size). This repository contains the two parts of the paper:

1. **FDR / FDH evaluation.** Measure *instruction factor bias* with the Factor Dominance Rate (FDR) for each factor pair and the Factor Dominance Hierarchy (FDH) across factors.
2. **Bias-aware data collection.** Collect demonstrations with a strategy (V, L, Random), fine-tune a policy, and measure its success rate on held-out instructions.

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Instruction factors](#instruction-factors)
- [Part 1: FDR / FDH evaluation](#part-1-fdr--fdh-evaluation)
- [Part 2: Data collection strategies and policy success rate](#part-2-data-collection-strategies-and-policy-success-rate)
- [Data and checkpoints](#data-and-checkpoints)
- [Citation](#citation)

## Repository layout

| Folder | What it does |
|---|---|
| [`maniskill/`](maniskill) | ManiSkill 3 fork with our tabletop task `VerbObjectColor-v1`, motion-planning solvers and assets. |
| [`data_collection/`](data_collection) | Scripted demonstration collection: conflict data for FDR, and strategy data (V, L, Random). Converts to LeRobot. |
| [`finetune/`](finetune) | Fine-tuning recipes for GR00T-N1.7 and π0.5 / π0. |
| [`evaluation/`](evaluation) | ManiSkill rollout clients, the VLM judge, and FDR / FDH computation. |

Each folder has its own README with all options. The sections below give the end-to-end workflow.

## Installation

```bash
git clone https://github.com/yqi19/CoRL2026-Scale-Up-Strategically
cd CoRL2026-Scale-Up-Strategically

# ManiSkill client environment (simulation, data collection, evaluation clients)
pip install -e maniskill
pip install -r data_collection/requirements_lerobot_convert.txt
pip install -r evaluation/requirements.txt
```

Motion planning needs Linux (`mplib`). Policies are trained and served from their own environments:
[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) for GR00T-N1.7 (see [`finetune/gr00t`](finetune/gr00t)) and [OpenPI](https://github.com/Physical-Intelligence/openpi) for π0.5 / π0 (see [`finetune/pi05`](finetune/pi05)). The evaluation clients talk to them over the network.

## Instruction factors

Every instruction in `VerbObjectColor-v1` is built from five factors:

| Factor | Values |
|---|---|
| verb | `lift`, `grasp`, `push`, `pull`, `rotate`, `slide` |
| color | `red`, `yellow`, `blue`, `orange`, `green`, `black` |
| object | `cube`, `sphere`, `cup`, `car`, `pyramid`, `star` |
| size | `small`, `large`, `smaller`, `larger`, `smallest`, `largest` |
| spatial | `left`, `right`, `middle`, `front`, `behind` |

## Part 1: FDR / FDH evaluation

**Idea.** For a factor pair (f1, f2), fine-tune a policy on data where the two factors always co-occur, then test it on held-out combinations whose scene lets the policy follow either f1 or f2, but not both. A VLM judge labels which factor each rollout followed.

### Step 1. Collect conflict data

Only the diagonal of the two-factor grid is collected (for verb x color: `lift`+`red`, `grasp`+`yellow`, ...). There are 10 factor pairs: `verb_color`, `verb_object`, `color_object`, `verb_size`, `verb_spatial`, `color_size`, `color_spatial`, `size_object`, `spatial_size`, `spatial_object`.

```bash
cd data_collection
# verb_color, verb_object, color_object
PIPELINE_OUT=./outputs/conflict/verb_color COLLECTION_EXPERIMENT=verb_color NUM_DEMOS=300 \
  ./conflict_experiment/collect_convert.sh
# pairs involving size or spatial
PIPELINE_OUT=./outputs/conflict/verb_size COLLECTION_EXPERIMENT=verb_size NUM_DEMOS=300 \
  ./conflict_experiment/collect_convert_attribute.sh
```

Each run collects 300 demonstrations by motion planning, replays them to RGB, and writes a LeRobot dataset to `outputs/conflict/<pair>/300/huggingface_data/<pair>/conflict/`. See [`data_collection/conflict_experiment`](data_collection/conflict_experiment).

### Step 2. Fine-tune one policy per pair

GR00T-N1.7 (convert to the GR00T format, then train; one run per pair):

```bash
cd finetune/gr00t
python gr00t_overrides/scripts/prepare_conflict_data.py \
    --src_root ../../data_collection/outputs/conflict --out_root <DATA_ROOT> \
    --num_demos 300 --fps 30 --categories verb_color
GR00T_DIR=/path/to/Isaac-GR00T DATA_PATH=<DATA_ROOT>/verb_color \
OUTPUT_DIR=<CKPT_ROOT>/verb_color NUM_GPUS=4 bash finetune.sh
```

For π0.5 / π0 see [`finetune/pi05`](finetune/pi05).

### Step 3. Run conflict rollouts

Start the policy server for one checkpoint, then roll out 200 held-out pairs x 2 scenes (400 rollouts, one video each):

```bash
GR00T_REPO=/path/to/Isaac-GR00T CUDA_VISIBLE_DEVICES=0 \
  bash evaluation/gr00t/start_server.sh <CKPT_ROOT>/verb_color 5555
PORT=5555 bash evaluation/common/run_conflict.sh gr00t verb_color 42 200 results/conflict_gr00t
```

Use `pi0` instead of `gr00t` (and `evaluation/pi0/start_server.sh`) for OpenPI policies.

### Step 4. Label rollouts with the VLM judge

Gemini-2.5-Flash watches each rollout video and answers which factor the robot followed (`f1`, `f2`, or `neither`):

```bash
export GEMINI_API_KEY=...
python evaluation/vlm_judge/vlm_eval.py --results-root results/conflict_gr00t --label gr00t
```

### Step 5. Compute FDR and FDH

```bash
python evaluation/vlm_judge/fdr_fdh.py --vlm-dir results/conflict_gr00t/vlm_eval/gr00t --tau 0.05
```

- **FDR** for a pair (f1, f2), with N_f1 and N_f2 the rollouts that followed each factor:

  FDR(f1, f2) = (N_f1 - N_f2) / (N_f1 + N_f2) ∈ (-1, 1)

  Positive means f1 dominates, negative means f2 dominates, near 0 means balanced.
- **FDH** ranks the five factors by a Copeland score over the 10 pairs: a factor scores +1 when its FDR margin exceeds τ = 5% and 0 for a loss or tie.

`--env-dir results/conflict_gr00t` computes the same FDR from the simulator's dual-axis success counts instead of the VLM labels. Details: [`evaluation/vlm_judge`](evaluation/vlm_judge).

## Part 2: Data collection strategies and policy success rate

**Idea.** The instruction space grows combinatorially, so a fixed demonstration budget cannot cover every combination. A strategy decides which combinations of factor values are collected; the fine-tuned policy is then evaluated on all cells, including held-out ones.

### Strategies

| Paper | Code name (pairwise, Verb-Color-Object) | Code name (All-Factor) | Collected cells |
|---|---|---|---|
| V (ours) | `stair` | `stairvss` | one full row on the under-grounded factor, the main diagonal and the sub-diagonal |
| L | `Lrandom` | `Lrandom` | first row and first column, plus a few random cells |
| Random | `random` | `random` | uniformly random cells |

`FACTOR_COUNT` (the factor change *f*) is the size of the grid: pairwise 6 / 12 / 18 (Spatial-Object 12 / 20 / 30), Verb-Color-Object 15 / 20 / 30, All-Factor 50. `NUM_DEMOS` is the demonstration budget *n*. Exact supports are listed in [`data_collection/collection_strategy`](data_collection/collection_strategy).

### Step 1. Collect strategy data

```bash
cd data_collection
# pairwise grid, e.g. color x spatial with V at f = 18
PIPELINE_OUT=./outputs STRATEGY=stair FACTOR_COUNT=18 NUM_DEMOS=200 \
  ./collection_strategy/color_spatial/collect_convert.sh

# three factors: verb x color x object, V at f = 20
PIPELINE_OUT=./outputs STRATEGY=stair FACTOR_COUNT=20 NUM_DEMOS=200 \
  ./collection_strategy/verb_color_object/collect_convert.sh

# all five factors, V at f = 50 with n = 400
PIPELINE_OUT=./outputs/all_factor STRATEGY=stairvss FACTOR_COUNT=50 NUM_DEMOS=400 \
  NUM_DISTRACTORS=3 DISTRACTOR_MAX=3 \
  SIZE_POOL=small,large,smaller,larger,smallest,largest SMALL_LARGE_DISTRACTOR_POLICY=random_uniform \
  ./collection_strategy/all_factor/collect_convert.sh
```

Pairwise grids: `verb_color`, `verb_object`, `color_object`, `verb_size`, `verb_spatial`, `color_size`, `color_spatial`, `spatial_size`, `spatial_object`. The LeRobot dataset is written to `outputs/<n>/huggingface_data/<experiment>/<strategy>_f<f>/`.

### Step 2. Fine-tune

Train one policy per (experiment, strategy, f, n) dataset with the recipes in [`finetune/gr00t`](finetune/gr00t) or [`finetune/pi05`](finetune/pi05).

### Step 3. Evaluate success rate

Start the policy server, then run the grid evaluation over 3 seeds (42, 40, 41):

```bash
# pairwise grid: every cell of the grid, ~200 episodes per seed
PORT=5555 bash evaluation/gr00t/run_pairwise_grid.sh color_spatial results/grid/color_spatial_stair_f18
PORT=8000 bash evaluation/pi0/run_pairwise_grid.sh   color_spatial results/grid_pi0/color_spatial_stair_f18

# All-Factor: 200 cells sampled from the 6 x 6 x 6 x 5 x 4 space
PORT=5555 bash evaluation/gr00t/run_all_factor.sh results/all_factor/stairvss_f50_n400
```

Each run writes one log per seed and a `SUMMARY.txt` with the mean success rate over seeds.

| | Pairwise grid | All-Factor |
|---|---|---|
| Episodes per seed | ~200, spread evenly over all grid cells | 200 sampled cells, 1 episode each |
| Seeds | 42, 40, 41 | 40, 41, 42 |
| Distractors | same-color objects that differ only in the probed factor | none in 70% of episodes |
| Metric | simulator success rate, mean over seeds | simulator success rate, mean over seeds |

Full protocol (max steps, replanning, task difficulty): [`evaluation`](evaluation).

## Data and checkpoints

Datasets and fine-tuned checkpoints are linked from the [project page](https://yqi19.github.io/scale-up-strategically/).

## Citation

```bibtex
@inproceedings{qi2026scaleup,
  title     = {Scale Up Strategically: Learning Compositional Generalization via
               Bias-Aware Evaluation and Data Collection for Robotic Manipulation},
  author    = {Qi, Yu and Ye, Zhang and Xu, Xinyi and Lu, Yuxuan and Sandhu, Amitoj and
               Hu, Boce and Huang, Haojie and Tremblay, Jonathan and Wong, Lawson L.S.},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
