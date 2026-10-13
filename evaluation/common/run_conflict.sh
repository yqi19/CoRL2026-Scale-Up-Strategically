#!/usr/bin/env bash
set -euo pipefail
# run_conflict.sh - conflict (FDR) rollouts for one checkpoint / one experiment.
#
# Samples `total_episodes` held-out (i, j) factor pairs, and for each pair runs
# one job per factor (2 x total_episodes jobs, 1 episode each). Every episode
# video is saved for the VLM judge; the env's dual-axis success is summarised
# as overall_<factor1>_success / overall_<factor2>_success.
#
# A policy server must already be running at ${HOST}:${PORT}
# (see gr00t/start_server.sh or pi0/start_server.sh).
#
# Usage:
#   bash common/run_conflict.sh <gr00t|pi0> <experiment> [seed=42] [total_episodes=200] [out_dir]
#
# experiment:
#   verb_color | verb_object | color_object | verb_size | verb_spatial |
#   size_object | color_size | color_spatial | spatial_size | spatial_object
#
# Output (default out_dir = results/conflict_<policy>):
#   <out_dir>/<experiment>/<experiment>_ood_seed<seed>.txt
#   <out_dir>/<experiment>/experiments/ood_<idx>_<experiment>_<i>_<j>_<run_type>_<ts>/video/ep000.mp4
#
# Env vars (defaults: gr00t / pi0):
#   HOST 127.0.0.1    PORT 5555 / 8000
#   SIM_BACKEND gpu / cpu    (pi0 uses the CPU simulator)
#   MAX_EPISODE_STEPS 300 / 150    REPLAN_STEPS 5 / 10    SEED_BASE 0
#   PYTHON python   (interpreter of the ManiSkill client environment)

POLICY="${1:-}"
EXPERIMENT="${2:-}"
SEED="${3:-42}"
TOTAL_EPISODES_TARGET="${4:-200}"
OUT_DIR="${5:-results/conflict_${POLICY}}"
THIRD_SEED="${SEED}"

if [[ -z "${POLICY}" || -z "${EXPERIMENT}" ]]; then
  echo "Usage: $0 <gr00t|pi0> <experiment> [seed] [total_episodes] [out_dir]"
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${POLICY}" in
  gr00t)
    MAIN="${SCRIPT_DIR}/../gr00t/groot_conflict_eval.py"
    : "${PORT:=5555}" "${SIM_BACKEND:=gpu}" "${MAX_EPISODE_STEPS:=300}" "${REPLAN_STEPS:=5}" ;;
  pi0)
    MAIN="${SCRIPT_DIR}/../pi0/pi0_conflict_eval.py"
    : "${PORT:=8000}" "${SIM_BACKEND:=cpu}" "${MAX_EPISODE_STEPS:=150}" "${REPLAN_STEPS:=10}" ;;
  *) echo "policy must be gr00t or pi0, got ${POLICY}"; exit 1 ;;
esac
case "${EXPERIMENT}" in
  verb_color|verb_object|color_object|verb_size|verb_spatial|\
  size_object|color_size|color_spatial|spatial_size|spatial_object) ;;
  *) echo "Unsupported experiment: ${EXPERIMENT}"; exit 1 ;;
esac
if ! [[ "${TOTAL_EPISODES_TARGET}" =~ ^[0-9]+$ ]] || [[ "${TOTAL_EPISODES_TARGET}" -le 0 ]]; then
  echo "total_episodes must be a positive integer"; exit 1
fi

HOST="${HOST:-127.0.0.1}"
SEED_BASE="${SEED_BASE:-0}"
PYTHON="${PYTHON:-python}"
EXPERIMENT_ROOT="${OUT_DIR}/${EXPERIMENT}/experiments"
RESULTS_TXT_PATH="${OUT_DIR}/${EXPERIMENT}/${EXPERIMENT}_ood_seed${SEED}.txt"
mkdir -p "${EXPERIMENT_ROOT}"

# Generate the job list (pair sampling and per-job seeds)
JOBS_JSON="$(mktemp -t conflict_jobs.XXXXXX)"
trap 'rm -f "${JOBS_JSON}"' EXIT
"${PYTHON}" - "${EXPERIMENT}" "${SEED}" "${TOTAL_EPISODES_TARGET}" "${SEED_BASE}" "${THIRD_SEED}" <<'PY' > "${JOBS_JSON}"
import random, sys, json, math
experiment  = sys.argv[1]
seed        = int(sys.argv[2])
n_episodes  = int(sys.argv[3])
seed_base   = int(sys.argv[4])
third_seed  = int(sys.argv[5])
rng = random.Random(seed)
def _size_swap(n_size):
    p = []
    for a, b in ((0,1),(2,3),(4,5)):
        if a < n_size and b < n_size:
            p += [(a,b),(b,a)]
    return p
_SIZE_EXPS    = {"verb_size", "size_object", "color_size"}
_SPATIAL_EXPS = {"verb_spatial", "color_spatial", "spatial_size", "spatial_object"}
if experiment in _SIZE_EXPS:
    all_pairs = _size_swap(6)
elif experiment == "spatial_size":
    all_pairs = _size_swap(5)
elif experiment in _SPATIAL_EXPS:
    n = 5
    all_pairs = [(i,j) for i in range(n) for j in range(n) if i != j]
else:
    n = 6
    all_pairs = [(i,j) for i in range(n) for j in range(n) if i != j]
_run_types = {
    "verb_color":("verb","color"), "verb_object":("verb","shape"),
    "verb_size":("verb","size"),   "verb_spatial":("verb","spatial"),
    "color_object":("color","shape"), "size_object":("size","shape"),
    "color_size":("color","size"), "color_spatial":("color","spatial"),
    "spatial_size":("spatial","size"), "spatial_object":("spatial","shape"),
}
first, second = _run_types[experiment]
raw_jobs = []
for ep_idx in range(n_episodes):
    i, j = rng.choice(all_pairs)
    raw_jobs.append((i, j, first, ep_idx))
    raw_jobs.append((i, j, second, ep_idx))
total = len(raw_jobs)
num_episodes = math.ceil(n_episodes / total)
jobs = []
for k, (i, j, run_type, ep_idx) in enumerate(raw_jobs):
    idx = k + 1
    run_name = f"ood_{idx:03d}_{experiment}_{i}_{j}_{run_type}"
    if experiment == "verb_object":
        run_seed = seed_base + ep_idx
        ep_third_seed = ep_idx
    else:
        run_seed = seed_base + idx
        ep_third_seed = third_seed
    jobs.append({
        "index": idx, "pair_i": i, "pair_j": j, "run_type": run_type,
        "seed": run_seed, "third_seed": ep_third_seed,
        "num_episodes": num_episodes, "experiment_name": run_name,
    })
print(json.dumps(jobs))
PY

read -r TOTAL NUM_EPISODES < <("${PYTHON}" - "${JOBS_JSON}" <<'PY'
import json, sys
jobs = json.loads(open(sys.argv[1]).read())
print(len(jobs), jobs[0]["num_episodes"] if jobs else 1)
PY
)
echo "Requested ${TOTAL_EPISODES_TARGET} total episodes across ${TOTAL} runs -> ${NUM_EPISODES} eps/run (actual: $((TOTAL * NUM_EPISODES)))"

{
  echo "# OOD pairwise conflict summary (${POLICY})"
  echo "experiment=${EXPERIMENT}"
  echo "seed=${SEED}"
  echo "total_episodes_target=${TOTAL_EPISODES_TARGET}"
  echo "num_episodes_per_run=${NUM_EPISODES}"
  echo "total_runs=${TOTAL}"
  echo "total_episodes_actual=$((TOTAL * NUM_EPISODES))"
  echo "third_seed=${THIRD_SEED}"
  echo "policy_server=${HOST}:${PORT}"
  echo "sim_backend=${SIM_BACKEND}"
  echo "max_episode_steps=${MAX_EPISODE_STEPS}"
  echo "replan_steps=${REPLAN_STEPS}"
  echo "seed_base=${SEED_BASE}"
  echo
  echo "index pair_i pair_j run_type success run_name"
} > "${RESULTS_TXT_PATH}"

# ManiSkill C-extensions (fast_kinematics / mplib) link libtorch.so; make the
# client environment's torch libs discoverable.
_TORCH_LIB="$("${PYTHON}" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"

py_status=0
"${PYTHON}" "${MAIN}" \
  --experiment          "${EXPERIMENT}" \
  --host                "${HOST}" \
  --port                "${PORT}" \
  --replan-steps        "${REPLAN_STEPS}" \
  --max-episode-steps   "${MAX_EPISODE_STEPS}" \
  --sim-backend         "${SIM_BACKEND}" \
  --experiment-root     "${EXPERIMENT_ROOT}" \
  --batch-jobs-file     "${JOBS_JSON}" \
  --batch-results-txt   "${RESULTS_TXT_PATH}" || py_status=$?
if [[ "${py_status}" -ne 0 ]]; then
  echo "Batch eval failed (exit ${py_status}); partial results in ${RESULTS_TXT_PATH}"
  exit "${py_status}"
fi
echo "Saved summary to ${RESULTS_TXT_PATH}"
