#!/usr/bin/env bash
set -uo pipefail
# run_pairwise_grid.sh - GR00T compositional-generalization eval on one
# pairwise grid (HARD protocol), several seeds, against a running GR00T server.
#
# Usage: bash gr00t/run_pairwise_grid.sh <experiment> <out_dir> [seeds="42 40 41"]
#   experiment: color_size | color_spatial | verb_spatial | spatial_size |
#               spatial_object | verb_size
# Output: <out_dir>/gr00t_<experiment>_seed<seed>.txt (+ SUMMARY.txt, 3-seed mean)
#
# Env (defaults = paper protocol):
#   HOST 127.0.0.1  PORT 5555  SIM_BACKEND gpu  TASK_DIFFICULTY 1.5
#   MAX_EPISODE_STEPS 150  REPLAN_STEPS 10  TARGET_EPISODES 200
#   COLOR_SIZE_MODE hard (easy = 1 distractor, ~2x size contrast)
#   MAX_CELLS (debug: only the first N cells)   PYTHON python
EXPERIMENT="${1:?experiment}"
OUT_DIR="${2:?out_dir}"
read -r -a SEEDS <<< "${3:-42 40 41}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
SIM_BACKEND="${SIM_BACKEND:-gpu}"
TASK_DIFFICULTY="${TASK_DIFFICULTY:-1.5}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-150}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
TARGET_EPISODES="${TARGET_EPISODES:-200}"
COLOR_SIZE_MODE="${COLOR_SIZE_MODE:-hard}"
PYTHON="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUT_DIR}"
_TORCH_LIB="$("${PYTHON}" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"

rc_all=0
for sb in "${SEEDS[@]}"; do
  rt="${OUT_DIR}/gr00t_${EXPERIMENT}_seed${sb}.txt"
  echo "[$(date +%H:%M:%S)] seed ${sb} ${EXPERIMENT} -> ${rt}"
  "${PYTHON}" "${HERE}/groot_grid_eval.py" \
    --experiment "${EXPERIMENT}" --host "${HOST}" --port "${PORT}" \
    --seed "${sb}" --sim-backend "${SIM_BACKEND}" \
    --task-difficulty "${TASK_DIFFICULTY}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" --replan-steps "${REPLAN_STEPS}" \
    --target-episodes "${TARGET_EPISODES}" \
    --color-size-mode "${COLOR_SIZE_MODE}" \
    ${MAX_CELLS:+--max-cells "${MAX_CELLS}"} \
    --results-txt "${rt}" || rc_all=1
done

"${PYTHON}" "${HERE}/../common/summarize_seeds.py" "${OUT_DIR}" "gr00t_${EXPERIMENT}_seed*.txt" \
  | tee "${OUT_DIR}/SUMMARY.txt"
exit ${rc_all}
