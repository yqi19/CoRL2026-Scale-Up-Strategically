#!/usr/bin/env bash
set -uo pipefail
# run_all_factor.sh - GR00T All-Factor eval (all 5 factors in one instruction,
# "<Verb> the <size> <color> <object> <spatial>."), several seeds, against a
# running GR00T server. Task sampling, prompts, distractors and success match
# the pi0.5 full-factor protocol.
#
# Usage: bash gr00t/run_all_factor.sh <out_dir> [seeds="40 41 42"]
# Output: <out_dir>/full_factor_seed<seed>.txt, <out_dir>/videos_seed<seed>/, SUMMARY.txt
#
# Env (defaults = paper protocol):
#   HOST 127.0.0.1  PORT 5555  SAMPLE_N 200 (cells sampled from 6x6x6x5x4)
#   SAMPLE_SEED 42  TOTAL_EPISODES 200  MAX_EPISODE_STEPS 500
#   NO_DISTRACTOR_PROB 0.70  REPLAN_STEPS 5  SIM_BACKEND gpu  RENDER_BACKEND gpu
#   PYTHON python
OUT_DIR="${1:?out_dir}"
read -r -a SEEDS <<< "${2:-40 41 42}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
SAMPLE_N="${SAMPLE_N:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
TOTAL_EPISODES="${TOTAL_EPISODES:-200}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
NO_DISTRACTOR_PROB="${NO_DISTRACTOR_PROB:-0.70}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
SIM_BACKEND="${SIM_BACKEND:-gpu}"
RENDER_BACKEND="${RENDER_BACKEND:-gpu}"
PYTHON="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUT_DIR}"
_TORCH_LIB="$("${PYTHON}" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"

rc_all=0
for sb in "${SEEDS[@]}"; do
  rt="${OUT_DIR}/full_factor_seed${sb}.txt"
  echo "[$(date +%H:%M:%S)] seed_base ${sb} -> ${rt}"
  "${PYTHON}" "${HERE}/groot_full_factor_batch.py" \
    --host "${HOST}" --port "${PORT}" \
    --results-txt "${rt}" \
    --video-root "${OUT_DIR}/videos_seed${sb}" \
    --sample-n "${SAMPLE_N}" --sample-seed "${SAMPLE_SEED}" --seed-base "${sb}" \
    --total-episodes "${TOTAL_EPISODES}" --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --no-distractor-prob "${NO_DISTRACTOR_PROB}" --replan-steps "${REPLAN_STEPS}" \
    --sim-backend "${SIM_BACKEND}" --render-backend "${RENDER_BACKEND}" || rc_all=1
done

"${PYTHON}" "${HERE}/../common/summarize_seeds.py" "${OUT_DIR}" "full_factor_seed*.txt" \
  | tee "${OUT_DIR}/SUMMARY.txt"
exit ${rc_all}
