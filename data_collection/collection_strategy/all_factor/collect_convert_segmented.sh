#!/usr/bin/env bash
# All-Factor segmented collection, then a single replay + LeRobot conversion.
#
# Why: after roughly 870 episodes in one process the pybind11 GIL state got corrupted and every
# later motion-planning call failed. This script splits collection into separate subprocesses,
# each adding at most SEGMENT_SIZE demos (the collector resumes from the existing files).
# Replay and LeRobot conversion run once at the end via collect_convert.sh (SKIP_COLLECT=1).
#
# Usage (same environment variables as collect_convert.sh, run from data_collection/):
#   PIPELINE_OUT=./outputs/all_factor_random_n800 \
#   STRATEGY=random FACTOR_COUNT=12 NUM_DEMOS=800 \
#   SEGMENT_SIZE=400 \
#   ./collection_strategy/all_factor/collect_convert_segmented.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

# =============================================================================
# Parameters (same as collect_convert.sh)
# =============================================================================
: "${STRATEGY:=random}"
: "${FACTOR_COUNT:=12}"
: "${NUM_DEMOS:=800}"
: "${SEED:=0}"
: "${SEGMENT_SIZE:=400}"           # max new demos per subprocess
# Python interpreters (default: whatever `python` is on PATH, i.e. the active environment).
: "${PYTHON:=python}"
: "${LEROBOT_PYTHON:=${PYTHON}}"
: "${SIM_BACKEND:=cpu}"
: "${RENDER_BACKEND:=cpu}"
: "${MP_OVERSAMPLE:=1.15}"
: "${MP_MAX_ATTEMPTS:=3}"
: "${NUM_DISTRACTORS:=2}"
: "${DISTRACTOR_MAX:=3}"
: "${SIZE_POOL:=small,large,smaller,larger,smallest,largest}"
: "${SMALL_LARGE_DISTRACTOR_POLICY:=random_uniform}"
: "${CLEAN_LEROBOT:=1}"
: "${REPLAY_SAVE_HUMAN_VIDEO:=0}"
: "${PRUNE_UNSELECTED_RGB:=0}"
: "${LEROBOT_MAX_EPISODES:=}"

if [[ -n "${1:-}" && "${1}" != -* ]]; then
  PIPELINE_OUT="${1}"; shift
fi
if [[ -z "${PIPELINE_OUT:-}" ]]; then
  echo "Error: PIPELINE_OUT is required" >&2; exit 1
fi
mkdir -p "${PIPELINE_OUT}"
PIPELINE_OUT=$(cd "${PIPELINE_OUT}" && pwd)

# Oversampled total target (same formula as collect_convert.sh)
MP_RUNS=$(python3 - <<PY
import math, os
n = int(os.environ.get("NUM_DEMOS", "${NUM_DEMOS}"))
o = float(os.environ.get("MP_OVERSAMPLE", "${MP_OVERSAMPLE}"))
print(max(n, math.ceil(n * o)))
PY
)

RECORD_BASE="${PIPELINE_OUT}/${NUM_DEMOS}/original_data"
RECORD_DIR="${RECORD_BASE}/all_factor/${STRATEGY}_f${FACTOR_COUNT}"

echo "=== Segmented all-factor pipeline ==="
echo "  STRATEGY=${STRATEGY}  FACTOR_COUNT=${FACTOR_COUNT}  NUM_DEMOS=${NUM_DEMOS}"
echo "  MP_RUNS(target)=${MP_RUNS}  SEGMENT_SIZE=${SEGMENT_SIZE}"
echo "  original_data -> ${RECORD_DIR}"

# =============================================================================
# Phase 1: segmented collection
# =============================================================================
_count_existing() {
  find "${RECORD_DIR}" -name '*.h5' ! -name '*_rgb.h5' 2>/dev/null | wc -l
}

mkdir -p "${RECORD_DIR}"

segment_target=${SEGMENT_SIZE}
segment_idx=0

while true; do
  n_existing=$(_count_existing)
  echo "--- [segment ${segment_idx}] existing=${n_existing} / need=${MP_RUNS} ---"

  if [[ "${n_existing}" -ge "${MP_RUNS}" ]]; then
    echo "Collection done: ${n_existing} >= ${MP_RUNS}"
    break
  fi

  # Target of this segment: advance by one SEGMENT_SIZE
  segment_target=$(( (segment_idx + 1) * SEGMENT_SIZE ))
  if [[ "${segment_target}" -gt "${MP_RUNS}" ]]; then
    segment_target="${MP_RUNS}"
  fi

  # Already past this segment target (resumed run): go to the next segment
  if [[ "${n_existing}" -ge "${segment_target}" ]]; then
    segment_idx=$(( segment_idx + 1 ))
    continue
  fi

  echo "  segment target: ${segment_target} (about $(( segment_target - n_existing )) new)"

  # Fresh subprocess
  set +e
  PYTHONPATH="${REPO_ROOT}" "${PYTHON}" -m collection_strategy.collect_all_factor \
    --strategy              "${STRATEGY}" \
    --factor-count          "${FACTOR_COUNT}" \
    --record-segment        "${STRATEGY}_f${FACTOR_COUNT}" \
    --record-base           "${RECORD_BASE}" \
    --num-demos             "${segment_target}" \
    --seed                  "${SEED}" \
    --sim-backend           "${SIM_BACKEND}" \
    --render-backend        "${RENDER_BACKEND}" \
    --num-distractors       "${NUM_DISTRACTORS}" \
    --distractor-max        "${DISTRACTOR_MAX}" \
    --size-pool             "${SIZE_POOL}" \
    --small-large-distractor-policy "${SMALL_LARGE_DISTRACTOR_POLICY}" \
    --max-attempts          "${MP_MAX_ATTEMPTS}" \
    --only-count-success
  rc=$?
  set -e

  n_after=$(_count_existing)
  echo "  subprocess exit code=${rc}, demos after segment=${n_after}"

  if [[ "${n_after}" -le "${n_existing}" ]] && [[ "${n_after}" -lt "${MP_RUNS}" ]]; then
    echo "Error: no progress in this segment (${n_existing} -> ${n_after}); stopping." >&2
    exit 1
  fi

  segment_idx=$(( segment_idx + 1 ))
done

# =============================================================================
# Phase 2+3: replay -> LeRobot conversion (once, for all segments)
# =============================================================================
echo "=== Phase 2+3: replay + LeRobot conversion ==="
env PIPELINE_OUT="${PIPELINE_OUT}" \
    STRATEGY="${STRATEGY}" \
    FACTOR_COUNT="${FACTOR_COUNT}" \
    NUM_DEMOS="${NUM_DEMOS}" \
    SEED="${SEED}" \
    SKIP_COLLECT=1 \
    CLEAN_RECORD=0 \
    CLEAN_LEROBOT="${CLEAN_LEROBOT}" \
    PYTHON="${PYTHON}" \
    LEROBOT_PYTHON="${LEROBOT_PYTHON}" \
    SIM_BACKEND="${SIM_BACKEND}" \
    RENDER_BACKEND="${RENDER_BACKEND}" \
    REPLAY_SAVE_HUMAN_VIDEO="${REPLAY_SAVE_HUMAN_VIDEO}" \
    PRUNE_UNSELECTED_RGB="${PRUNE_UNSELECTED_RGB}" \
    ${LEROBOT_MAX_EPISODES:+LEROBOT_MAX_EPISODES="${LEROBOT_MAX_EPISODES}"} \
    "${SCRIPT_DIR}/collect_convert.sh"
