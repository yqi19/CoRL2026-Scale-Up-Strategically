#!/usr/bin/env bash
# All-Factor (verb x color x shape x spatial x size): collect -> replay to RGB H5 -> LeRobot dataset.
# Same pipeline as the pairwise scripts (collect, select first NUM_DEMOS successful replays, convert).
# For large NUM_DEMOS use collect_convert_segmented.sh (restarts the collector every SEGMENT_SIZE demos).
#
# STRATEGY: random | stairvss | Lrandom   (see lib/all_factor_support_f50.py)
# FACTOR_COUNT: random -> 12 | 20 | 40; stairvss -> 20 | 40 | 50; Lrandom -> 12 | 50
#
# Output layout:
#   <PIPELINE_OUT>/<NUM_DEMOS>/original_data/all_factor/<strategy>_f<FACTOR_COUNT>/
#   <PIPELINE_OUT>/<NUM_DEMOS>/replay_data/all_factor/<strategy>_f<FACTOR_COUNT>/
#   <PIPELINE_OUT>/<NUM_DEMOS>/huggingface_data/all_factor/<strategy>_f<FACTOR_COUNT>/
#
# Usage (run from data_collection/):
#   PIPELINE_OUT=./outputs STRATEGY=random FACTOR_COUNT=12 NUM_DEMOS=200 \
#     ./collection_strategy/all_factor/collect_convert.sh
#   PIPELINE_OUT=./outputs STRATEGY=stairvss FACTOR_COUNT=50 NUM_DEMOS=800 \
#     ./collection_strategy/all_factor/collect_convert.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

COLLECTION_EXPERIMENT=all_factor

if [[ -n "${1:-}" && "${1}" != -* ]]; then
  PIPELINE_OUT="${1}"
  shift
fi
if [[ -z "${PIPELINE_OUT:-}" ]]; then
  echo "Error: PIPELINE_OUT is required (env var or first CLI argument)" >&2
  exit 1
fi
mkdir -p "${PIPELINE_OUT}"
PIPELINE_OUT=$(cd "${PIPELINE_OUT}" && pwd)

# =============================================================================
# Parameters (override with environment variables)
# =============================================================================
: "${STRATEGY:=random}"
: "${FACTOR_COUNT:=12}"
: "${NUM_DEMOS:=200}"
: "${SEED:=0}"
: "${CLEAN_LEROBOT:=1}"
: "${CLEAN_RECORD:=1}"
: "${SAVE_DUAL_MP4:=0}"
: "${MP_SAVE_VIDEO:=0}"
: "${DUAL_MP4_FPS:=30}"
: "${REPLAY_SAVE_HUMAN_VIDEO:=0}"
: "${SKIP_COLLECT:=0}"
: "${SKIP_REPLAY:=0}"
: "${SKIP_CONVERT:=0}"
# Python interpreters (default: whatever `python` is on PATH, i.e. the active environment).
: "${PYTHON:=python}"
: "${LEROBOT_PYTHON:=${PYTHON}}"
: "${SIM_BACKEND:=cpu}"
: "${RENDER_BACKEND:=cpu}"
: "${MP_OVERSAMPLE:=1.15}"
: "${MP_ONLY_COUNT_SUCCESS:=1}"
: "${MP_MAX_ATTEMPTS:=3}"
: "${PRUNE_UNSELECTED_RGB:=0}"
: "${LEROBOT_MAX_EPISODES:=}"
: "${NUM_DISTRACTORS:=2}"
: "${DISTRACTOR_MAX:=3}"
: "${SIZE_POOL:=small,large,smaller,larger,smallest,largest}"
: "${SMALL_LARGE_DISTRACTOR_POLICY:=random_uniform}"

case "${STRATEGY}" in
  random|stairvss|Lrandom) ;;
  *)
    echo "Error: STRATEGY must be one of random, stairvss, Lrandom (got ${STRATEGY})" >&2
    exit 1
    ;;
esac

MP_RUNS=$(NUM_DEMOS="${NUM_DEMOS}" MP_OVERSAMPLE="${MP_OVERSAMPLE}" python3 - <<'PY'
import math, os
n = int(os.environ["NUM_DEMOS"])
o = float(os.environ["MP_OVERSAMPLE"])
print(max(n, math.ceil(n * o)))
PY
)

if [[ "${SKIP_COLLECT}" == "1" ]]; then
  CLEAN_RECORD=0
fi

# -----------------------------------------------------------------------------
# Directory layout
# -----------------------------------------------------------------------------
RUN_ROOT="${PIPELINE_OUT}/${NUM_DEMOS}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT=$(cd "${RUN_ROOT}" && pwd)

SEGMENT="${COLLECTION_EXPERIMENT}/${STRATEGY}_f${FACTOR_COUNT}"
ORIGINAL_DATA_ROOT="${RUN_ROOT}/original_data/${SEGMENT}"
REPLAY_DATA_ROOT="${RUN_ROOT}/replay_data/${SEGMENT}"
HUGGINGFACE_DATA_ROOT="${RUN_ROOT}/huggingface_data/${SEGMENT}"
TASK_MAP_JSON="${REPLAY_DATA_ROOT}/task_map.json"
SELECTED_STEMS_FILE="${REPLAY_DATA_ROOT}/selected_rgb_stems.txt"
REPO_ID="${REPO_ID:-maniskill_${COLLECTION_EXPERIMENT}_${STRATEGY}_f${FACTOR_COUNT}_n${NUM_DEMOS}}"

echo "=== All-factor pipeline ==="
echo "  STRATEGY=${STRATEGY}  FACTOR_COUNT=${FACTOR_COUNT}  NUM_DEMOS=${NUM_DEMOS}  MP_RUNS=${MP_RUNS}"
echo "  original_data  -> ${ORIGINAL_DATA_ROOT}"
echo "  replay_data    -> ${REPLAY_DATA_ROOT}"
echo "  huggingface    -> ${HUGGINGFACE_DATA_ROOT}"

if [[ "${CLEAN_LEROBOT}" == "1" ]]; then
  rm -rf "${HUGGINGFACE_DATA_ROOT}"
  rm -f "${TASK_MAP_JSON}" "${SELECTED_STEMS_FILE}"
fi
if [[ "${CLEAN_RECORD}" == "1" ]]; then
  rm -rf "${ORIGINAL_DATA_ROOT}" "${REPLAY_DATA_ROOT}"
fi

# -----------------------------------------------------------------------------
# Step 1: motion-planning collection
# -----------------------------------------------------------------------------
if [[ "${SKIP_COLLECT}" != "1" ]]; then
  echo "=== Step 1: collect -> ${ORIGINAL_DATA_ROOT} ==="
  MP_OCS_ARGS=()
  if [[ "${MP_ONLY_COUNT_SUCCESS}" == "1" ]]; then
    MP_OCS_ARGS=(--only-count-success)
  else
    MP_OCS_ARGS=(--no-only-count-success)
  fi
  MP_VID_ARGS=()
  if [[ "${MP_SAVE_VIDEO}" == "1" ]]; then
    MP_VID_ARGS=(--save-video)
  fi
  PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/collection_strategy/all_factor/collect.py" \
    --strategy "${STRATEGY}" \
    --factor-count "${FACTOR_COUNT}" \
    --record-segment "${STRATEGY}_f${FACTOR_COUNT}" \
    --record-base "${RUN_ROOT}/original_data" \
    --num-demos "${MP_RUNS}" \
    --seed "${SEED}" \
    --sim-backend "${SIM_BACKEND}" \
    --render-backend "${RENDER_BACKEND}" \
    --num-distractors "${NUM_DISTRACTORS}" \
    --distractor-max "${DISTRACTOR_MAX}" \
    --size-pool "${SIZE_POOL}" \
    --small-large-distractor-policy "${SMALL_LARGE_DISTRACTOR_POLICY}" \
    --max-attempts "${MP_MAX_ATTEMPTS}" \
    "${MP_OCS_ARGS[@]}" \
    "${MP_VID_ARGS[@]}"
else
  echo "=== Step 1: skipped ==="
fi

N_RAW_MP=$(find "${ORIGINAL_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' 2>/dev/null | wc -l)
if [[ "${N_RAW_MP}" -eq 0 ]]; then
  echo "No raw MP .h5 under ${ORIGINAL_DATA_ROOT}" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Replay -> *_rgb.h5
# -----------------------------------------------------------------------------
if [[ "${SKIP_REPLAY}" != "1" ]]; then
  echo "=== Step 2a: copy raw h5 -> replay_data ==="
  while IFS= read -r -d '' h5; do
    base=$(basename "${h5}" .h5)
    rel="${h5#"${ORIGINAL_DATA_ROOT}/"}"
    dest_dir="${REPLAY_DATA_ROOT}/$(dirname "${rel}")"
    mkdir -p "${dest_dir}"
    cp -f "${h5}" "${dest_dir}/${base}.h5"
    json_src="$(dirname "${h5}")/${base}.json"
    if [[ -f "${json_src}" ]]; then
      cp -f "${json_src}" "${dest_dir}/${base}.json"
    fi
  done < <(find "${ORIGINAL_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' -print0 2>/dev/null)

  echo "=== Step 2b: replay -> first ${NUM_DEMOS} successful *_rgb.h5 ==="
  mkdir -p "${REPLAY_DATA_ROOT}"
  : > "${SELECTED_STEMS_FILE}"
  REPLAY_EXTRA=(--allow-failure)
  if [[ "${REPLAY_SAVE_HUMAN_VIDEO}" == "1" ]]; then
    REPLAY_EXTRA+=(--save-video)
  fi
  SUCCESS_COUNT=0
  while IFS= read -r -d '' h5; do
    if [[ "${SUCCESS_COUNT}" -ge "${NUM_DEMOS}" ]]; then
      break
    fi
    base=$(basename "${h5}" .h5)
    d=$(dirname "${h5}")
    out_rgb="${d}/${base}_rgb.h5"
    if [[ -f "${out_rgb}" ]]; then
      echo "$(basename "${out_rgb}" .h5)" >> "${SELECTED_STEMS_FILE}"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      continue
    fi
    set +e
    PYTHONPATH="${REPO_ROOT}" "${PYTHON}" -m mani_skill.trajectory.replay_trajectory \
      --traj-path "${h5}" \
      --save-traj \
      --save-traj-name "${base}_rgb" \
      --obs-mode rgb \
      "${REPLAY_EXTRA[@]}"
    set -e
    if [[ -f "${out_rgb}" ]]; then
      echo "$(basename "${out_rgb}" .h5)" >> "${SELECTED_STEMS_FILE}"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
  done < <(find "${REPLAY_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' -print0 2>/dev/null | LC_ALL=C sort -z)

  if [[ "${SUCCESS_COUNT}" -lt "${NUM_DEMOS}" ]]; then
    echo "Error: need ${NUM_DEMOS} *_rgb.h5, got ${SUCCESS_COUNT}" >&2
    exit 1
  fi

  if [[ "${PRUNE_UNSELECTED_RGB}" == "1" ]]; then
    while IFS= read -r -d '' rgb; do
      stem=$(basename "${rgb}" .h5)
      if ! grep -qxF "${stem}" "${SELECTED_STEMS_FILE}"; then
        rm -f "${rgb}"
      fi
    done < <(find "${REPLAY_DATA_ROOT}" -type f -name '*_rgb.h5' -print0 2>/dev/null)
  fi
else
  echo "=== Step 2: skipped ==="
fi

mkdir -p "${REPLAY_DATA_ROOT}"

if [[ "${SAVE_DUAL_MP4}" == "1" ]]; then
  DUAL_ARGS=(--motion-dir "${REPLAY_DATA_ROOT}" --fps "${DUAL_MP4_FPS}")
  if [[ -f "${SELECTED_STEMS_FILE}" ]] && [[ -s "${SELECTED_STEMS_FILE}" ]]; then
    DUAL_ARGS+=(--stems-file "${SELECTED_STEMS_FILE}")
  fi
  "${PYTHON}" "${REPO_ROOT}/collection_strategy/rgb_h5_to_dual_camera_mp4.py" "${DUAL_ARGS[@]}"
fi

# -----------------------------------------------------------------------------
# Step 3: task_map + LeRobot conversion
# -----------------------------------------------------------------------------
if [[ "${SKIP_CONVERT}" != "1" ]]; then
  echo "=== Step 3a: build_attribute_task_map -> ${TASK_MAP_JSON} ==="
  mkdir -p "${REPLAY_DATA_ROOT}"
  MAP_EXTRA=()
  if [[ -f "${SELECTED_STEMS_FILE}" ]] && [[ -s "${SELECTED_STEMS_FILE}" ]]; then
    MAP_EXTRA=(--stems-file "${SELECTED_STEMS_FILE}")
  fi
  PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/collection_strategy/build_attribute_task_map.py" \
    --experiment "${COLLECTION_EXPERIMENT}" \
    --demos-dir "${REPLAY_DATA_ROOT}" \
    --output "${TASK_MAP_JSON}" \
    "${MAP_EXTRA[@]}"

  echo "=== Step 3b: convert to LeRobot -> ${HUGGINGFACE_DATA_ROOT} ==="
  mkdir -p "${HUGGINGFACE_DATA_ROOT}"
  MAX_ARGS=()
  if [[ -n "${LEROBOT_MAX_EPISODES:-}" ]]; then
    MAX_ARGS=(--max-episodes "${LEROBOT_MAX_EPISODES}")
  fi
  "${LEROBOT_PYTHON}" "${REPO_ROOT}/collection_strategy/convert_maniskill_to_lerobot.py" \
    --demos-dir "${REPLAY_DATA_ROOT}" \
    --task-map-file "${TASK_MAP_JSON}" \
    --repo-id "${REPO_ID}" \
    --output-dir "${HUGGINGFACE_DATA_ROOT}" \
    "${MAX_ARGS[@]}"
else
  echo "=== Step 3: skipped ==="
fi

echo "Done."
echo "  original_data    -> ${ORIGINAL_DATA_ROOT}"
echo "  replay_data      -> ${REPLAY_DATA_ROOT}"
echo "  huggingface_data -> ${HUGGINGFACE_DATA_ROOT}"
