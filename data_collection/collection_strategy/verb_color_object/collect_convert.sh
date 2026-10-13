#!/usr/bin/env bash
# Three-factor Verb x Color x Object: collect (motion planning) -> replay to RGB H5 -> LeRobot dataset.
#
# Output layout (PIPELINE_OUT is required, via env var or first CLI arg):
#   <PIPELINE_OUT>/<NUM_DEMOS>/original_data/verb_color_object/<STRATEGY>_f<FACTOR_COUNT>/     raw motion-planning trajectories
#   <PIPELINE_OUT>/<NUM_DEMOS>/replay_data/verb_color_object/<STRATEGY>_f<FACTOR_COUNT>/       *_rgb.h5, task_map.json, selected_rgb_stems.txt
#   <PIPELINE_OUT>/<NUM_DEMOS>/huggingface_data/verb_color_object/<STRATEGY>_f<FACTOR_COUNT>/  LeRobot dataset
#
# Example (run from data_collection/):
#   PIPELINE_OUT=./outputs STRATEGY=stair FACTOR_COUNT=20 NUM_DEMOS=300 \
#     ./collection_strategy/verb_color_object/collect_convert.sh
#
# Main environment variables (defaults below):
#   STRATEGY             random | stair | Lrandom  (see collection_strategy/README.md)
#   FACTOR_COUNT         15 | 20 | 30 (verb x color x object cube)
#   NUM_DEMOS            number of successful replays that go into the LeRobot dataset
#   MP_OVERSAMPLE        Step 1 collects MP_RUNS = max(NUM_DEMOS, ceil(NUM_DEMOS * MP_OVERSAMPLE)) raw demos
#   MP_ONLY_COUNT_SUCCESS 1: retry motion planning until success (up to MP_MAX_ATTEMPTS, 0 = unlimited)
#   SEED                 seed for support sampling, per-pass shuffles and third-factor sampling
#   NUM_DISTRACTORS / DISTRACTOR_MAX   distractor objects per episode
#   CLEAN_LEROBOT        1: delete this segment's LeRobot output, task_map.json and selected_rgb_stems.txt first
#   CLEAN_RECORD         1: delete this segment's original_data and replay_data first (forced to 0 if SKIP_COLLECT=1)
#   PRUNE_UNSELECTED_RGB 1: delete *_rgb.h5 not listed in selected_rgb_stems.txt
#   MP_SAVE_VIDEO / REPLAY_SAVE_HUMAN_VIDEO / SAVE_DUAL_MP4   optional videos
#   SKIP_COLLECT / SKIP_REPLAY / SKIP_CONVERT                 skip pipeline steps
#   SIM_BACKEND / RENDER_BACKEND (cpu by default), OBJ_XY_RANGE
#   PYTHON / LEROBOT_PYTHON   interpreters for ManiSkill steps and LeRobot conversion
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

COLLECTION_EXPERIMENT=verb_color_object

# The first CLI argument may be used as PIPELINE_OUT.
if [[ -n "${1:-}" && "${1}" != -* ]]; then
  PIPELINE_OUT="${1}"
  shift
fi
if [[ -z "${PIPELINE_OUT:-}" ]]; then
  echo "Error: PIPELINE_OUT is required; data goes to PIPELINE_OUT/<NUM_DEMOS>/{original_data,replay_data,huggingface_data}" >&2
  echo "  usage: PIPELINE_OUT=/path/to/base $0" >&2
  echo "     or: $0 /path/to/base" >&2
  exit 1
fi
command mkdir -p "${PIPELINE_OUT}"
PIPELINE_OUT=$(cd "${PIPELINE_OUT}" && pwd)

# Python interpreters (default: whatever `python` is on PATH, i.e. the active environment).
: "${PYTHON:=python}"
: "${LEROBOT_PYTHON:=${PYTHON}}"

: "${STRATEGY:=random}"
: "${FACTOR_COUNT:=15}"
: "${NUM_DEMOS:=36}"
: "${SEED:=0}"
: "${CLEAN_RUN:=0}"
: "${CLEAN_LEROBOT:=0}"
: "${CLEAN_RECORD:=1}"
: "${SAVE_DUAL_MP4:=0}"
: "${MP_SAVE_VIDEO:=1}"
: "${DUAL_MP4_FPS:=30}"
: "${REPLAY_SAVE_HUMAN_VIDEO:=1}"
: "${SKIP_COLLECT:=0}"
: "${SKIP_REPLAY:=0}"
: "${SKIP_CONVERT:=0}"
: "${REPLAY_ALLOW_FAILURE:=0}"
: "${SIM_BACKEND:=cpu}"
: "${RENDER_BACKEND:=cpu}"
# Vulkan / SAPIEN: with SIM/RENDER=cpu, Lavapipe is used by default (see lib/vulkan_headless_default.sh).
# To use NVIDIA Vulkan set MANISKILL_USE_GPU=1 or MANISKILL_USE_SYSTEM_VULKAN=1, or set MS_VULKAN_ICD_FILE.
# On NVIDIA machines with nvidia-smi but broken Vulkan: install user-space GL matching the driver, e.g.
#   apt-get install -y "libnvidia-gl-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)"
# then check: vulkaninfo --summary | grep -i nvidia
if [[ -n "${MS_VULKAN_ICD_FILE:-}" ]]; then
  export VK_ICD_FILENAMES="${MS_VULKAN_ICD_FILE}"
fi
: "${OBJ_XY_RANGE:=-0.1 0.1 -0.1 0.1}"
: "${MP_OVERSAMPLE:=1.15}"
: "${PRUNE_UNSELECTED_RGB:=0}"
: "${MP_ONLY_COUNT_SUCCESS:=1}"
: "${MP_MAX_ATTEMPTS:=3}"
: "${NUM_DISTRACTORS:=2}"
: "${DISTRACTOR_MAX:=2}"
: "${CONVERT_KEEP_OUTPUT:=0}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/collection_strategy/lib/vulkan_headless_default.sh"
vulkan_headless_default

case "${FACTOR_COUNT}" in
  15|20|30) ;;
  *)
    echo "Error: FACTOR_COUNT must be 15, 20 or 30 (got ${FACTOR_COUNT})" >&2
    exit 1
    ;;
esac
case "${STRATEGY}" in
  random|Lrandom|stair) ;;
  *)
    echo "Error: STRATEGY must be random, Lrandom or stair (got ${STRATEGY})" >&2
    exit 1
    ;;
esac

MP_RUNS=$(NUM_DEMOS="${NUM_DEMOS}" MP_OVERSAMPLE="${MP_OVERSAMPLE}" python3 - <<'PY'
import math
import os
n = int(os.environ["NUM_DEMOS"])
o = float(os.environ["MP_OVERSAMPLE"])
print(max(n, math.ceil(n * o)))
PY
)

if [[ "${SKIP_COLLECT}" == "1" ]]; then
  CLEAN_RECORD=0
fi

# -----------------------------------------------------------------------------
# Directory layout (one segment per experiment/strategy/factor count, so runs do not overwrite each other):
#   PIPELINE_OUT/<NUM_DEMOS>/{original_data,replay_data,huggingface_data}/<experiment>/<strategy>/
# -----------------------------------------------------------------------------
RUN_ROOT="${PIPELINE_OUT}/${NUM_DEMOS}"
command mkdir -p "${RUN_ROOT}"
RUN_ROOT=$(cd "${RUN_ROOT}" && pwd)

RECORD_BASE_ORIGINAL="${RUN_ROOT}/original_data"
SEGMENT="${COLLECTION_EXPERIMENT}/${STRATEGY}_f${FACTOR_COUNT}"
ORIGINAL_DATA_ROOT="${RECORD_BASE_ORIGINAL}/${SEGMENT}"
REPLAY_DATA_ROOT="${RUN_ROOT}/replay_data/${SEGMENT}"
HUGGINGFACE_DATA_ROOT="${RUN_ROOT}/huggingface_data/${SEGMENT}"

RUN_STAMP="${COLLECTION_EXPERIMENT}_${STRATEGY}_f${FACTOR_COUNT}"
REPO_ID="${REPO_ID:-maniskill_${RUN_STAMP}_n${NUM_DEMOS}}"
TASK_MAP_JSON="${REPLAY_DATA_ROOT}/task_map.json"
SELECTED_STEMS_FILE="${REPLAY_DATA_ROOT}/selected_rgb_stems.txt"
LEROBOT_MAX_EPISODES="${LEROBOT_MAX_EPISODES:-}"

echo "=== Triple (verb×color×object) pipeline (${COLLECTION_EXPERIMENT}) ==="
echo "  PIPELINE_OUT(base)=${PIPELINE_OUT}"
echo "  RUN_ROOT=${RUN_ROOT}  (subdirectory name = NUM_DEMOS=${NUM_DEMOS})"
echo "  PYTHON=${PYTHON}  LEROBOT_PYTHON=${LEROBOT_PYTHON}"
echo "  target HF episodes=${NUM_DEMOS}  FACTOR_COUNT=${FACTOR_COUNT}  MP collect runs=${MP_RUNS}  (MP_OVERSAMPLE=${MP_OVERSAMPLE})  MP_ONLY_COUNT_SUCCESS=${MP_ONLY_COUNT_SUCCESS}  MP_MAX_ATTEMPTS=${MP_MAX_ATTEMPTS}  NUM_DISTRACTORS=${NUM_DISTRACTORS}  MP_SAVE_VIDEO=${MP_SAVE_VIDEO}  SAVE_DUAL_MP4=${SAVE_DUAL_MP4}  REPLAY_SAVE_HUMAN_VIDEO=${REPLAY_SAVE_HUMAN_VIDEO}  CLEAN_LEROBOT=${CLEAN_LEROBOT}  CONVERT_KEEP_OUTPUT=${CONVERT_KEEP_OUTPUT}"
echo "  strategy=${STRATEGY}"
echo "  original_data -> ${ORIGINAL_DATA_ROOT}"
echo "  replay_data   -> ${REPLAY_DATA_ROOT}"
echo "  huggingface_data -> ${HUGGINGFACE_DATA_ROOT}"

if [[ "${CLEAN_RUN}" == "1" ]]; then
  CLEAN_LEROBOT=1
fi
if [[ "${CLEAN_LEROBOT}" == "1" ]]; then
  echo "=== Cleaning huggingface_data + task_map + selected_rgb_stems ==="
  rm -rf "${HUGGINGFACE_DATA_ROOT}"
  rm -f "${TASK_MAP_JSON}" "${SELECTED_STEMS_FILE}"
fi
if [[ "${CLEAN_RECORD}" == "1" ]]; then
  echo "=== Cleaning this run under original_data + replay_data ==="
  rm -rf "${ORIGINAL_DATA_ROOT}" "${REPLAY_DATA_ROOT}"
fi

# -----------------------------------------------------------------------------
# Step 1: motion-planning collection -> original_data/<exp>/<segment>/.../motionplanning/*.h5
# -----------------------------------------------------------------------------
if [[ "${SKIP_COLLECT}" != "1" ]]; then
  echo "=== Step 1: collect_verb_color_object -> ${ORIGINAL_DATA_ROOT} ==="
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
  # shellcheck disable=SC2086
  RECORD_SEGMENT="${RECORD_SEGMENT:-${STRATEGY}_f${FACTOR_COUNT}}"
  "${PYTHON}" "${REPO_ROOT}/collection_strategy/${COLLECTION_EXPERIMENT}/collect.py" \
    --strategy "${STRATEGY}" \
    --record-segment "${RECORD_SEGMENT}" \
    --factor-count "${FACTOR_COUNT}" \
    --num-demos "${MP_RUNS}" \
    --seed "${SEED}" \
    --record-base "${RECORD_BASE_ORIGINAL}" \
    --sim-backend "${SIM_BACKEND}" \
    --render-backend "${RENDER_BACKEND}" \
    --obj-xy-range ${OBJ_XY_RANGE} \
    --max-attempts "${MP_MAX_ATTEMPTS}" \
    --num-distractors "${NUM_DISTRACTORS}" \
    --distractor-max "${DISTRACTOR_MAX}" \
    "${MP_OCS_ARGS[@]}" \
    "${MP_VID_ARGS[@]}"
else
  echo "=== Step 1: skipped (SKIP_COLLECT=1) ==="
fi

N_RAW_MP=$(command find "${ORIGINAL_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' 2>/dev/null | wc -l)
if [[ "${N_RAW_MP}" -eq 0 ]]; then
  echo "No raw MP .h5 under ${ORIGINAL_DATA_ROOT} (expected */motionplanning/*.h5)" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: copy every raw .h5 + .json (all env subdirectories) into replay_data with the same relative
# paths, then replay them one by one until NUM_DEMOS succeed.
# -----------------------------------------------------------------------------
if [[ "${SKIP_REPLAY}" != "1" ]]; then
  echo "=== Step 2a: copy raw h5 + json (all env */motionplanning) -> replay_data ==="
  while IFS= read -r -d '' h5; do
    base=$(basename "${h5}" .h5)
    if [[ "${base}" == *_rgb ]]; then
      continue
    fi
    rel="${h5#"${ORIGINAL_DATA_ROOT}/"}"
    dest_dir="${REPLAY_DATA_ROOT}/$(dirname "${rel}")"
    command mkdir -p "${dest_dir}"
    cp -f "${h5}" "${dest_dir}/${base}.h5"
    json_src="$(dirname "${h5}")/${base}.json"
    if [[ -f "${json_src}" ]]; then
      cp -f "${json_src}" "${dest_dir}/${base}.json"
    fi
  done < <(command find "${ORIGINAL_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' -print0 2>/dev/null)

  echo "=== Step 2b: replay -> first ${NUM_DEMOS} successful *_rgb.h5 (sorted raw paths) -> ${SELECTED_STEMS_FILE} ==="
  command mkdir -p "${REPLAY_DATA_ROOT}"
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
    if [[ "${base}" == *_rgb ]]; then
      continue
    fi
    d=$(dirname "${h5}")
    out_rgb="${d}/${base}_rgb.h5"
    if [[ -f "${out_rgb}" ]]; then
      echo "  count existing rgb: ${out_rgb}"
      echo "$(basename "${out_rgb}" .h5)" >> "${SELECTED_STEMS_FILE}"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      continue
    fi
    echo "  replay ${base}.h5"
    set +e
    "${PYTHON}" -m mani_skill.trajectory.replay_trajectory \
      --traj-path "${h5}" \
      --save-traj \
      --save-traj-name "${base}_rgb" \
      --obs-mode rgb \
      "${REPLAY_EXTRA[@]}"
    rp=$?
    set -e
    if [[ ! -f "${out_rgb}" ]]; then
      echo "  no rgb output (exit ${rp}): ${out_rgb}" >&2
      continue
    fi
    echo "$(basename "${out_rgb}" .h5)" >> "${SELECTED_STEMS_FILE}"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  done < <(command find "${REPLAY_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' -print0 2>/dev/null | LC_ALL=C sort -z)

  if [[ "${SUCCESS_COUNT}" -lt "${NUM_DEMOS}" ]]; then
    echo "Error: need ${NUM_DEMOS} successful *_rgb.h5, got ${SUCCESS_COUNT} (increase MP_OVERSAMPLE or check the MP trajectories)" >&2
    exit 1
  fi

  if [[ "${PRUNE_UNSELECTED_RGB}" == "1" ]]; then
    echo "=== Step 2b-prune: remove *_rgb.h5 not in selected_rgb_stems.txt ==="
    while IFS= read -r -d '' rgb; do
      stem=$(basename "${rgb}" .h5)
      if ! grep -qxF "${stem}" "${SELECTED_STEMS_FILE}"; then
        echo "  rm unselected ${rgb}"
        rm -f "${rgb}"
      fi
    done < <(command find "${REPLAY_DATA_ROOT}" -type f -name '*_rgb.h5' -print0 2>/dev/null)
  fi
else
  echo "=== Step 2: skipped (SKIP_REPLAY=1) ==="
fi

command mkdir -p "${REPLAY_DATA_ROOT}"

# -----------------------------------------------------------------------------
# Step 2c: *_rgb.h5 -> agent_view / in_hand mp4 (recursive over replay_data)
# -----------------------------------------------------------------------------
if [[ "${SAVE_DUAL_MP4}" == "1" ]]; then
  echo "=== Step 2c: *_rgb.h5 -> agent_view + in_hand mp4 ==="
  DUAL_ARGS=(--motion-dir "${REPLAY_DATA_ROOT}" --fps "${DUAL_MP4_FPS}")
  if [[ -f "${SELECTED_STEMS_FILE}" ]] && [[ -s "${SELECTED_STEMS_FILE}" ]]; then
    DUAL_ARGS+=(--stems-file "${SELECTED_STEMS_FILE}")
  fi
  "${PYTHON}" "${REPO_ROOT}/collection_strategy/rgb_h5_to_dual_camera_mp4.py" "${DUAL_ARGS[@]}"
fi

# -----------------------------------------------------------------------------
# Step 3a–b: task_map + LeRobot -> huggingface_data/
# -----------------------------------------------------------------------------
if [[ "${SKIP_CONVERT}" != "1" ]]; then
  echo "=== Step 3a: build task_map.json -> ${TASK_MAP_JSON} ==="
  command mkdir -p "${REPLAY_DATA_ROOT}"
  MAP_EXTRA=()
  if [[ -f "${SELECTED_STEMS_FILE}" ]] && [[ -s "${SELECTED_STEMS_FILE}" ]]; then
    MAP_EXTRA=(--stems-file "${SELECTED_STEMS_FILE}")
  fi
  "${PYTHON}" "${REPO_ROOT}/collection_strategy/build_pairwise_task_map.py" \
    --demos-dir "${REPLAY_DATA_ROOT}" \
    --output "${TASK_MAP_JSON}" \
    "${MAP_EXTRA[@]}"

  echo "=== Step 3b: convert to LeRobot -> ${HUGGINGFACE_DATA_ROOT} ==="
  command mkdir -p "${HUGGINGFACE_DATA_ROOT}"

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
  echo "=== Step 3: skipped (SKIP_CONVERT=1) ==="
fi

echo "Done. RUN_ROOT=${RUN_ROOT}"
echo "  original_data=${ORIGINAL_DATA_ROOT}"
echo "  replay_data=${REPLAY_DATA_ROOT}"
echo "  huggingface_data=${HUGGINGFACE_DATA_ROOT}"

