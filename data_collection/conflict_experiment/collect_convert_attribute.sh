#!/usr/bin/env bash
# Conflict (factor-correlation) data for FDR on the attribute pairs:
# collect -> replay to RGB H5 -> task_map -> LeRobot dataset.
# COLLECTION_EXPERIMENT: verb_size | verb_spatial | object_spatial | color_spatial | color_size |
#   spatial_size | spatial_object | size_object
#
# Example (run from data_collection/):
#   PIPELINE_OUT=./outputs COLLECTION_EXPERIMENT=verb_size NUM_DEMOS=200 \
#     ./conflict_experiment/collect_convert_attribute.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

: "${COLLECTION_EXPERIMENT:=verb_size}"
if [[ "${COLLECTION_EXPERIMENT}" != "verb_size" && "${COLLECTION_EXPERIMENT}" != "verb_spatial" && "${COLLECTION_EXPERIMENT}" != "object_spatial" && "${COLLECTION_EXPERIMENT}" != "color_spatial" && "${COLLECTION_EXPERIMENT}" != "color_size" && "${COLLECTION_EXPERIMENT}" != "spatial_size" && "${COLLECTION_EXPERIMENT}" != "spatial_object" && "${COLLECTION_EXPERIMENT}" != "size_object" ]]; then
  echo "Error: COLLECTION_EXPERIMENT must be one of verb_size, verb_spatial, object_spatial, color_spatial, color_size, spatial_size, spatial_object, size_object (got ${COLLECTION_EXPERIMENT})" >&2
  exit 1
fi

if [[ -n "${1:-}" && "${1}" != -* ]]; then
  PIPELINE_OUT="${1}"
  shift
fi
if [[ -z "${PIPELINE_OUT:-}" ]]; then
  echo "Error: PIPELINE_OUT is required (env var or first CLI argument)" >&2
  exit 1
fi
command mkdir -p "${PIPELINE_OUT}"
PIPELINE_OUT=$(cd "${PIPELINE_OUT}" && pwd)

# Python interpreters (default: whatever `python` is on PATH, i.e. the active environment).
: "${PYTHON:=python}"
: "${LEROBOT_PYTHON:=${PYTHON}}"

: "${NUM_DEMOS:=300}"
: "${SEED:=0}"
: "${THIRD_INDICES:=0,1}"
: "${MP_OVERSAMPLE:=1.15}"
: "${MP_SAVE_VIDEO:=1}"
: "${REPLAY_SAVE_HUMAN_VIDEO:=1}"
: "${SAVE_DUAL_MP4:=0}"
: "${DUAL_MP4_FPS:=30}"
: "${SKIP_COLLECT:=0}"
: "${SKIP_REPLAY:=0}"
: "${SKIP_CONVERT:=0}"
: "${CLEAN_LEROBOT:=0}"
: "${CLEAN_RECORD:=1}"
: "${SIM_BACKEND:=cpu}"
: "${RENDER_BACKEND:=cpu}"
: "${LEROBOT_MAX_EPISODES:=}"

# Vulkan / SAPIEN: same as collection_strategy/*/collect_convert.sh (avoids
# 'Failed to find a supported physical device "cpu"' on the CPU render path).
# shellcheck disable=SC1091
source "${REPO_ROOT}/collection_strategy/lib/vulkan_headless_default.sh"
vulkan_headless_default

MP_RUNS=$(NUM_DEMOS="${NUM_DEMOS}" MP_OVERSAMPLE="${MP_OVERSAMPLE}" python3 - <<'PY'
import math, os
n = int(os.environ["NUM_DEMOS"])
o = float(os.environ["MP_OVERSAMPLE"])
print(max(n, math.ceil(n * o)))
PY
)

RUN_ROOT="${PIPELINE_OUT}/${NUM_DEMOS}"
command mkdir -p "${RUN_ROOT}"
RUN_ROOT=$(cd "${RUN_ROOT}" && pwd)
SEGMENT="${COLLECTION_EXPERIMENT}/conflict"
ORIGINAL_DATA_ROOT="${RUN_ROOT}/original_data/${SEGMENT}"
REPLAY_DATA_ROOT="${RUN_ROOT}/replay_data/${SEGMENT}"
HUGGINGFACE_DATA_ROOT="${RUN_ROOT}/huggingface_data/${SEGMENT}"
TASK_MAP_JSON="${REPLAY_DATA_ROOT}/task_map.json"
SELECTED_STEMS_FILE="${REPLAY_DATA_ROOT}/selected_rgb_stems.txt"
REPO_ID="${REPO_ID:-maniskill_${COLLECTION_EXPERIMENT}_conflict_n${NUM_DEMOS}}"

if [[ "${CLEAN_LEROBOT}" == "1" ]]; then
  rm -rf "${HUGGINGFACE_DATA_ROOT}"
  rm -f "${TASK_MAP_JSON}" "${SELECTED_STEMS_FILE}"
fi
if [[ "${CLEAN_RECORD}" == "1" ]]; then
  rm -rf "${ORIGINAL_DATA_ROOT}" "${REPLAY_DATA_ROOT}"
fi

if [[ "${SKIP_COLLECT}" != "1" ]]; then
  VID_ARGS=()
  if [[ "${MP_SAVE_VIDEO}" == "1" ]]; then
    VID_ARGS=(--save-video)
  fi
  "${PYTHON}" "${REPO_ROOT}/conflict_experiment/collect_conflict_attribute.py" \
    --experiment "${COLLECTION_EXPERIMENT}" \
    --record-base "${RUN_ROOT}/original_data" \
    --record-segment conflict \
    --third-indices "${THIRD_INDICES}" \
    --num-demos "${MP_RUNS}" \
    --seed "${SEED}" \
    --sim-backend "${SIM_BACKEND}" \
    --render-backend "${RENDER_BACKEND}" \
    "${VID_ARGS[@]}"
fi

N_RAW_MP=$(command find "${ORIGINAL_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' 2>/dev/null | wc -l)
if [[ "${N_RAW_MP}" -eq 0 ]]; then
  echo "No raw MP .h5 under ${ORIGINAL_DATA_ROOT}" >&2
  exit 1
fi

if [[ "${SKIP_REPLAY}" != "1" ]]; then
  while IFS= read -r -d '' h5; do
    base=$(basename "${h5}" .h5)
    rel="${h5#"${ORIGINAL_DATA_ROOT}/"}"
    dest_dir="${REPLAY_DATA_ROOT}/$(dirname "${rel}")"
    command mkdir -p "${dest_dir}"
    cp -f "${h5}" "${dest_dir}/${base}.h5"
    json_src="$(dirname "${h5}")/${base}.json"
    if [[ -f "${json_src}" ]]; then
      cp -f "${json_src}" "${dest_dir}/${base}.json"
    fi
  done < <(command find "${ORIGINAL_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' -print0 2>/dev/null)

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
    d=$(dirname "${h5}")
    out_rgb="${d}/${base}_rgb.h5"
    if [[ -f "${out_rgb}" ]]; then
      echo "$(basename "${out_rgb}" .h5)" >> "${SELECTED_STEMS_FILE}"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      continue
    fi
    set +e
    "${PYTHON}" -m mani_skill.trajectory.replay_trajectory \
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
  done < <(command find "${REPLAY_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' -print0 2>/dev/null | LC_ALL=C sort -z)
fi

if [[ "${SAVE_DUAL_MP4}" == "1" ]]; then
  DUAL_ARGS=(--motion-dir "${REPLAY_DATA_ROOT}" --fps "${DUAL_MP4_FPS}")
  if [[ -f "${SELECTED_STEMS_FILE}" ]] && [[ -s "${SELECTED_STEMS_FILE}" ]]; then
    DUAL_ARGS+=(--stems-file "${SELECTED_STEMS_FILE}")
  fi
  "${PYTHON}" "${REPO_ROOT}/collection_strategy/rgb_h5_to_dual_camera_mp4.py" "${DUAL_ARGS[@]}"
fi

if [[ "${SKIP_CONVERT}" != "1" ]]; then
  MAP_EXTRA=()
  if [[ -f "${SELECTED_STEMS_FILE}" ]] && [[ -s "${SELECTED_STEMS_FILE}" ]]; then
    MAP_EXTRA=(--stems-file "${SELECTED_STEMS_FILE}")
  fi
  "${PYTHON}" "${REPO_ROOT}/collection_strategy/build_attribute_task_map.py" \
    --experiment "${COLLECTION_EXPERIMENT}" \
    --demos-dir "${REPLAY_DATA_ROOT}" \
    --output "${TASK_MAP_JSON}" \
    "${MAP_EXTRA[@]}"

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
fi

