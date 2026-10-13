#!/usr/bin/env bash
# Pairwise Color x Spatial: collect (motion planning) -> replay to RGB H5 -> LeRobot dataset.
#
# Output layout (PIPELINE_OUT is required, via env var or first CLI arg):
#   <PIPELINE_OUT>/<NUM_DEMOS>/{original_data,replay_data,huggingface_data}/color_spatial/<STRATEGY>_f<FACTOR_COUNT>/
#
# Example (run from data_collection/):
#   PIPELINE_OUT=./outputs STRATEGY=stair FACTOR_COUNT=12 NUM_DEMOS=300 \
#     ./collection_strategy/color_spatial/collect_convert.sh
#
# STRATEGY: random | stair | Lrandom.  FACTOR_COUNT: 6 | 12 | 18.
# AUTO_SKIP_COLLECT / AUTO_SKIP_REPLAY / AUTO_SKIP_CONVERT (default 0): skip a step when its output already exists.
# SKIP_CONVERT_VIDEO=0 additionally writes a video-based LeRobot dataset with LEROBOT_VIDEO_PYTHON
# (see requirements_lerobot_video.txt).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

COLLECTION_EXPERIMENT=color_spatial

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

: "${STRATEGY:=random}"
: "${FACTOR_COUNT:=6}"
: "${NUM_DEMOS:=300}"
: "${SEED:=0}"
: "${CLEAN_LEROBOT:=0}"
: "${CLEAN_RECORD:=1}"
: "${SAVE_DUAL_MP4:=0}"
: "${MP_SAVE_VIDEO:=1}"
: "${DUAL_MP4_FPS:=30}"
: "${REPLAY_SAVE_HUMAN_VIDEO:=1}"
: "${SKIP_COLLECT:=0}"
: "${SKIP_REPLAY:=0}"
: "${SKIP_CONVERT:=0}"
: "${SKIP_CONVERT_VIDEO:=1}"
: "${AUTO_SKIP_COLLECT:=0}"
: "${AUTO_SKIP_REPLAY:=0}"
: "${AUTO_SKIP_CONVERT:=0}"
: "${AUTO_SKIP_CONVERT_VIDEO:=1}"
: "${LEROBOT_VIDEO_PYTHON:=${LEROBOT_PYTHON}}"
: "${SIM_BACKEND:=cpu}"
: "${RENDER_BACKEND:=cpu}"
: "${MP_OVERSAMPLE:=1.15}"
: "${LEROBOT_MAX_EPISODES:=}"

case "${STRATEGY}" in
  random|stair|Lrandom) ;;
  *)
    echo "Error: STRATEGY must be one of random, stair, Lrandom (got ${STRATEGY})" >&2
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

RUN_ROOT="${PIPELINE_OUT}/${NUM_DEMOS}"
command mkdir -p "${RUN_ROOT}"
RUN_ROOT=$(cd "${RUN_ROOT}" && pwd)
SEGMENT="${COLLECTION_EXPERIMENT}/${STRATEGY}_f${FACTOR_COUNT}"
ORIGINAL_DATA_ROOT="${RUN_ROOT}/original_data/${SEGMENT}"
REPLAY_DATA_ROOT="${RUN_ROOT}/replay_data/${SEGMENT}"
HUGGINGFACE_DATA_ROOT="${RUN_ROOT}/huggingface_data/${SEGMENT}"
HUGGINGFACE_DATA_VIDEO_ROOT="${RUN_ROOT}/huggingface_data_video/${SEGMENT}"
TASK_MAP_JSON="${REPLAY_DATA_ROOT}/task_map.json"
SELECTED_STEMS_FILE="${REPLAY_DATA_ROOT}/selected_rgb_stems.txt"
REPO_ID="${REPO_ID:-maniskill_${COLLECTION_EXPERIMENT}_${STRATEGY}_f${FACTOR_COUNT}_n${NUM_DEMOS}}"
: "${REPO_ID_VIDEO:=${REPO_ID}_video}"

if [[ "${CLEAN_LEROBOT}" == "1" ]]; then
  rm -rf "${HUGGINGFACE_DATA_ROOT}"
  rm -f "${TASK_MAP_JSON}" "${SELECTED_STEMS_FILE}"
fi
: "${CLEAN_LEROBOT_VIDEO:=0}"
if [[ "${CLEAN_LEROBOT_VIDEO}" == "1" ]]; then
  rm -rf "${HUGGINGFACE_DATA_VIDEO_ROOT}"
fi
if [[ "${CLEAN_RECORD}" == "1" ]]; then
  rm -rf "${ORIGINAL_DATA_ROOT}" "${REPLAY_DATA_ROOT}"
fi

DO_COLLECT=1
if [[ "${SKIP_COLLECT}" != "1" && "${AUTO_SKIP_COLLECT}" == "1" ]]; then
  command mkdir -p "${ORIGINAL_DATA_ROOT}"
  N_EXIST_RAW=$(command find "${ORIGINAL_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' 2>/dev/null | wc -l)
  if [[ "${N_EXIST_RAW}" -ge "${MP_RUNS}" ]]; then
    DO_COLLECT=0
    echo "=== AUTO: skip collect (raw .h5 count ${N_EXIST_RAW} >= MP_RUNS=${MP_RUNS}) ==="
  fi
fi
if [[ "${SKIP_COLLECT}" != "1" && "${DO_COLLECT}" == "1" ]]; then
  VID_ARGS=()
  if [[ "${MP_SAVE_VIDEO}" == "1" ]]; then
    VID_ARGS=(--save-video)
  fi
  "${PYTHON}" "${REPO_ROOT}/collection_strategy/color_spatial/collect.py" \
    --strategy "${STRATEGY}" \
    --factor-count "${FACTOR_COUNT}" \
    --record-segment "${STRATEGY}_f${FACTOR_COUNT}" \
    --record-base "${RUN_ROOT}/original_data" \
    --num-demos "${MP_RUNS}" \
    --seed "${SEED}" \
    --sim-backend "${SIM_BACKEND}" \
    --render-backend "${RENDER_BACKEND}" \
    "${VID_ARGS[@]}"
fi

if [[ "${SKIP_COLLECT}" == "1" && "${SKIP_REPLAY}" == "1" ]]; then
  command mkdir -p "${REPLAY_DATA_ROOT}"
  N_RGB_GATE=$(command find "${REPLAY_DATA_ROOT}" -type f -name '*_rgb.h5' 2>/dev/null | wc -l)
  if [[ "${N_RGB_GATE}" -lt "${NUM_DEMOS}" ]]; then
    echo "SKIP_COLLECT=1 SKIP_REPLAY=1 requires >= NUM_DEMOS *_rgb.h5 under ${REPLAY_DATA_ROOT} (found ${N_RGB_GATE})" >&2
    exit 1
  fi
  echo "=== convert only (collect/replay skipped): ${N_RGB_GATE} *_rgb.h5 >= NUM_DEMOS=${NUM_DEMOS} ==="
else
  N_RAW_MP=$(command find "${ORIGINAL_DATA_ROOT}" -type f -name '*.h5' ! -name '*_rgb.h5' 2>/dev/null | wc -l)
  if [[ "${N_RAW_MP}" -eq 0 ]]; then
    echo "No raw MP .h5 under ${ORIGINAL_DATA_ROOT}" >&2
    exit 1
  fi
fi

# set -o pipefail: find on missing dir breaks "find | wc" and aborts the whole script before replay/convert.
command mkdir -p "${REPLAY_DATA_ROOT}"

REPLAY_FROM_CACHE=0
if [[ "${SKIP_REPLAY}" != "1" && "${AUTO_SKIP_REPLAY}" == "1" ]]; then
  N_RGB_CACHED=$(command find "${REPLAY_DATA_ROOT}" -type f -name '*_rgb.h5' 2>/dev/null | wc -l)
  if [[ "${N_RGB_CACHED}" -ge "${NUM_DEMOS}" ]]; then
    REPLAY_FROM_CACHE=1
    echo "=== AUTO: skip replay (found ${N_RGB_CACHED} *_rgb.h5 >= NUM_DEMOS=${NUM_DEMOS}) ==="
  fi
fi

if [[ "${SKIP_REPLAY}" != "1" && "${REPLAY_FROM_CACHE}" == "0" ]]; then
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
elif [[ "${REPLAY_FROM_CACHE}" == "1" ]]; then
  command mkdir -p "${REPLAY_DATA_ROOT}"
  : > "${SELECTED_STEMS_FILE}"
  while IFS= read -r p; do
    echo "$(basename "${p}" .h5)" >> "${SELECTED_STEMS_FILE}"
  done < <(command find "${REPLAY_DATA_ROOT}" -type f -name '*_rgb.h5' 2>/dev/null | LC_ALL=C sort | command head -n "${NUM_DEMOS}")
fi

if [[ "${SKIP_REPLAY}" == "1" ]]; then
  command mkdir -p "${REPLAY_DATA_ROOT}"
  : > "${SELECTED_STEMS_FILE}"
  while IFS= read -r p; do
    echo "$(basename "${p}" .h5)" >> "${SELECTED_STEMS_FILE}"
  done < <(command find "${REPLAY_DATA_ROOT}" -type f -name '*_rgb.h5' 2>/dev/null | LC_ALL=C sort | command head -n "${NUM_DEMOS}")
fi

if [[ "${SAVE_DUAL_MP4}" == "1" ]]; then
  DUAL_ARGS=(--motion-dir "${REPLAY_DATA_ROOT}" --fps "${DUAL_MP4_FPS}")
  if [[ -f "${SELECTED_STEMS_FILE}" ]] && [[ -s "${SELECTED_STEMS_FILE}" ]]; then
    DUAL_ARGS+=(--stems-file "${SELECTED_STEMS_FILE}")
  fi
  "${PYTHON}" "${REPO_ROOT}/collection_strategy/rgb_h5_to_dual_camera_mp4.py" "${DUAL_ARGS[@]}"
fi

hf_image_complete() {
  [[ -f "${HUGGINGFACE_DATA_ROOT}/meta/info.json" ]] && \
    command find "${HUGGINGFACE_DATA_ROOT}/data" -name '*.parquet' -print -quit 2>/dev/null | grep -q .
}

hf_video_complete() {
  [[ -f "${HUGGINGFACE_DATA_VIDEO_ROOT}/meta/info.json" ]] && \
    command find "${HUGGINGFACE_DATA_VIDEO_ROOT}/data" -name '*.parquet' -print -quit 2>/dev/null | grep -q .
}

IMAGE_RUN=0
if [[ "${SKIP_CONVERT}" != "1" ]]; then
  if [[ "${AUTO_SKIP_CONVERT}" != "1" ]] || ! hf_image_complete; then
    IMAGE_RUN=1
  else
    echo "=== AUTO: skip image LeRobot (already exists): ${HUGGINGFACE_DATA_ROOT} ==="
  fi
fi

VIDEO_NEEDS_RUN=0
if [[ "${SKIP_CONVERT_VIDEO}" != "1" ]]; then
  if [[ "${AUTO_SKIP_CONVERT_VIDEO}" != "1" ]] || ! hf_video_complete; then
    VIDEO_NEEDS_RUN=1
  else
    echo "=== AUTO: skip video LeRobot (already exists): ${HUGGINGFACE_DATA_VIDEO_ROOT} ==="
  fi
fi

NEED_TASK_MAP=0
if [[ "${IMAGE_RUN}" == "1" || "${VIDEO_NEEDS_RUN}" == "1" ]]; then
  NEED_TASK_MAP=1
fi

if [[ "${NEED_TASK_MAP}" == "1" ]]; then
  MAP_EXTRA=()
  if [[ -f "${SELECTED_STEMS_FILE}" ]] && [[ -s "${SELECTED_STEMS_FILE}" ]]; then
    MAP_EXTRA=(--stems-file "${SELECTED_STEMS_FILE}")
  fi
  "${PYTHON}" "${REPO_ROOT}/collection_strategy/build_attribute_task_map.py" \
    --experiment "${COLLECTION_EXPERIMENT}" \
    --demos-dir "${REPLAY_DATA_ROOT}" \
    --output "${TASK_MAP_JSON}" \
    "${MAP_EXTRA[@]}"
fi

if [[ "${IMAGE_RUN}" == "1" ]]; then
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

if [[ "${SKIP_CONVERT_VIDEO}" != "1" && "${VIDEO_NEEDS_RUN}" == "1" ]]; then
  if [[ ! -f "${TASK_MAP_JSON}" ]]; then
    echo "=== Skip video LeRobot (missing task map: ${TASK_MAP_JSON}) ===" >&2
    exit 1
  fi
  if ! "${LEROBOT_VIDEO_PYTHON}" -c "import lerobot" 2>/dev/null; then
    echo "Error: cannot import lerobot with ${LEROBOT_VIDEO_PYTHON}." >&2
    echo "Install: pip install -r ${REPO_ROOT}/collection_strategy/requirements_lerobot_video.txt" >&2
    exit 1
  fi
  command mkdir -p "${HUGGINGFACE_DATA_VIDEO_ROOT}"
  MAX_V_ARGS=()
  if [[ -n "${LEROBOT_MAX_EPISODES:-}" ]]; then
    MAX_V_ARGS=(--max-episodes "${LEROBOT_MAX_EPISODES}")
  fi
  "${LEROBOT_VIDEO_PYTHON}" \
    "${REPO_ROOT}/collection_strategy/convert_maniskill_to_lerobot_video.py" \
    --demos-dir "${REPLAY_DATA_ROOT}" \
    --task-map-file "${TASK_MAP_JSON}" \
    --repo-id "${REPO_ID_VIDEO}" \
    --output-dir "${HUGGINGFACE_DATA_VIDEO_ROOT}" \
    --skip-if-output-complete \
    "${MAX_V_ARGS[@]}"
fi

