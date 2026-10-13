#!/usr/bin/env bash
# Provide a usable Vulkan ICD for SAPIEN / ManiSkill (avoids vk::createInstanceUnique: ErrorIncompatibleDriver).
#
# Rules (an already exported, valid VK_ICD_FILENAMES or MS_VULKAN_ICD_FILE always wins):
# - MANISKILL_USE_GPU=1: do nothing (system / NVIDIA ICD)
# - MANISKILL_USE_SYSTEM_VULKAN=1: do nothing (use the system default Vulkan)
# - SIM_BACKEND=cpu and RENDER_BACKEND=cpu (collect_convert default): use Lavapipe if lvp_icd*.json exists
# - otherwise: fall back to Lavapipe only if nvidia-smi fails within 5 s
vulkan_headless_default() {
  if [[ -n "${VK_ICD_FILENAMES:-}" ]]; then
    # Respect caller-provided valid ICDs; if all paths are stale, fall through and repair.
    local _ok=1
    local _p=""
    IFS=':' read -r -a _vk_paths <<<"${VK_ICD_FILENAMES}"
    for _p in "${_vk_paths[@]}"; do
      if [[ -f "${_p}" ]]; then
        _ok=0
        break
      fi
    done
    if [[ "${_ok}" -eq 0 ]]; then
      return 0
    fi
    unset VK_ICD_FILENAMES
  fi
  if [[ -n "${MS_VULKAN_ICD_FILE:-}" ]]; then
    return 0
  fi

  local lvp=""
  if [[ -f /usr/share/vulkan/icd.d/lvp_icd.json ]]; then
    lvp=/usr/share/vulkan/icd.d/lvp_icd.json
  elif [[ -f /usr/share/vulkan/icd.d/lvp_icd.x86_64.json ]]; then
    lvp=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
  else
    return 0
  fi

  if [[ "${MANISKILL_USE_GPU:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "${MANISKILL_USE_SYSTEM_VULKAN:-0}" == "1" ]]; then
    return 0
  fi

  if [[ "${SIM_BACKEND:-cpu}" == "cpu" ]] && [[ "${RENDER_BACKEND:-cpu}" == "cpu" ]]; then
    export VK_ICD_FILENAMES="${lvp}"
    [[ -n "${DEBUG_VULKAN:-}" ]] && echo "[vulkan_headless_default] VK_ICD_FILENAMES=${lvp} (SIM/RENDER=cpu)" >&2
    return 0
  fi

  if command -v nvidia-smi &>/dev/null && timeout 5 nvidia-smi &>/dev/null; then
    return 0
  fi

  export VK_ICD_FILENAMES="${lvp}"
  [[ -n "${DEBUG_VULKAN:-}" ]] && echo "[vulkan_headless_default] VK_ICD_FILENAMES=${lvp} (nvidia-smi unavailable)" >&2
}
