#!/usr/bin/env bash
set -euo pipefail
# Start an OpenPI websocket policy server for one pi0 / pi0.5 checkpoint.
# Runs in the foreground; start it in its own terminal (or background it).
#
# Usage: bash pi0/start_server.sh <checkpoint_dir> [port=8000] [config=pi0_maniskill]
#   config: pi0_maniskill (pi0) or pi05_maniskill (pi0.5), defined in the OpenPI fork
# Env:
#   OPENPI_REPO  path to the OpenPI checkout that defines the ManiSkill configs (required)
#   CUDA_VISIBLE_DEVICES  GPU to load the model on
CKPT_DIR="${1:?checkpoint_dir (e.g. <run>/<step>)}"
PORT="${2:-8000}"
CONFIG="${3:-pi0_maniskill}"
: "${OPENPI_REPO:?set OPENPI_REPO to your OpenPI checkout}"

cd "${OPENPI_REPO}"
exec uv run scripts/serve_policy.py --port "${PORT}" policy:checkpoint \
  --policy.config "${CONFIG}" --policy.dir "${CKPT_DIR}"
