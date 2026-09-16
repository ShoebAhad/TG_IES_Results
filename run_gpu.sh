#!/usr/bin/env bash
set -euo pipefail

# Usage: from workspace root run: Research_implementations/2025_ICLR_IES/run_gpu.sh [--args]
# Ensure .venv exists in workspace root. To install dependencies, run ./setup_venv.sh first.

VENV_DIR=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$VENV_DIR" ]; then
  echo "Virtualenv $VENV_DIR not found in workspace root. Run ./setup_venv.sh first." >&2
  exit 1
fi

# Activate venv
. "$VENV_DIR/bin/activate"

# Default to first GPU if available; override with CUDA_VISIBLE_DEVICES env var
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u "$SCRIPT_DIR/cifar_main.py" "$@"
