#!/usr/bin/env bash
# Sequentially resumes the four still-incomplete experiment cells (each script has its
# own per-run SKIP logic keyed on csv line count, so this is safe to resume/re-invoke).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== run_remaining.sh START $(date) ==="

echo "--- run_adam.sh $(date) ---"
./run_adam.sh
echo "--- run_vgg16_seeds.sh $(date) ---"
./run_vgg16_seeds.sh
echo "--- run_resnetfixed_seeds.sh $(date) ---"
./run_resnetfixed_seeds.sh
echo "--- run_densenet.sh $(date) ---"
./run_densenet.sh

echo "=== run_remaining.sh ALL_DONE $(date) ==="
