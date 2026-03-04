#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SEED="${SEED:-8407}"
CASE_ID="a12_v14_mse_hard"
RUN_TAG="${RUN_TAG:-${CASE_ID}_seed${SEED}}"

CASE_ID="${CASE_ID}" \
SEED="${SEED}" \
RUN_TAG="${RUN_TAG}" \
GPU_ID="${GPU_ID:-0}" \
BATCH_SIZE="${BATCH_SIZE:-20}" \
NUM_WORKERS="${NUM_WORKERS:-0}" \
THREADS_PER_RUN="${THREADS_PER_RUN:-1}" \
    bash "${SCRIPT_DIR}/train_v24_ablation_single_case.sh"
