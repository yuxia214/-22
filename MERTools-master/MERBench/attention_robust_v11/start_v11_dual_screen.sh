#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v11/outputs"
mkdir -p "${OUTPUT_DIR}/logs"

# Start two extra groups concurrently.
# Example:
#   bash attention_robust_v11/start_v11_dual_screen.sh
#   GPU_A=0 GPU_B=0 bash attention_robust_v11/start_v11_dual_screen.sh

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    GPU_COUNT=1
fi

GPU_A="${GPU_A:-0}"
if [[ -z "${GPU_B+x}" ]]; then
    if [[ "${GPU_COUNT}" -ge 2 ]]; then
        GPU_B=1
    else
        GPU_B=0
    fi
else
    GPU_B="${GPU_B}"
fi

THREADS_A="${THREADS_A:-2}"
THREADS_B="${THREADS_B:-2}"
WORKERS_A="${WORKERS_A:-0}"
WORKERS_B="${WORKERS_B:-0}"

if [[ -z "${CPU_A+x}" || -z "${CPU_B+x}" ]]; then
    CPU_TOTAL=$(nproc)
    if [[ "${CPU_TOTAL}" -ge 12 ]]; then
        CPU_A="${CPU_A:-0-5}"
        CPU_B="${CPU_B:-6-11}"
    elif [[ "${CPU_TOTAL}" -ge 8 ]]; then
        CPU_A="${CPU_A:-0-3}"
        CPU_B="${CPU_B:-4-7}"
    else
        CPU_A="${CPU_A:-}"
        CPU_B="${CPU_B:-}"
    fi
fi
TS=$(date +%Y%m%d_%H%M%S)

SESSION_A="v11_group_d_${TS}"
SESSION_B="v11_group_e_${TS}"
LOG_A="${OUTPUT_DIR}/logs/${SESSION_A}.log"
LOG_B="${OUTPUT_DIR}/logs/${SESSION_B}.log"

screen -dmS "${SESSION_A}" bash -lc "cd '${MERBENCH_ROOT}' && GPU_ID='${GPU_A}' THREADS_PER_RUN='${THREADS_A}' NUM_WORKERS='${WORKERS_A}' CPU_CORES='${CPU_A}' bash attention_robust_v11/train_v11_group_d.sh 2>&1 | tee '${LOG_A}'"
screen -dmS "${SESSION_B}" bash -lc "cd '${MERBENCH_ROOT}' && GPU_ID='${GPU_B}' THREADS_PER_RUN='${THREADS_B}' NUM_WORKERS='${WORKERS_B}' CPU_CORES='${CPU_B}' bash attention_robust_v11/train_v11_group_e.sh 2>&1 | tee '${LOG_B}'"

if [[ "${GPU_A}" == "${GPU_B}" ]]; then
    echo "WARN: both sessions use GPU ${GPU_A}; lower batch size if OOM."
fi
echo "SESSION_A=${SESSION_A} GPU=${GPU_A} THREADS=${THREADS_A} WORKERS=${WORKERS_A} CPU=${CPU_A} LOG=${LOG_A}"
echo "SESSION_B=${SESSION_B} GPU=${GPU_B} THREADS=${THREADS_B} WORKERS=${WORKERS_B} CPU=${CPU_B} LOG=${LOG_B}"
echo "Check: screen -ls"
