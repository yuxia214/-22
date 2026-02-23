#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v11/outputs"
mkdir -p "${OUTPUT_DIR}/logs"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    GPU_COUNT=1
fi

GPU_F="${GPU_F:-0}"
if [[ -z "${GPU_G+x}" ]]; then
    if [[ "${GPU_COUNT}" -ge 2 ]]; then
        GPU_G=1
    else
        GPU_G=0
    fi
else
    GPU_G="${GPU_G}"
fi

THREADS_F="${THREADS_F:-1}"
THREADS_G="${THREADS_G:-1}"
WORKERS_F="${WORKERS_F:-1}"
WORKERS_G="${WORKERS_G:-1}"

if [[ -z "${CPU_F+x}" || -z "${CPU_G+x}" ]]; then
    CPU_TOTAL=$(nproc)
    if [[ "${CPU_TOTAL}" -ge 12 ]]; then
        CPU_F="${CPU_F:-0-5}"
        CPU_G="${CPU_G:-6-11}"
    elif [[ "${CPU_TOTAL}" -ge 8 ]]; then
        CPU_F="${CPU_F:-0-3}"
        CPU_G="${CPU_G:-4-7}"
    else
        CPU_F="${CPU_F:-}"
        CPU_G="${CPU_G:-}"
    fi
fi

TS=$(date +%Y%m%d_%H%M%S)
SESSION_F="v11_group_f_${TS}"
SESSION_G="v11_group_g_${TS}"
LOG_F="${OUTPUT_DIR}/logs/${SESSION_F}.log"
LOG_G="${OUTPUT_DIR}/logs/${SESSION_G}.log"

screen -dmS "${SESSION_F}" bash -lc "cd '${MERBENCH_ROOT}' && GPU_ID='${GPU_F}' THREADS_PER_RUN='${THREADS_F}' NUM_WORKERS='${WORKERS_F}' CPU_CORES='${CPU_F}' bash attention_robust_v11/train_v11_group_f.sh 2>&1 | tee '${LOG_F}'"
screen -dmS "${SESSION_G}" bash -lc "cd '${MERBENCH_ROOT}' && GPU_ID='${GPU_G}' THREADS_PER_RUN='${THREADS_G}' NUM_WORKERS='${WORKERS_G}' CPU_CORES='${CPU_G}' bash attention_robust_v11/train_v11_group_g.sh 2>&1 | tee '${LOG_G}'"

if [[ "${GPU_F}" == "${GPU_G}" ]]; then
    echo "WARN: both sessions use GPU ${GPU_F}; keep small batch sizes."
fi
echo "SESSION_F=${SESSION_F} GPU=${GPU_F} THREADS=${THREADS_F} WORKERS=${WORKERS_F} CPU=${CPU_F} LOG=${LOG_F}"
echo "SESSION_G=${SESSION_G} GPU=${GPU_G} THREADS=${THREADS_G} WORKERS=${WORKERS_G} CPU=${CPU_G} LOG=${LOG_G}"
echo "Check: screen -ls"
