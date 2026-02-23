#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v11/outputs"
mkdir -p "${OUTPUT_DIR}/logs"

STAGE="${1:-short}"  # smoke | short | formal | promote | group_d | group_e | group_f | group_g
case "${STAGE}" in
    smoke) SCRIPT="attention_robust_v11/train_v11_smoke.sh" ;;
    short) SCRIPT="attention_robust_v11/train_v11_short.sh" ;;
    formal) SCRIPT="attention_robust_v11/train_v11_formal.sh" ;;
    promote) SCRIPT="attention_robust_v11/train_v11_formal_from_short.sh" ;;
    group_d) SCRIPT="attention_robust_v11/train_v11_group_d.sh" ;;
    group_e) SCRIPT="attention_robust_v11/train_v11_group_e.sh" ;;
    group_f) SCRIPT="attention_robust_v11/train_v11_group_f.sh" ;;
    group_g) SCRIPT="attention_robust_v11/train_v11_group_g.sh" ;;
    *)
        echo "Unknown stage: ${STAGE}. Use smoke | short | formal | promote | group_d | group_e | group_f | group_g."
        exit 1
        ;;
esac

TS=$(date +%Y%m%d_%H%M%S)
SESSION="v11_${STAGE}_${TS}"
LOG_FILE="${OUTPUT_DIR}/logs/${SESSION}.log"

screen -dmS "${SESSION}" bash -lc "cd '${MERBENCH_ROOT}' && GPU_ID='${GPU_ID:-0}' THREADS_PER_RUN='${THREADS_PER_RUN:-2}' NUM_WORKERS='${NUM_WORKERS:-0}' CPU_CORES='${CPU_CORES:-}' bash '${SCRIPT}' 2>&1 | tee '${LOG_FILE}'"

echo "SESSION=${SESSION}"
echo "GPU=${GPU_ID:-0} THREADS_PER_RUN=${THREADS_PER_RUN:-2} NUM_WORKERS=${NUM_WORKERS:-0} CPU_CORES=${CPU_CORES:-}"
echo "LOG=${LOG_FILE}"
echo "Check: screen -ls"
