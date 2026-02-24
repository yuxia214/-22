#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
LOG_DIR="${MERBENCH_ROOT}/attention_robust_v13/outputs/logs"
mkdir -p "${LOG_DIR}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-8407}"
THREADS_PER_RUN="${THREADS_PER_RUN:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-20}"
EPOCHS="${EPOCHS:-45}"

CASES=(b1 b2 b3 b4 b5 b6)
CORE_MAP=(0-1 2-3 4-5 6-7 8-9 10-11)

cd "${MERBENCH_ROOT}"

for i in "${!CASES[@]}"; do
    case_id="${CASES[$i]}"
    cpu_cores="${CORE_MAP[$i]}"
    run_tag="p6_${case_id}"
    session_name="v13_${run_tag}"
    log_file="${LOG_DIR}/${session_name}_$(date +%Y%m%d_%H%M%S).log"

    if screen -list | grep -q "\\.${session_name}[[:space:]]"; then
        screen -S "${session_name}" -X quit || true
    fi

    cmd="cd ${MERBENCH_ROOT} && GPU_ID=${GPU_ID} SEED=${SEED} THREADS_PER_RUN=${THREADS_PER_RUN} NUM_WORKERS=${NUM_WORKERS} BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS} CPU_CORES=${cpu_cores} CASE_ID=${case_id} RUN_TAG=${run_tag} bash attention_robust_v13/train_v13_single_case.sh 2>&1 | tee ${log_file}"
    screen -dmS "${session_name}" bash -lc "${cmd}"

    echo "started session=${session_name} case=${case_id} cpu_cores=${cpu_cores}"
    echo "log=${log_file}"
done

echo "------ active screens ------"
screen -ls || true
