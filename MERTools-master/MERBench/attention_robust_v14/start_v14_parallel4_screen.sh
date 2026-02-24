#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
LOG_DIR="${MERBENCH_ROOT}/attention_robust_v14/outputs/logs"
mkdir -p "${LOG_DIR}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-8407}"
THREADS_PER_RUN="${THREADS_PER_RUN:-2}"
NUM_WORKERS="${NUM_WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-20}"
EPOCHS="${EPOCHS:-45}"

CASES=(e1 e2 e3 e4)
CORE_MAP=(0-2 3-5 6-8 9-11)

if [[ "${#CASES[@]}" -ne "${#CORE_MAP[@]}" ]]; then
    echo "cases/core_map length mismatch"
    exit 2
fi

cd "${MERBENCH_ROOT}"

for i in "${!CASES[@]}"; do
    case_id="${CASES[$i]}"
    cpu_cores="${CORE_MAP[$i]}"
    run_tag="p4_${case_id}"
    session_name="v14_${run_tag}"
    log_file="${LOG_DIR}/${session_name}_$(date +%Y%m%d_%H%M%S).log"

    if screen -list | grep -q "\\.${session_name}[[:space:]]"; then
        screen -S "${session_name}" -X quit || true
    fi

    cmd="cd ${MERBENCH_ROOT} && GPU_ID=${GPU_ID} SEED=${SEED} THREADS_PER_RUN=${THREADS_PER_RUN} NUM_WORKERS=${NUM_WORKERS} BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS} CPU_CORES=${cpu_cores} CASE_ID=${case_id} RUN_TAG=${run_tag} bash attention_robust_v14/train_v14_single_case.sh 2>&1 | tee ${log_file}"
    screen -dmS "${session_name}" bash -lc "${cmd}"

    echo "started session=${session_name} case=${case_id} cpu_cores=${cpu_cores}"
    echo "log=${log_file}"
done

echo "------ active screens ------"
screen -ls || true
