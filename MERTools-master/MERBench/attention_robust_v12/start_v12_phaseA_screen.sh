#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
LOG_DIR="${MERBENCH_ROOT}/attention_robust_v12/outputs/logs"
mkdir -p "${LOG_DIR}"

SESSION_NAME="v12_phaseA_fast"
LOG_FILE="${LOG_DIR}/${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Kill stale session with same name.
if screen -list | grep -q "\.${SESSION_NAME}[[:space:]]"; then
    screen -S "${SESSION_NAME}" -X quit || true
fi

CMD="cd ${MERBENCH_ROOT} && bash attention_robust_v12/train_v12_phaseA_fast.sh 2>&1 | tee ${LOG_FILE}"
screen -dmS "${SESSION_NAME}" bash -lc "${CMD}"

echo "started session=${SESSION_NAME}"
echo "log=${LOG_FILE}"
echo "attach: screen -r ${SESSION_NAME}"
echo "tail:   tail -f ${LOG_FILE}"
