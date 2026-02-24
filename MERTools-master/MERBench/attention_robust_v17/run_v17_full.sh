#!/bin/bash
set -euo pipefail

# ================================================================
# run_v17_full.sh — 多种子 × 多配置 训练 + 集成 流水线
# 目标: 在 MER-MULTI(test1) 上 Combined > 0.7005
#
# 策略:
#   4 configs (s1-s4) × 3 seeds (8407, 42, 2023) = 12 diverse models
#   然后用 ensemble_predictions.py 等权平均所有 test1 预测
# ================================================================

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
V17_DIR="${MERBENCH_ROOT}/attention_robust_v17"
OUTPUT_DIR="${V17_DIR}/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"

CONFIGS=(s1 s2 s3 s4)
SEEDS=(8407 42 2023)

EPOCHS="${EPOCHS:-130}"
BATCH_SIZE="${BATCH_SIZE:-20}"
NUM_WORKERS="${NUM_WORKERS:-1}"

TOTAL=$((${#CONFIGS[@]} * ${#SEEDS[@]}))
COUNT=0
FAILED=0

echo "================================================================"
echo " V17 Full Training + Ensemble Pipeline"
echo " Configs: ${CONFIGS[*]}"
echo " Seeds:   ${SEEDS[*]}"
echo " Total runs: ${TOTAL}"
echo " Epochs: ${EPOCHS}, Batch: ${BATCH_SIZE}"
echo "================================================================"
echo ""

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/summary"

# ============================================================
# Phase 1: Sequential training (single GPU)
# ============================================================
echo "======== Phase 1: Training ${TOTAL} models ========"

for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        RUN_TAG="${cfg}_seed${seed}"
        LOG_FILE="${OUTPUT_DIR}/logs/v17_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"

        echo ""
        echo "------------------------------------------------------------"
        echo " [${COUNT}/${TOTAL}] Config=${cfg} Seed=${seed} Tag=${RUN_TAG}"
        echo " Log: ${LOG_FILE}"
        echo "------------------------------------------------------------"

        set +e
        CASE_ID="${cfg}" \
        SEED="${seed}" \
        RUN_TAG="${RUN_TAG}" \
        GPU_ID="${GPU_ID}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        EPOCHS="${EPOCHS}" \
        NUM_WORKERS="${NUM_WORKERS}" \
            bash "${V17_DIR}/train_v17_single_case.sh" 2>&1 | tee "${LOG_FILE}"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        if [[ ${EXIT_CODE} -ne 0 ]]; then
            echo "  >>> FAILED (exit code ${EXIT_CODE})"
            FAILED=$((FAILED + 1))
        else
            echo "  >>> DONE"
        fi
    done
done

echo ""
echo "======== Phase 1 complete: ${COUNT} runs, ${FAILED} failed ========"

if [[ ${FAILED} -ge ${TOTAL} ]]; then
    echo "ERROR: All runs failed. Aborting."
    exit 1
fi

# ============================================================
# Phase 2: Collect result directories for ensemble
# ============================================================
echo ""
echo "======== Phase 2: Ensemble ========"

RESULT_DIRS=()
for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_TAG="${cfg}_seed${seed}"
        DIR="${OUTPUT_DIR}/results/${RUN_TAG}"
        if [[ -d "${DIR}" ]]; then
            RESULT_DIRS+=("${DIR}")
        else
            echo "  [WARN] Missing result dir: ${DIR}"
        fi
    done
done

if [[ ${#RESULT_DIRS[@]} -lt 2 ]]; then
    echo "ERROR: Need at least 2 result dirs for ensemble, found ${#RESULT_DIRS[@]}"
    exit 1
fi

echo "  Found ${#RESULT_DIRS[@]} result directories for ensemble"

ENSEMBLE_OUTPUT="${OUTPUT_DIR}/ensemble"
mkdir -p "${ENSEMBLE_OUTPUT}"

"${PYTHON_BIN}" "${V17_DIR}/ensemble_predictions.py" \
    --result_dirs "${RESULT_DIRS[@]}" \
    --test_set test1 \
    --output_dir "${ENSEMBLE_OUTPUT}"

echo ""
echo "======== Phase 2 complete ========"

# ============================================================
# Phase 3: Also ensemble test2 and test3 if available
# ============================================================
for ts in test2 test3; do
    echo ""
    echo "  Attempting ensemble for ${ts}..."
    set +e
    "${PYTHON_BIN}" "${V17_DIR}/ensemble_predictions.py" \
        --result_dirs "${RESULT_DIRS[@]}" \
        --test_set "${ts}" \
        --output_dir "${ENSEMBLE_OUTPUT}" 2>&1
    set -e
done

echo ""
echo "================================================================"
echo " V17 Pipeline Complete!"
echo " Ensemble results saved in: ${ENSEMBLE_OUTPUT}"
echo " Target: Combined > 0.7005"
echo "================================================================"
