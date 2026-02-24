#!/bin/bash
set -euo pipefail

# ================================================================
# run_v17_parallel.sh — 并行多种子 × 多配置 训练 + 集成
# 4 configs × 2 seeds = 8 runs, 4 parallel at a time
# GPU: ~510 MiB per process (32 GB total → 轻松并行)
# CPU: 12 cores → 3 cores per process (4 parallel)
# ================================================================

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
V17_DIR="${MERBENCH_ROOT}/attention_robust_v17"
OUTPUT_DIR="${V17_DIR}/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"

CONFIGS=(s1 s2 s3 s4)
SEEDS=(8407 42)

EPOCHS="${EPOCHS:-130}"
BATCH_SIZE="${BATCH_SIZE:-20}"

TOTAL=$((${#CONFIGS[@]} * ${#SEEDS[@]}))

echo "================================================================"
echo " V17 Parallel Training + Ensemble Pipeline"
echo " Configs: ${CONFIGS[*]}"
echo " Seeds:   ${SEEDS[*]}"
echo " Total runs: ${TOTAL}"
echo " Parallel: 4 at a time"
echo "================================================================"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/summary"
cd "${MERBENCH_ROOT}"

# ============================================================
# Phase 1: Parallel training (4 at a time)
# ============================================================
run_single() {
    local cfg="$1"
    local seed="$2"
    local cpu_cores="$3"
    local run_tag="${cfg}_seed${seed}"
    local log_file="${OUTPUT_DIR}/logs/v17_${run_tag}_$(date +%Y%m%d_%H%M%S).log"

    echo "[START] ${run_tag} on cores ${cpu_cores}"

    CASE_ID="${cfg}" \
    SEED="${seed}" \
    RUN_TAG="${run_tag}" \
    GPU_ID="${GPU_ID}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    EPOCHS="${EPOCHS}" \
    NUM_WORKERS=0 \
    THREADS_PER_RUN=3 \
    CPU_CORES="${cpu_cores}" \
        bash "${V17_DIR}/train_v17_single_case.sh" \
        > "${log_file}" 2>&1

    local exit_code=$?
    if [[ ${exit_code} -eq 0 ]]; then
        echo "[DONE]  ${run_tag}"
    else
        echo "[FAIL]  ${run_tag} (exit ${exit_code})"
    fi
    return ${exit_code}
}

run_batch() {
    # Run up to 4 jobs in parallel, assigning CPU cores
    local pids=()
    local tags=()
    local cores=("0-2" "3-5" "6-8" "9-11")
    local idx=0

    for job in "$@"; do
        local cfg=$(echo "${job}" | cut -d: -f1)
        local seed=$(echo "${job}" | cut -d: -f2)
        run_single "${cfg}" "${seed}" "${cores[$idx]}" &
        pids+=($!)
        tags+=("${cfg}_seed${seed}")
        idx=$((idx + 1))
    done

    # Wait for all to finish
    local failed=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || {
            echo "[WARN] ${tags[$i]} failed"
            failed=$((failed + 1))
        }
    done
    return ${failed}
}

echo ""
echo "======== Phase 1: Training ${TOTAL} models (4 parallel) ========"
START_TIME=$(date +%s)

# Batch 1: s1+s2 × seed 8407, s3+s4 × seed 8407
echo ""
echo "--- Batch 1/2: 4 runs (configs s1-s4, seed 8407) ---"
run_batch "s1:8407" "s2:8407" "s3:8407" "s4:8407"
BATCH1_FAILED=$?

# Batch 2: s1+s2 × seed 42, s3+s4 × seed 42
echo ""
echo "--- Batch 2/2: 4 runs (configs s1-s4, seed 42) ---"
run_batch "s1:42" "s2:42" "s3:42" "s4:42"
BATCH2_FAILED=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "======== Phase 1 complete in ${ELAPSED}s ($(( ELAPSED / 60 ))m) ========"
echo "  Batch 1 failures: ${BATCH1_FAILED}"
echo "  Batch 2 failures: ${BATCH2_FAILED}"

# ============================================================
# Phase 2: Ensemble
# ============================================================
echo ""
echo "======== Phase 2: Ensemble ========"

RESULT_DIRS=()
for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        DIR="${OUTPUT_DIR}/results/${cfg}_seed${seed}"
        if [[ -d "${DIR}" ]]; then
            RESULT_DIRS+=("${DIR}")
        else
            echo "  [WARN] Missing: ${DIR}"
        fi
    done
done

if [[ ${#RESULT_DIRS[@]} -lt 2 ]]; then
    echo "ERROR: Need at least 2 result dirs for ensemble"
    exit 1
fi

echo "  Ensembling ${#RESULT_DIRS[@]} models"
ENSEMBLE_OUTPUT="${OUTPUT_DIR}/ensemble"
mkdir -p "${ENSEMBLE_OUTPUT}"

for ts in test1 test2 test3; do
    echo ""
    echo "  --- ${ts} ---"
    "${PYTHON_BIN}" "${V17_DIR}/ensemble_predictions.py" \
        --result_dirs "${RESULT_DIRS[@]}" \
        --test_set "${ts}" \
        --output_dir "${ENSEMBLE_OUTPUT}" 2>&1 || true
done

echo ""
echo "================================================================"
echo " V17 Pipeline Complete! (${ELAPSED}s total)"
echo " Ensemble: ${ENSEMBLE_OUTPUT}"
echo " Target: Combined > 0.7005 on test1"
echo "================================================================"
