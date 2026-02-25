#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
V17_DIR="${MERBENCH_ROOT}/attention_robust_v17"
OUTPUT_DIR="${V17_DIR}/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"

CASES_STR="${CASES:-c1 c2 c3 c4}"
SEEDS_STR="${SEEDS:-8407}"
read -r -a CASES <<< "${CASES_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"

EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
THREADS_PER_RUN="${THREADS_PER_RUN:-2}"
USE_TASKSET="${USE_TASKSET:-0}"
CPU_SLOTS_STR="${CPU_SLOTS:-0-2 3-5 6-8 9-11}"
read -r -a CPU_SLOTS <<< "${CPU_SLOTS_STR}"

TUNE_SET="${TUNE_SET:-test1}"
EVAL_SETS_STR="${EVAL_SETS:-test1 test2 test3}"
read -r -a EVAL_SETS <<< "${EVAL_SETS_STR}"
MAX_MODELS="${MAX_MODELS:-6}"
SEARCH_TRIALS="${SEARCH_TRIALS:-10000}"
ALPHA_GRID="${ALPHA_GRID:-0.05,0.10,0.15,0.20,0.30,0.40,0.50}"
CORR_PENALTY="${CORR_PENALTY:-0.004}"
CORR_THRESHOLD="${CORR_THRESHOLD:-0.985}"
SKIP_ENSEMBLE="${SKIP_ENSEMBLE:-0}"

TOTAL=$((${#CASES[@]} * ${#SEEDS[@]}))
COUNT=0
FAILED=0

echo "================================================================"
echo " V19 Boost Pipeline"
echo " Cases: ${CASES[*]}"
echo " Seeds: ${SEEDS[*]}"
echo " Total runs: ${TOTAL}"
echo " Batch: ${BATCH_SIZE}"
echo " Parallel jobs: ${PARALLEL_JOBS}"
echo " Taskset: ${USE_TASKSET} (CPU slots: ${CPU_SLOTS_STR})"
echo "================================================================"
echo ""

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/summary" "${OUTPUT_DIR}/ensemble_v19"

if [[ "${PARALLEL_JOBS}" -lt 1 ]]; then
    echo "ERROR: PARALLEL_JOBS must be >=1, got ${PARALLEL_JOBS}"
    exit 2
fi
if [[ "${USE_TASKSET}" == "1" && "${PARALLEL_JOBS}" -gt "${#CPU_SLOTS[@]}" ]]; then
    echo "ERROR: PARALLEL_JOBS=${PARALLEL_JOBS}, but CPU_SLOTS only has ${#CPU_SLOTS[@]} entries"
    echo "       set CPU_SLOTS or lower PARALLEL_JOBS"
    exit 2
fi

# ============================================================
# Phase 1: training
# ============================================================
echo "======== Phase 1: Training ${TOTAL} runs (${PARALLEL_JOBS} parallel) ========"

run_single() {
    local cfg="$1"
    local seed="$2"
    local cpu_cores="${3:-}"

    local run_tag="${cfg}_seed${seed}"
    local log_file="${OUTPUT_DIR}/logs/v19_${run_tag}_$(date +%Y%m%d_%H%M%S).log"

    echo "[START] ${run_tag} cpu=${cpu_cores:-none} log=${log_file}"

    if [[ -n "${EPOCHS}" ]]; then
        CASE_ID="${cfg}" \
        SEED="${seed}" \
        RUN_TAG="${run_tag}" \
        GPU_ID="${GPU_ID}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        EPOCHS="${EPOCHS}" \
        NUM_WORKERS="${NUM_WORKERS}" \
        THREADS_PER_RUN="${THREADS_PER_RUN}" \
        CPU_CORES="${cpu_cores}" \
            bash "${V17_DIR}/train_v19_single_case.sh" > "${log_file}" 2>&1
    else
        CASE_ID="${cfg}" \
        SEED="${seed}" \
        RUN_TAG="${run_tag}" \
        GPU_ID="${GPU_ID}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        NUM_WORKERS="${NUM_WORKERS}" \
        THREADS_PER_RUN="${THREADS_PER_RUN}" \
        CPU_CORES="${cpu_cores}" \
            bash "${V17_DIR}/train_v19_single_case.sh" > "${log_file}" 2>&1
    fi

    local exit_code=$?
    if [[ ${exit_code} -eq 0 ]]; then
        echo "[DONE] ${run_tag}"
    else
        echo "[FAIL] ${run_tag} (exit ${exit_code})"
    fi
    return ${exit_code}
}

run_batch() {
    local pids=()
    local idx=0

    for job in "$@"; do
        local cfg="${job%%:*}"
        local seed="${job##*:}"
        local cpu_cores=""
        if [[ "${USE_TASKSET}" == "1" ]]; then
            cpu_cores="${CPU_SLOTS[$idx]}"
        fi
        run_single "${cfg}" "${seed}" "${cpu_cores}" &
        pids+=($!)
        idx=$((idx + 1))
    done

    local failed=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || {
            failed=$((failed + 1))
        }
    done
    return ${failed}
}

JOBS=()
for cfg in "${CASES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${cfg}:${seed}")
    done
done

TOTAL_JOBS=${#JOBS[@]}
BATCH_INDEX=0
START_TS=$(date +%s)
for ((offset=0; offset<TOTAL_JOBS; offset+=PARALLEL_JOBS)); do
    BATCH_INDEX=$((BATCH_INDEX + 1))
    BATCH=("${JOBS[@]:offset:PARALLEL_JOBS}")
    BATCH_SIZE_REAL=${#BATCH[@]}
    BATCH_END=$((offset + BATCH_SIZE_REAL))
    echo ""
    echo "------------------------------------------------------------"
    echo " Batch ${BATCH_INDEX}: jobs $((offset + 1))-${BATCH_END}/${TOTAL_JOBS}"
    echo "------------------------------------------------------------"
    for job in "${BATCH[@]}"; do
        COUNT=$((COUNT + 1))
        echo "  [${COUNT}/${TOTAL}] ${job}"
    done

    set +e
    run_batch "${BATCH[@]}"
    BATCH_FAILED=$?
    set -e
    FAILED=$((FAILED + BATCH_FAILED))
    echo " Batch ${BATCH_INDEX} done, failed=${BATCH_FAILED}"
done
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo ""
echo "======== Phase 1 complete: ${COUNT} runs, ${FAILED} failed, elapsed=${ELAPSED}s ========"

if [[ ${FAILED} -ge ${TOTAL} ]]; then
    echo "ERROR: all runs failed. aborting."
    exit 1
fi

if [[ "${SKIP_ENSEMBLE}" == "1" ]]; then
    echo "skip ensemble because SKIP_ENSEMBLE=1"
    exit 0
fi

# ============================================================
# Phase 2: collect result dirs
# ============================================================
echo ""
echo "======== Phase 2: Collect result dirs ========"

RESULT_DIRS=()
for cfg in "${CASES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_TAG="${cfg}_seed${seed}"
        DIR="${OUTPUT_DIR}/results/${RUN_TAG}"
        if [[ -d "${DIR}" ]]; then
            RESULT_DIRS+=("${DIR}")
        else
            echo "  [WARN] missing result dir: ${DIR}"
        fi
    done
done

if [[ ${#RESULT_DIRS[@]} -lt 2 ]]; then
    echo "ERROR: need at least 2 result dirs for ensemble, found ${#RESULT_DIRS[@]}"
    exit 1
fi

echo "  Found ${#RESULT_DIRS[@]} result directories"

# ============================================================
# Phase 3: v19 ensemble
# ============================================================
echo ""
echo "======== Phase 3: v19 Ensemble ========"

ENSEMBLE_OUTPUT="${OUTPUT_DIR}/ensemble_v19"
mkdir -p "${ENSEMBLE_OUTPUT}"

"${PYTHON_BIN}" "${V17_DIR}/ensemble_predictions_v19.py" \
    --result_dirs "${RESULT_DIRS[@]}" \
    --tune_set "${TUNE_SET}" \
    --eval_sets "${EVAL_SETS[@]}" \
    --max_models "${MAX_MODELS}" \
    --alpha_grid "${ALPHA_GRID}" \
    --search_trials "${SEARCH_TRIALS}" \
    --corr_penalty "${CORR_PENALTY}" \
    --corr_threshold "${CORR_THRESHOLD}" \
    --seed 42 \
    --output_dir "${ENSEMBLE_OUTPUT}" \
    --save_prefix "v19_${TUNE_SET}"

echo ""
echo "================================================================"
echo " V19 Boost Pipeline Complete"
echo " Ensemble outputs: ${ENSEMBLE_OUTPUT}"
echo " Target: test1 Combined > 0.7005"
echo "================================================================"
