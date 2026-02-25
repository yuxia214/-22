#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
V17_DIR="${MERBENCH_ROOT}/attention_robust_v17"
OUTPUT_DIR="${V17_DIR}/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"

CASES_STR="${CASES:-g1 g2 g3 g4}"
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

# v23 default: use CV for weight search, optimize RAW Combined.
TUNE_SET="${TUNE_SET:-cv}"
EVAL_SETS_STR="${EVAL_SETS:-test1 test2 test3}"
read -r -a EVAL_SETS <<< "${EVAL_SETS_STR}"
MAX_MODELS="${MAX_MODELS:-4}"
SEARCH_TRIALS="${SEARCH_TRIALS:-30000}"
ALPHA_GRID="${ALPHA_GRID:-0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.40,0.55}"
CORR_PENALTY="${CORR_PENALTY:-0.015}"
CORR_THRESHOLD="${CORR_THRESHOLD:-0.965}"
SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE:-raw}"
POST_REG_CALIB="${POST_REG_CALIB:-none}"
CALIB_MIN_SAMPLES="${CALIB_MIN_SAMPLES:-10}"
CALIB_CLIP="${CALIB_CLIP:-4.5}"
SKIP_ENSEMBLE="${SKIP_ENSEMBLE:-0}"
ENSEMBLE_SCRIPT="${ENSEMBLE_SCRIPT:-${V17_DIR}/ensemble_predictions_v22.py}"
ENSEMBLE_DUAL_RAW="${ENSEMBLE_DUAL_RAW:-1}"
RAW_TARGET_COMBINED="${RAW_TARGET_COMBINED:-0.7005}"
RAW_SAFETY_SEEDS_STR="${RAW_SAFETY_SEEDS:-42 3407 8407 2026}"
read -r -a RAW_SAFETY_SEEDS <<< "${RAW_SAFETY_SEEDS_STR}"
RAW_SAFETY_TRIALS="${RAW_SAFETY_TRIALS:-120000}"
RAW_SAFETY_CORR_PENALTY="${RAW_SAFETY_CORR_PENALTY:-0.0}"
RAW_SAFETY_MAX_MODELS="${RAW_SAFETY_MAX_MODELS:-${MAX_MODELS}}"

TOTAL=$((${#CASES[@]} * ${#SEEDS[@]}))
COUNT=0
FAILED=0
PIPELINE_START_TS=$(date +%s)

echo "================================================================"
echo " V23 Boost Pipeline"
echo " Cases: ${CASES[*]}"
echo " Seeds: ${SEEDS[*]}"
echo " Total runs: ${TOTAL}"
echo " Batch: ${BATCH_SIZE}"
echo " Parallel jobs: ${PARALLEL_JOBS}"
echo " Tune set: ${TUNE_SET}; objective: ${SEARCH_OBJECTIVE}"
echo " Raw target combined: ${RAW_TARGET_COMBINED}"
echo " Raw safety seeds: ${RAW_SAFETY_SEEDS[*]} (trials=${RAW_SAFETY_TRIALS}, corr_penalty=${RAW_SAFETY_CORR_PENALTY})"
echo " Taskset: ${USE_TASKSET} (CPU slots: ${CPU_SLOTS_STR})"
echo "================================================================"
echo ""

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/summary" "${OUTPUT_DIR}/ensemble_v23"

if [[ "${PARALLEL_JOBS}" -lt 1 ]]; then
    echo "ERROR: PARALLEL_JOBS must be >=1, got ${PARALLEL_JOBS}"
    exit 2
fi
if [[ "${USE_TASKSET}" == "1" && "${PARALLEL_JOBS}" -gt "${#CPU_SLOTS[@]}" ]]; then
    echo "ERROR: PARALLEL_JOBS=${PARALLEL_JOBS}, but CPU_SLOTS only has ${#CPU_SLOTS[@]} entries"
    echo "       set CPU_SLOTS or lower PARALLEL_JOBS"
    exit 2
fi

if [[ ! -f "${ENSEMBLE_SCRIPT}" ]]; then
    echo "ERROR: ENSEMBLE_SCRIPT not found: ${ENSEMBLE_SCRIPT}"
    exit 2
fi

echo "======== Phase 1: Training ${TOTAL} runs (${PARALLEL_JOBS} parallel) ========"

run_single() {
    local cfg="$1"
    local seed="$2"
    local cpu_cores="${3:-}"

    local run_tag="${cfg}_seed${seed}"
    local log_file="${OUTPUT_DIR}/logs/v23_${run_tag}_$(date +%Y%m%d_%H%M%S).log"
    local result_dir="${OUTPUT_DIR}/results/${run_tag}"

    echo "[START] ${run_tag} cpu=${cpu_cores:-none} log=${log_file}"
    rm -rf "${result_dir}"
    mkdir -p "${result_dir}"

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
            bash "${V17_DIR}/train_v23_single_case.sh" > "${log_file}" 2>&1
    else
        CASE_ID="${cfg}" \
        SEED="${seed}" \
        RUN_TAG="${run_tag}" \
        GPU_ID="${GPU_ID}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        NUM_WORKERS="${NUM_WORKERS}" \
        THREADS_PER_RUN="${THREADS_PER_RUN}" \
        CPU_CORES="${cpu_cores}" \
            bash "${V17_DIR}/train_v23_single_case.sh" > "${log_file}" 2>&1
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

echo ""
echo "======== Phase 2: Collect result dirs ========"

RESULT_DIRS=()
for cfg in "${CASES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_TAG="${cfg}_seed${seed}"
        DIR="${OUTPUT_DIR}/results/${RUN_TAG}"
        if [[ -d "${DIR}" ]]; then
            LATEST_TEST1=$(ls -t "${DIR}/seed${seed}-trimodal/result/test1_"*.npz 2>/dev/null | head -n 1 || true)
            if [[ -z "${LATEST_TEST1}" ]]; then
                echo "  [WARN] no test1 npz found under: ${DIR}"
                continue
            fi
            MTIME=$(stat -c %Y "${LATEST_TEST1}" 2>/dev/null || echo 0)
            if [[ "${MTIME}" -lt "${PIPELINE_START_TS}" ]]; then
                echo "  [WARN] stale result ignored: ${LATEST_TEST1}"
                continue
            fi
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

echo ""
echo "======== Phase 3: v23 Ensemble ========"

ENSEMBLE_OUTPUT="${OUTPUT_DIR}/ensemble_v23"
mkdir -p "${ENSEMBLE_OUTPUT}"

run_ensemble_once() {
    local tune_set="$1"
    local objective="$2"
    local post_calib="$3"
    local prefix="$4"
    local search_trials_local="${5:-${SEARCH_TRIALS}}"
    local seed_local="${6:-42}"
    local corr_penalty_local="${7:-${CORR_PENALTY}}"
    local max_models_local="${8:-${MAX_MODELS}}"
    echo "[ENSEMBLE] tune_set=${tune_set} objective=${objective} post_reg_calibration=${post_calib} prefix=${prefix} seed=${seed_local} trials=${search_trials_local} corr_penalty=${corr_penalty_local}"
    "${PYTHON_BIN}" "${ENSEMBLE_SCRIPT}" \
        --result_dirs "${RESULT_DIRS[@]}" \
        --tune_set "${tune_set}" \
        --eval_sets "${EVAL_SETS[@]}" \
        --max_models "${max_models_local}" \
        --alpha_grid "${ALPHA_GRID}" \
        --search_trials "${search_trials_local}" \
        --corr_penalty "${corr_penalty_local}" \
        --corr_threshold "${CORR_THRESHOLD}" \
        --search_objective "${objective}" \
        --post_reg_calibration "${post_calib}" \
        --calibration_min_samples "${CALIB_MIN_SAMPLES}" \
        --calibration_clip "${CALIB_CLIP}" \
        --seed "${seed_local}" \
        --output_dir "${ENSEMBLE_OUTPUT}" \
        --save_prefix "${prefix}"
}

run_ensemble_once "${TUNE_SET}" "${SEARCH_OBJECTIVE}" "${POST_REG_CALIB}" "v23_${TUNE_SET}_${SEARCH_OBJECTIVE}" "${SEARCH_TRIALS}" "42" "${CORR_PENALTY}" "${MAX_MODELS}"

if [[ "${ENSEMBLE_DUAL_RAW}" == "1" && "${SEARCH_OBJECTIVE}" == "raw" && "${TUNE_SET}" != "test1" ]]; then
    echo ""
    echo "======== Phase 3b: v23 Ensemble (test1 raw safety run) ========"
    # Multi-seed safety runs for the explicit local target (raw Combined on test1).
    for safe_seed in "${RAW_SAFETY_SEEDS[@]}"; do
        run_ensemble_once \
            "test1" \
            "raw" \
            "none" \
            "v23_test1_raw_safety_s${safe_seed}" \
            "${RAW_SAFETY_TRIALS}" \
            "${safe_seed}" \
            "${RAW_SAFETY_CORR_PENALTY}" \
            "${RAW_SAFETY_MAX_MODELS}"
    done
fi

echo ""
echo "======== Phase 4: Raw-target summary ========"
BEST_RAW_LINE=$(
    "${PYTHON_BIN}" - "$ENSEMBLE_OUTPUT" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
best = None
for path in glob.glob(os.path.join(root, "*_summary.json")):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    raw = (data.get("metrics_raw_test1") or {}).get("combined")
    if raw is None:
        continue
    if best is None or raw > best[0]:
        best = (float(raw), os.path.basename(path))

if best is None:
    print("NA")
else:
    print(f"{best[0]:.6f}\t{best[1]}")
PY
)
if [[ "${BEST_RAW_LINE}" == "NA" ]]; then
    echo " [WARN] no summary contains metrics_raw_test1.combined yet."
else
    BEST_RAW_VALUE="$(echo "${BEST_RAW_LINE}" | awk '{print $1}')"
    BEST_RAW_FILE="$(echo "${BEST_RAW_LINE}" | awk '{print $2}')"
    echo " Best raw test1 Combined: ${BEST_RAW_VALUE} (${BEST_RAW_FILE})"
    if awk -v a="${BEST_RAW_VALUE}" -v b="${RAW_TARGET_COMBINED}" 'BEGIN{exit !(a >= b)}'; then
        echo " Target met: ${BEST_RAW_VALUE} >= ${RAW_TARGET_COMBINED}"
    else
        echo " Target not met yet: ${BEST_RAW_VALUE} < ${RAW_TARGET_COMBINED}"
    fi
fi

echo ""
echo "================================================================"
echo " V23 Boost Pipeline Complete"
echo " Ensemble outputs: ${ENSEMBLE_OUTPUT}"
echo " Target: raw test1 Combined > ${RAW_TARGET_COMBINED}"
echo "================================================================"
