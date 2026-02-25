#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
V17_DIR="${MERBENCH_ROOT}/attention_robust_v17"
OUTPUT_ROOT="${V17_DIR}/outputs/ablation_v24"

GPU_ID="${GPU_ID:-0}"
CASES_STR="${CASES:-a0_base a1_no_prior a2_no_uncertainty a3_no_regft}"
SEEDS_STR="${SEEDS:-8407}"
read -r -a CASES <<< "${CASES_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"

EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
THREADS_PER_RUN="${THREADS_PER_RUN:-2}"
USE_TASKSET="${USE_TASKSET:-0}"
CPU_SLOTS_STR="${CPU_SLOTS:-0-2 3-5 6-8 9-11}"
read -r -a CPU_SLOTS <<< "${CPU_SLOTS_STR}"

TOTAL=$((${#CASES[@]} * ${#SEEDS[@]}))
COUNT=0
FAILED=0

echo "================================================================"
echo " V24 Ablation4 Pipeline"
echo " Cases: ${CASES[*]}"
echo " Seeds: ${SEEDS[*]}"
echo " Total runs: ${TOTAL}"
echo " Batch: ${BATCH_SIZE}"
echo " Parallel jobs: ${PARALLEL_JOBS}"
echo " Taskset: ${USE_TASKSET} (CPU slots: ${CPU_SLOTS_STR})"
echo "================================================================"
echo ""

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/summary"

if [[ "${PARALLEL_JOBS}" -lt 1 ]]; then
    echo "ERROR: PARALLEL_JOBS must be >=1, got ${PARALLEL_JOBS}"
    exit 2
fi
if [[ "${USE_TASKSET}" == "1" && "${PARALLEL_JOBS}" -gt "${#CPU_SLOTS[@]}" ]]; then
    echo "ERROR: PARALLEL_JOBS=${PARALLEL_JOBS}, but CPU_SLOTS only has ${#CPU_SLOTS[@]} entries"
    exit 2
fi

run_single() {
    local cfg="$1"
    local seed="$2"
    local cpu_cores="${3:-}"

    local run_tag="${cfg}_seed${seed}"
    local log_file="${OUTPUT_ROOT}/logs/v24_ablation_${run_tag}_$(date +%Y%m%d_%H%M%S).log"
    local result_dir="${OUTPUT_ROOT}/results/${run_tag}"

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
            bash "${V17_DIR}/train_v24_ablation_single_case.sh" > "${log_file}" 2>&1
    else
        CASE_ID="${cfg}" \
        SEED="${seed}" \
        RUN_TAG="${run_tag}" \
        GPU_ID="${GPU_ID}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        NUM_WORKERS="${NUM_WORKERS}" \
        THREADS_PER_RUN="${THREADS_PER_RUN}" \
        CPU_CORES="${cpu_cores}" \
            bash "${V17_DIR}/train_v24_ablation_single_case.sh" > "${log_file}" 2>&1
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
echo "======== Ablation training complete: ${COUNT} runs, ${FAILED} failed, elapsed=${ELAPSED}s ========"

echo ""
echo "======== Test1 quick summary (f1/mse/combined) ========"
python - "$OUTPUT_ROOT" "${CASES_STR}" "${SEEDS_STR}" <<'PY'
import os
import re
import sys
from glob import glob

root = sys.argv[1]
cases = sys.argv[2].split()
seeds = sys.argv[3].split()
pat = re.compile(r"f1:([0-9.]+)_acc:[0-9.]+_val:([0-9.]+)_")
rows = []
for c in cases:
    for s in seeds:
        run_tag = f"{c}_seed{s}"
        d = os.path.join(root, "results", run_tag, f"seed{s}-trimodal", "result")
        files = sorted(glob(os.path.join(d, "test1_*.npz")), key=os.path.getmtime, reverse=True)
        if not files:
            rows.append((run_tag, None, None, None))
            continue
        name = os.path.basename(files[0])
        m = pat.search(name)
        if not m:
            rows.append((run_tag, None, None, None))
            continue
        f1 = float(m.group(1))
        mse = float(m.group(2))
        comb = f1 - 0.25 * mse
        rows.append((run_tag, f1, mse, comb))

rows.sort(key=lambda x: (-1e9 if x[3] is None else -x[3]))
for tag, f1, mse, comb in rows:
    if comb is None:
        print(f"{tag:28s}  N/A")
    else:
        print(f"{tag:28s}  f1={f1:.4f}  mse={mse:.4f}  combined={comb:.6f}")
PY

echo ""
echo "================================================================"
echo " V24 Ablation4 Pipeline Complete"
echo " Logs: ${OUTPUT_ROOT}/logs"
echo " Results: ${OUTPUT_ROOT}/results"
echo "================================================================"
