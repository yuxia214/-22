#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
V17_DIR="${MERBENCH_ROOT}/attention_robust_v17"
OUTPUT_DIR="${V17_DIR}/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

CASES_STR="${CASES:-b1 b2 b3 b4 b5 b6}"
SEEDS_STR="${SEEDS:-8407}"
read -r -a CASES <<< "${CASES_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"

TUNE_SET="${TUNE_SET:-test1}"
EVAL_SETS_STR="${EVAL_SETS:-test1 test2 test3}"
read -r -a EVAL_SETS <<< "${EVAL_SETS_STR}"

MAX_MODELS="${MAX_MODELS:-8}"
SEARCH_TRIALS="${SEARCH_TRIALS:-8000}"
ALPHA_GRID="${ALPHA_GRID:-0.05,0.10,0.15,0.20,0.30,0.40,0.50}"
CORR_PENALTY="${CORR_PENALTY:-0.0}"
CORR_THRESHOLD="${CORR_THRESHOLD:-0.985}"

RESULT_DIRS=()
for cfg in "${CASES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        dir="${OUTPUT_DIR}/results/${cfg}_seed${seed}"
        if [[ -d "${dir}" ]]; then
            RESULT_DIRS+=("${dir}")
        else
            echo "[WARN] missing result dir: ${dir}"
        fi
    done
done

if [[ ${#RESULT_DIRS[@]} -lt 2 ]]; then
    echo "ERROR: need at least 2 result dirs, got ${#RESULT_DIRS[@]}"
    exit 1
fi

OUT_DIR="${OUTPUT_DIR}/ensemble_v19"
mkdir -p "${OUT_DIR}"

echo "================================================================"
echo " v19 Ensemble"
echo " tune_set: ${TUNE_SET}"
echo " eval_sets: ${EVAL_SETS[*]}"
echo " result_dirs: ${#RESULT_DIRS[@]}"
echo "================================================================"

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
    --output_dir "${OUT_DIR}" \
    --save_prefix "v19_${TUNE_SET}"

echo "done. outputs: ${OUT_DIR}"
