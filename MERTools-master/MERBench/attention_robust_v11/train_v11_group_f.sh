#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v11/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"

# CPU-lean preset for concurrent runs on one GPU.
THREADS_PER_RUN="${THREADS_PER_RUN:-1}"
CPU_CORES="${CPU_CORES:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS_PER_RUN}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS_PER_RUN}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS_PER_RUN}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${THREADS_PER_RUN}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
export OMP_PLACES="${OMP_PLACES:-cores}"

run_cmd() {
    if [[ -n "${CPU_CORES}" ]]; then
        taskset -c "${CPU_CORES}" "$@"
    else
        "$@"
    fi
}

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/summary"

DATASET="MER2023"
AUDIO_FEAT="chinese-hubert-large-UTT"
TEXT_FEAT="Baichuan-13B-Base-UTT"
VIDEO_FEAT="clip-vit-large-patch14-UTT"
FEAT_TYPE="utt"

BASE_SEED="${BASE_SEED:-8407}"
REPEATS="${REPEATS:-2}"
BATCH_SIZE="${BATCH_SIZE:-20}"
FAILED_RUNS=0

cd "${MERBENCH_ROOT}"

run_case() {
    local run_tag="$1"
    local seed="$2"
    shift 2

    local save_root="${OUTPUT_DIR}/results/${run_tag}/seed${seed}"
    echo ""
    echo "==================================================================="
    echo "run_tag=${run_tag} seed=${seed} gpu=${GPU_ID} batch=${BATCH_SIZE} workers=${NUM_WORKERS}"
    echo "save_root=${save_root}"
    echo "==================================================================="

    if run_cmd "${PYTHON_BIN}" attention_robust_v11/launch_main_robust_seeded.py \
        --seed="${seed}" \
        --main_robust="${MERBENCH_ROOT}/main-robust.py" \
        --model=attention_robust_v2 \
        --dataset="${DATASET}" \
        --feat_type="${FEAT_TYPE}" \
        --audio_feature="${AUDIO_FEAT}" \
        --text_feature="${TEXT_FEAT}" \
        --video_feature="${VIDEO_FEAT}" \
        --save_root="${save_root}" \
        --hidden_dim=128 \
        --use_vae \
        --use_proxy_attention \
        --num_attention_heads=4 \
        --batch_size="${BATCH_SIZE}" \
        --num_workers="${NUM_WORKERS}" \
        --gpu="${GPU_ID}" \
        "$@"; then
        return 0
    else
        local rc=$?
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "[WARN] run failed: run_tag=${run_tag} seed=${seed} rc=${rc}, continue."
        return 0
    fi
}

echo "==== V11 group F start (repeats=${REPEATS}, base_seed=${BASE_SEED}) ===="

for ((i=0; i<REPEATS; i++)); do
    seed=$((BASE_SEED + i))

    # f11: CPU-lean balanced profile
    run_case "f11_cpulean_balanced" "${seed}" \
        --dropout=0.35 \
        --kl_weight=0.0005 \
        --recon_weight=0.012 \
        --cross_kl_weight=0.001 \
        --fusion_temperature=0.30 \
        --modality_dropout=0.18 \
        --use_modality_dropout \
        --modality_dropout_warmup=35 \
        --lr=4e-4 \
        --l2=5e-5 \
        --epochs=130 \
        --early_stopping_patience=45 \
        --lr_patience=10 \
        --emo_loss_weight=1.1 \
        --val_loss_weight=1.0 \
        --reg_loss_type=smoothl1 \
        --huber_beta=0.8

    # f12: CPU-lean MER-MULTI preference profile
    run_case "f12_cpulean_multi" "${seed}" \
        --dropout=0.35 \
        --kl_weight=0.0004 \
        --recon_weight=0.010 \
        --cross_kl_weight=0.001 \
        --fusion_temperature=0.24 \
        --modality_dropout=0.12 \
        --use_modality_dropout \
        --modality_dropout_warmup=40 \
        --lr=4.5e-4 \
        --l2=5e-5 \
        --epochs=130 \
        --early_stopping_patience=45 \
        --lr_patience=10 \
        --emo_loss_weight=1.3 \
        --val_loss_weight=0.7 \
        --reg_loss_type=mse
done

run_cmd "${PYTHON_BIN}" attention_robust_v11/summarize_v11.py \
    --root "${OUTPUT_DIR}/results" \
    --tag_prefix "f1" \
    --out_csv "${OUTPUT_DIR}/summary/group_f_summary.csv"

if [[ "${FAILED_RUNS}" -gt 0 ]]; then
    echo "[WARN] group F finished with failed_runs=${FAILED_RUNS}"
fi
echo "==== V11 group F done ===="

