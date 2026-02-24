#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v12/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"

# CPU-lean defaults for fast screening.
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

SEED="${SEED:-8407}"
BATCH_SIZE="${BATCH_SIZE:-20}"
EPOCHS="${EPOCHS:-45}"
FAILED_RUNS=0

cd "${MERBENCH_ROOT}"

run_case() {
    local run_tag="$1"
    shift 1

    local save_root="${OUTPUT_DIR}/results/${run_tag}/seed${SEED}"
    echo ""
    echo "==================================================================="
    echo "run_tag=${run_tag} seed=${SEED} gpu=${GPU_ID} batch=${BATCH_SIZE} workers=${NUM_WORKERS}"
    echo "save_root=${save_root}"
    echo "==================================================================="

    if run_cmd "${PYTHON_BIN}" attention_robust_v12/launch_main_robust_seeded.py \
        --seed="${SEED}" \
        --main_robust="${MERBENCH_ROOT}/main-robust-v12.py" \
        --model=attention_robust_v12 \
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
        --dropout=0.35 \
        --kl_weight=0.0005 \
        --recon_weight=0.012 \
        --cross_kl_weight=0.001 \
        --fusion_temperature=0.30 \
        --modality_dropout=0.18 \
        --use_modality_dropout \
        --modality_dropout_warmup=20 \
        --lr=4e-4 \
        --l2=5e-5 \
        --epochs="${EPOCHS}" \
        --early_stopping_patience=12 \
        --lr_patience=4 \
        --lr_factor=0.5 \
        --emo_loss_weight=1.1 \
        --val_loss_weight=1.0 \
        --reg_loss_type=smoothl1 \
        --huber_beta=0.8 \
        "$@"; then
        return 0
    else
        local rc=$?
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "[WARN] run failed: run_tag=${run_tag} seed=${SEED} rc=${rc}, continue."
        return 0
    fi
}

echo "==== V12 Phase-A fast start (seed=${SEED}, epochs=${EPOCHS}) ===="

# a1: v2 + loss升级
run_case "a1_v2_lossup" \
    --use_uncertainty_weighted_mt \
    --mt_init_log_var_cls=0.0 \
    --mt_init_log_var_reg=0.0 \
    --no_valence_prior

# a2: v2 + prior
run_case "a2_v2_prior" \
    --no_contrastive \
    --use_valence_prior \
    --valence_consistency_weight=0.06 \
    --valence_center_reg_weight=0.003

# a3: v2 + 两者
run_case "a3_v2_both" \
    --use_uncertainty_weighted_mt \
    --mt_init_log_var_cls=0.0 \
    --mt_init_log_var_reg=0.0 \
    --use_valence_prior \
    --valence_consistency_weight=0.06 \
    --valence_center_reg_weight=0.003

# a4: v2 + 两者 + SWA
run_case "a4_v2_both_swa" \
    --use_uncertainty_weighted_mt \
    --mt_init_log_var_cls=0.0 \
    --mt_init_log_var_reg=0.0 \
    --use_valence_prior \
    --valence_consistency_weight=0.06 \
    --valence_center_reg_weight=0.003 \
    --use_swa \
    --swa_start_epoch=30 \
    --swa_lr=2e-4

run_cmd "${PYTHON_BIN}" attention_robust_v12/summarize_v12.py \
    --root "${OUTPUT_DIR}/results" \
    --tag_prefix "a" \
    --out_csv "${OUTPUT_DIR}/summary/phaseA_fast_summary.csv"

if [[ "${FAILED_RUNS}" -gt 0 ]]; then
    echo "[WARN] V12 phase-A fast finished with failed_runs=${FAILED_RUNS}"
fi
echo "==== V12 Phase-A fast done ===="
