#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v16/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-8407}"
CASE_ID="${CASE_ID:-c1}"
RUN_TAG="${RUN_TAG:-${CASE_ID}}"

THREADS_PER_RUN="${THREADS_PER_RUN:-2}"
CPU_CORES="${CPU_CORES:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-20}"
EPOCHS="${EPOCHS:-90}"

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
cd "${MERBENCH_ROOT}"

DATASET="MER2023"
AUDIO_FEAT="chinese-hubert-large-UTT"
TEXT_FEAT="Baichuan-13B-Base-UTT"
VIDEO_FEAT="clip-vit-large-patch14-UTT"
FEAT_TYPE="utt"
SAVE_ROOT="${OUTPUT_DIR}/results/${RUN_TAG}/seed${SEED}"

BASE_ARGS=(
    --dropout=0.35
    --kl_weight=5e-4
    --recon_weight=0.012
    --cross_kl_weight=0.001
    --fusion_temperature=0.30
    --modality_dropout=0.18
    --use_modality_dropout
    --modality_dropout_warmup=35
    --lr=4e-4
    --l2=5e-5
    --early_stopping_patience=35
    --lr_patience=10
    --lr_factor=0.5
    --emo_loss_weight=1.1
    --val_loss_weight=1.0
    --reg_loss_type=smoothl1
    --huber_beta=0.8
    --use_valence_prior
    --valence_consistency_weight=0.08
    --valence_center_reg_weight=0.005
)

EXTRA_ARGS=()
case "${CASE_ID}" in
    c1)
        # C1: baseline_ref (f11-style)
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
        )
        ;;
    c2)
        # C2: baseline + emotion-group calibration
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --use_emotion_group_calibration
            --emotion_group_calibration_min_samples=20
            --valence_calibration_clip=3.0
        )
        ;;
    c3)
        # C3: baseline + regression-head finetune
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --use_reg_head_finetune
            --reg_finetune_epochs=10
            --reg_finetune_lr=1e-4
            --reg_finetune_patience=3
            --reg_finetune_lr_patience=2
            --reg_finetune_min_delta=5e-5
        )
        ;;
    c4)
        # C4: noise-robust + calibration + reg-head finetune + TTA
        EXTRA_ARGS=(
            --dropout=0.35
            --kl_weight=5e-4
            --recon_weight=0.012
            --cross_kl_weight=0.001
            --fusion_temperature=0.34
            --modality_dropout=0.24
            --use_modality_dropout
            --modality_dropout_warmup=30
            --lr=4e-4
            --l2=5e-5
            --early_stopping_patience=35
            --lr_patience=10
            --lr_factor=0.5
            --emo_loss_weight=1.1
            --val_loss_weight=1.15
            --reg_loss_type=smoothl1
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=35
            --huber_beta=0.8
            --use_valence_prior
            --valence_consistency_weight=0.08
            --valence_center_reg_weight=0.005
            --use_emotion_group_calibration
            --emotion_group_calibration_min_samples=20
            --valence_calibration_clip=3.0
            --use_reg_head_finetune
            --reg_finetune_epochs=10
            --reg_finetune_lr=1e-4
            --reg_finetune_patience=3
            --reg_finetune_lr_patience=2
            --reg_finetune_min_delta=5e-5
            --tta_passes=5
            --tta_use_train_mode
        )
        ;;
    *)
        echo "Unknown CASE_ID=${CASE_ID}, expected one of: c1 c2 c3 c4"
        exit 2
        ;;
esac

echo "run_tag=${RUN_TAG}"
echo "case_id=${CASE_ID}"
echo "seed=${SEED} gpu=${GPU_ID} batch=${BATCH_SIZE} workers=${NUM_WORKERS} cpu_cores=${CPU_CORES:-none}"
echo "save_root=${SAVE_ROOT}"

run_cmd "${PYTHON_BIN}" attention_robust_v16/launch_main_robust_seeded.py \
    --seed="${SEED}" \
    --main_robust="${MERBENCH_ROOT}/main-robust-v16.py" \
    --model=attention_robust_v2 \
    --dataset="${DATASET}" \
    --feat_type="${FEAT_TYPE}" \
    --audio_feature="${AUDIO_FEAT}" \
    --text_feature="${TEXT_FEAT}" \
    --video_feature="${VIDEO_FEAT}" \
    --save_root="${SAVE_ROOT}" \
    --hidden_dim=128 \
    --use_vae \
    --use_proxy_attention \
    --num_attention_heads=4 \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --gpu="${GPU_ID}" \
    --epochs="${EPOCHS}" \
    "${EXTRA_ARGS[@]}"
