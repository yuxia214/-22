#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v17/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-8407}"
CASE_ID="${CASE_ID:-s1}"
RUN_TAG="${RUN_TAG:-${CASE_ID}}"

THREADS_PER_RUN="${THREADS_PER_RUN:-2}"
CPU_CORES="${CPU_CORES:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-20}"
EPOCHS="${EPOCHS:-130}"

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

# ============================================================
# Shared base args (v16 proven defaults + label smoothing)
# ============================================================
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
    --early_stopping_patience=45
    --lr_patience=10
    --lr_factor=0.5
    --reg_loss_type=smoothl1
    --huber_beta=0.8
    --use_valence_prior
    --init_prior_from_fold_train
    --valence_consistency_weight=0.06
    --valence_center_reg_weight=0.003
    --use_uncertainty_weighted_mt
    --use_emotion_group_calibration
    --emotion_group_calibration_min_samples=15
    --valence_calibration_clip=3.0
)

EXTRA_ARGS=()
case "${CASE_ID}" in
    # ===========================================================
    # Ensemble configs (s1-s4): designed for maximum diversity
    # ===========================================================
    s1)
        # S1: balanced + label smoothing (核心基线, 最稳健)
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --cls_loss_type=label_smoothing
            --label_smoothing=0.1
            --emo_loss_weight=1.15
            --val_loss_weight=1.0
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=70
        )
        ;;
    s2)
        # S2: F1-boost + SWA + label smoothing (偏重分类)
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --cls_loss_type=label_smoothing
            --label_smoothing=0.08
            --emo_loss_weight=1.25
            --val_loss_weight=0.9
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=70
            --use_swa
            --swa_start_epoch=80
            --swa_lr=2e-4
        )
        ;;
    s3)
        # S3: balanced + reg-head finetune (偏重回归质量)
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --cls_loss_type=label_smoothing
            --label_smoothing=0.1
            --emo_loss_weight=1.1
            --val_loss_weight=1.1
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=65
            --use_reg_head_finetune
            --reg_finetune_epochs=12
            --reg_finetune_lr=8e-5
            --reg_finetune_patience=4
            --reg_finetune_lr_patience=3
            --reg_finetune_min_delta=5e-5
        )
        ;;
    s4)
        # S4: focal+label_smoothing + cosine schedule (最大多样性)
        EXTRA_ARGS=(
            --dropout=0.32
            --kl_weight=5e-4
            --recon_weight=0.012
            --cross_kl_weight=0.001
            --fusion_temperature=0.30
            --modality_dropout=0.20
            --use_modality_dropout
            --modality_dropout_warmup=30
            --lr=5e-4
            --l2=5e-5
            --early_stopping_patience=45
            --lr_patience=10
            --lr_factor=0.5
            --reg_loss_type=smoothl1
            --huber_beta=0.8
            --use_valence_prior
            --init_prior_from_fold_train
            --valence_consistency_weight=0.06
            --valence_center_reg_weight=0.003
            --use_uncertainty_weighted_mt
            --use_emotion_group_calibration
            --emotion_group_calibration_min_samples=15
            --valence_calibration_clip=3.0
            --cls_loss_type=focal_label_smoothing
            --focal_gamma=1.5
            --label_smoothing=0.08
            --emo_loss_weight=1.2
            --val_loss_weight=1.0
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=70
            --lr_schedule=cosine
            --cosine_t0=25
            --cosine_t_mult=2
        )
        ;;
    # ===========================================================
    # Legacy configs (a1-c1): kept for reference
    # ===========================================================
    a1)
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --emo_loss_weight=1.2
            --val_loss_weight=1.0
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=70
        )
        ;;
    a2)
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --emo_loss_weight=1.2
            --val_loss_weight=1.0
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=70
            --use_swa
            --swa_start_epoch=85
            --swa_lr=2e-4
        )
        ;;
    b1)
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --emo_loss_weight=1.2
            --val_loss_weight=1.0
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=70
            --use_reg_head_finetune
            --reg_finetune_epochs=15
            --reg_finetune_lr=8e-5
            --reg_finetune_patience=4
            --reg_finetune_lr_patience=3
            --reg_finetune_min_delta=5e-5
            --tta_passes=5
            --tta_use_train_mode
        )
        ;;
    c1)
        EXTRA_ARGS=(
            --dropout=0.30
            --kl_weight=5e-4
            --recon_weight=0.012
            --cross_kl_weight=0.001
            --fusion_temperature=0.30
            --modality_dropout=0.18
            --use_modality_dropout
            --modality_dropout_warmup=35
            --lr=4e-4
            --l2=5e-5
            --early_stopping_patience=45
            --lr_patience=10
            --lr_factor=0.5
            --emo_loss_weight=1.35
            --val_loss_weight=0.85
            --reg_loss_type=smoothl1
            --huber_beta=0.8
            --use_uncertainty_weighted_mt
            --use_emotion_group_calibration
            --emotion_group_calibration_min_samples=15
            --valence_calibration_clip=3.0
        )
        ;;
    *)
        echo "Unknown CASE_ID=${CASE_ID}, expected: s1 s2 s3 s4 a1 a2 b1 c1"
        exit 2
        ;;
esac

echo "run_tag=${RUN_TAG}"
echo "case_id=${CASE_ID}"
echo "seed=${SEED} gpu=${GPU_ID} batch=${BATCH_SIZE} epochs=${EPOCHS}"
echo "save_root=${SAVE_ROOT}"

run_cmd "${PYTHON_BIN}" attention_robust_v17/launch_main_robust_seeded.py \
    --seed="${SEED}" \
    --main_robust="${MERBENCH_ROOT}/main-robust-v17.py" \
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
