#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
V17_DIR="${MERBENCH_ROOT}/attention_robust_v17"
OUTPUT_DIR="${V17_DIR}/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-8407}"
CASE_ID="${CASE_ID:-d1}"
RUN_TAG="${RUN_TAG:-${CASE_ID}}"

THREADS_PER_RUN="${THREADS_PER_RUN:-2}"
CPU_CORES="${CPU_CORES:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-20}"

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

MODEL_NAME="attention_robust_v2"
EPOCHS_DEFAULT="120"

BASE_ARGS=(
    --dropout=0.34
    --hidden_dim=128
    --kl_weight=5e-4
    --recon_weight=0.012
    --cross_kl_weight=0.001
    --fusion_temperature=0.30
    --modality_dropout=0.16
    --use_modality_dropout
    --modality_dropout_warmup=20
    --lr=4e-4
    --l2=5e-5
    --early_stopping_patience=34
    --lr_patience=8
    --lr_factor=0.5
    --reg_loss_type=smoothl1
    --huber_beta=0.8
    --use_uncertainty_weighted_mt
    --mt_init_log_var_cls=0.0
    --mt_init_log_var_reg=0.0
    --use_valence_prior
    --init_prior_from_fold_train
    --valence_consistency_weight=0.08
    --valence_center_reg_weight=0.004
    --valence_prior_hidden_dim=96
    --valence_prior_gate_dropout=0.10
    --use_proxy_attention
    --num_attention_heads=4
)

EXTRA_ARGS=()
case "${CASE_ID}" in
    d1)
        # Regression-oriented v2 branch: linear calibration + reg head finetune
        MODEL_NAME="attention_robust_v2"
        EPOCHS_DEFAULT="135"
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --cls_loss_type=label_smoothing
            --label_smoothing=0.06
            --emo_loss_weight=1.00
            --val_loss_weight=1.32
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=50
            --use_reg_head_finetune
            --reg_finetune_epochs=16
            --reg_finetune_lr=8e-5
            --reg_finetune_patience=5
            --reg_finetune_lr_patience=3
            --reg_finetune_min_delta=4e-5
            --use_valence_calibration
            --valence_calibration_clip=3.0
            --use_swa
            --swa_start_epoch=78
            --swa_lr=1.8e-4
            --tta_passes=3
        )
        ;;
    d2)
        # v13 branch: sample-wise prior gate + emotion-group calibration
        MODEL_NAME="attention_robust_v13"
        EPOCHS_DEFAULT="125"
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --hidden_dim=160
            --dropout=0.33
            --modality_dropout=0.18
            --modality_dropout_warmup=18
            --lr=3.5e-4
            --cls_loss_type=label_smoothing
            --label_smoothing=0.07
            --emo_loss_weight=1.03
            --val_loss_weight=1.26
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=46
            --use_reg_head_finetune
            --reg_finetune_epochs=12
            --reg_finetune_lr=9e-5
            --reg_finetune_patience=4
            --reg_finetune_lr_patience=3
            --reg_finetune_min_delta=5e-5
            --use_emotion_group_calibration
            --emotion_group_calibration_min_samples=12
            --valence_calibration_clip=3.0
            --use_swa
            --swa_start_epoch=70
            --swa_lr=1.8e-4
            --tta_passes=5
            --tta_use_train_mode
        )
        ;;
    d3)
        # F1 anchor branch: v8 + focal smoothing + stronger class emphasis
        MODEL_NAME="attention_robust_v8"
        EPOCHS_DEFAULT="120"
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --dropout=0.36
            --modality_dropout=0.22
            --modality_dropout_warmup=24
            --cls_loss_type=focal_label_smoothing
            --focal_gamma=1.7
            --label_smoothing=0.08
            --emo_loss_weight=1.28
            --val_loss_weight=0.92
            --reg_loss_type_stage2=none
            --use_mixup
            --mixup_alpha=0.25
            --use_swa
            --swa_start_epoch=72
            --swa_lr=2e-4
            --tta_passes=3
        )
        ;;
    d4)
        # Low-correlation branch: v9 with stronger corruption and quality-aware fusion
        MODEL_NAME="attention_robust_v9"
        EPOCHS_DEFAULT="120"
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --hidden_dim=160
            --dropout=0.35
            --modality_dropout=0.20
            --modality_dropout_warmup=20
            --lr=3.2e-4
            --cls_loss_type=label_smoothing
            --label_smoothing=0.08
            --emo_loss_weight=1.08
            --val_loss_weight=1.18
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=52
            --corruption_max_rate=0.55
            --corruption_warmup_epochs=20
            --double_mask_ratio=0.45
            --quality_weight=0.75
            --impute_loss_weight=0.14
            --consistency_emo_weight=0.06
            --consistency_val_weight=0.08
            --use_emotion_group_calibration
            --emotion_group_calibration_min_samples=10
            --valence_calibration_clip=3.0
            --use_swa
            --swa_start_epoch=68
            --swa_lr=1.8e-4
            --tta_passes=5
        )
        ;;
    *)
        echo "Unknown CASE_ID=${CASE_ID}, expected one of: d1 d2 d3 d4"
        exit 2
        ;;
esac

EPOCHS="${EPOCHS:-${EPOCHS_DEFAULT}}"

echo "run_tag=${RUN_TAG}"
echo "case_id=${CASE_ID}"
echo "model=${MODEL_NAME}"
echo "seed=${SEED} gpu=${GPU_ID} batch=${BATCH_SIZE} epochs=${EPOCHS}"
echo "save_root=${SAVE_ROOT}"

run_cmd "${PYTHON_BIN}" "${V17_DIR}/launch_main_robust_seeded.py" \
    --seed="${SEED}" \
    --main_robust="${MERBENCH_ROOT}/main-robust-v17.py" \
    --model="${MODEL_NAME}" \
    --dataset="${DATASET}" \
    --feat_type="${FEAT_TYPE}" \
    --audio_feature="${AUDIO_FEAT}" \
    --text_feature="${TEXT_FEAT}" \
    --video_feature="${VIDEO_FEAT}" \
    --save_root="${SAVE_ROOT}" \
    --use_vae \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --gpu="${GPU_ID}" \
    --epochs="${EPOCHS}" \
    "${EXTRA_ARGS[@]}"
