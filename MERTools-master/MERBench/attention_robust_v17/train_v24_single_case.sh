#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
V17_DIR="${MERBENCH_ROOT}/attention_robust_v17"
OUTPUT_DIR="${V17_DIR}/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-8407}"
CASE_ID="${CASE_ID:-g1}"
RUN_TAG="${RUN_TAG:-${CASE_ID}}"

THREADS_PER_RUN="${THREADS_PER_RUN:-2}"
CPU_CORES="${CPU_CORES:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

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
EPOCHS_DEFAULT="125"

BASE_ARGS=(
    --hidden_dim=128
    --dropout=0.33
    --kl_weight=5e-4
    --recon_weight=0.012
    --cross_kl_weight=0.001
    --fusion_temperature=0.30
    --modality_dropout=0.15
    --use_modality_dropout
    --modality_dropout_warmup=18
    --lr=3.7e-4
    --l2=5e-5
    --early_stopping_patience=36
    --lr_patience=8
    --lr_factor=0.5
    --reg_loss_type=smoothl1
    --huber_beta=0.8
    --use_uncertainty_weighted_mt
    --mt_init_log_var_cls=0.0
    --mt_init_log_var_reg=0.0
    --use_valence_prior
    --init_prior_from_fold_train
    --valence_consistency_weight=0.10
    --valence_center_reg_weight=0.0050
    --valence_prior_hidden_dim=96
    --valence_prior_gate_dropout=0.08
    --feature_noise_std=0.012
    --feature_noise_prob=0.18
    --feature_noise_warmup=10
    --use_proxy_attention
    --num_attention_heads=4
)

EXTRA_ARGS=()
case "${CASE_ID}" in
    g1)
        # Regression-first branch: pull down raw MSE.
        MODEL_NAME="attention_robust_v2"
        EPOCHS_DEFAULT="134"
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --hidden_dim=160
            --dropout=0.31
            --modality_dropout=0.12
            --modality_dropout_warmup=14
            --lr=3.5e-4
            --cls_loss_type=label_smoothing
            --label_smoothing=0.05
            --emo_loss_weight=0.92
            --val_loss_weight=1.52
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=34
            --reg_loss_type=smoothl1
            --huber_beta=0.7
            --use_reg_head_finetune
            --reg_finetune_epochs=20
            --reg_finetune_lr=7e-5
            --reg_finetune_patience=6
            --reg_finetune_lr_patience=3
            --reg_finetune_min_delta=4e-5
            --use_valence_calibration
            --valence_calibration_clip=4.5
        )
        ;;
    g2)
        # Bias-aware regression branch: class-dependent valence bias correction.
        MODEL_NAME="attention_robust_v14"
        EPOCHS_DEFAULT="130"
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --hidden_dim=160
            --dropout=0.32
            --modality_dropout=0.13
            --modality_dropout_warmup=15
            --lr=3.6e-4
            --cls_loss_type=label_smoothing
            --label_smoothing=0.05
            --emo_loss_weight=0.95
            --val_loss_weight=1.46
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=36
            --valence_consistency_weight=0.14
            --valence_center_reg_weight=0.007
            --valence_prior_hidden_dim=128
            --valence_prior_gate_dropout=0.05
            --use_reg_head_finetune
            --reg_finetune_epochs=16
            --reg_finetune_lr=8e-5
            --reg_finetune_patience=5
            --reg_finetune_lr_patience=3
            --reg_finetune_min_delta=4e-5
            --use_emotion_group_calibration
            --emotion_group_calibration_min_samples=6
            --valence_calibration_clip=4.5
        )
        ;;
    g3)
        # Diverse robust branch: keep regression diversity, avoid all branches collapsing.
        MODEL_NAME="attention_robust_v15"
        EPOCHS_DEFAULT="126"
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --hidden_dim=128
            --dropout=0.34
            --modality_dropout=0.16
            --modality_dropout_warmup=18
            --lr=3.6e-4
            --cls_loss_type=label_smoothing
            --label_smoothing=0.07
            --emo_loss_weight=1.00
            --val_loss_weight=1.34
            --reg_loss_type=smoothl1
            --huber_beta=0.9
            --reg_loss_type_stage2=mse
            --reg_stage2_start_epoch=42
            --use_reg_head_finetune
            --reg_finetune_epochs=14
            --reg_finetune_lr=8e-5
            --reg_finetune_patience=5
            --reg_finetune_lr_patience=3
            --reg_finetune_min_delta=5e-5
            --use_valence_calibration
            --valence_calibration_clip=4.5
            --use_swa
            --swa_start_epoch=80
            --swa_lr=2.0e-4
        )
        ;;
    g4)
        # Classification anchor: keep F1 stable while ensemble pushes MSE down.
        MODEL_NAME="attention_robust_v8"
        EPOCHS_DEFAULT="124"
        EXTRA_ARGS=(
            "${BASE_ARGS[@]}"
            --dropout=0.35
            --modality_dropout=0.20
            --modality_dropout_warmup=22
            --lr=3.9e-4
            --cls_loss_type=focal_label_smoothing
            --focal_gamma=1.7
            --label_smoothing=0.08
            --emo_loss_weight=1.22
            --val_loss_weight=1.05
            --reg_loss_type_stage2=none
            --use_mixup
            --mixup_alpha=0.20
            --use_valence_calibration
            --valence_calibration_clip=4.5
            --use_swa
            --swa_start_epoch=74
            --swa_lr=2.0e-4
        )
        ;;
    *)
        echo "Unknown CASE_ID=${CASE_ID}, expected one of: g1 g2 g3 g4"
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
