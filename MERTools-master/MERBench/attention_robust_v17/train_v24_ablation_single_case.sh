#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
V17_DIR="${MERBENCH_ROOT}/attention_robust_v17"
OUTPUT_ROOT="${V17_DIR}/outputs/ablation_v24"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-8407}"
CASE_ID="${CASE_ID:-a0_base}"
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

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results"
cd "${MERBENCH_ROOT}"

DATASET="MER2023"
AUDIO_FEAT="chinese-hubert-large-UTT"
TEXT_FEAT="Baichuan-13B-Base-UTT"
VIDEO_FEAT="clip-vit-large-patch14-UTT"
FEAT_TYPE="utt"
MODEL_NAME="attention_robust_v14"
EPOCHS_DEFAULT="130"
SAVE_ROOT="${OUTPUT_ROOT}/results/${RUN_TAG}/seed${SEED}"

COMMON_ARGS=(
    --hidden_dim=160
    --dropout=0.32
    --kl_weight=5e-4
    --recon_weight=0.012
    --cross_kl_weight=0.001
    --fusion_temperature=0.30
    --modality_dropout=0.13
    --use_modality_dropout
    --modality_dropout_warmup=15
    --lr=3.6e-4
    --l2=5e-5
    --early_stopping_patience=36
    --lr_patience=8
    --lr_factor=0.5
    --cls_loss_type=label_smoothing
    --label_smoothing=0.05
    --emo_loss_weight=0.95
    --val_loss_weight=1.46
    --reg_loss_type=smoothl1
    --huber_beta=0.8
    --reg_loss_type_stage2=mse
    --reg_stage2_start_epoch=36
    --use_proxy_attention
    --num_attention_heads=4
    --feature_noise_std=0.012
    --feature_noise_prob=0.18
    --feature_noise_warmup=10
    --use_valence_calibration
    --valence_calibration_clip=4.5
    --use_emotion_group_calibration
    --emotion_group_calibration_min_samples=6
)

ARGS_UNCERT=(
    --use_uncertainty_weighted_mt
    --mt_init_log_var_cls=0.0
    --mt_init_log_var_reg=0.0
)

ARGS_PRIOR=(
    --use_valence_prior
    --init_prior_from_fold_train
    --valence_consistency_weight=0.14
    --valence_center_reg_weight=0.007
    --valence_prior_hidden_dim=128
    --valence_prior_gate_dropout=0.05
)

ARGS_REGFT=(
    --use_reg_head_finetune
    --reg_finetune_epochs=16
    --reg_finetune_lr=8e-5
    --reg_finetune_patience=5
    --reg_finetune_lr_patience=3
    --reg_finetune_min_delta=4e-5
)

EXTRA_ARGS=()
case "${CASE_ID}" in
    a0_base)
        # Baseline: full v24 g2 stack.
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_UNCERT[@]}"
            "${ARGS_PRIOR[@]}"
            "${ARGS_REGFT[@]}"
        )
        ;;
    a1_no_prior)
        # Ablation-1: remove emotion-guided valence prior branch.
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_UNCERT[@]}"
            "${ARGS_REGFT[@]}"
        )
        ;;
    a2_no_uncertainty)
        # Ablation-2: remove uncertainty-weighted multi-task balancing.
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_PRIOR[@]}"
            "${ARGS_REGFT[@]}"
        )
        ;;
    a3_no_regft)
        # Ablation-3: remove regression-head finetune stage.
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_UNCERT[@]}"
            "${ARGS_PRIOR[@]}"
        )
        ;;
    *)
        echo "Unknown CASE_ID=${CASE_ID}, expected one of: a0_base a1_no_prior a2_no_uncertainty a3_no_regft"
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
