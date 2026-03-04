#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
V17_DIR="${MERBENCH_ROOT}/attention_robust_v17"
OUTPUT_ROOT="${V17_DIR}/outputs/ablation_v24"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-8407}"
CASE_ID="${CASE_ID:-a9_v14_g2_mseplus}"
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
    a4_mse_push)
        # Single-model score push:
        # keep v14 prior + uncertainty backbone, but bias toward lower valence MSE
        # while preserving classification with stronger smoothing and TTA.
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_UNCERT[@]}"
            "${ARGS_PRIOR[@]}"
            --dropout=0.31
            --modality_dropout=0.12
            --modality_dropout_warmup=14
            --lr=3.5e-4
            --cls_loss_type=label_smoothing
            --label_smoothing=0.08
            --emo_loss_weight=1.08
            --val_loss_weight=1.30
            --reg_stage2_start_epoch=30
            --huber_beta=0.7
            --valence_consistency_weight=0.12
            --valence_center_reg_weight=0.006
            --valence_prior_hidden_dim=128
            --valence_prior_gate_dropout=0.05
            --emotion_group_calibration_min_samples=10
            --valence_calibration_clip=3.0
            --use_reg_head_finetune
            --reg_finetune_epochs=24
            --reg_finetune_lr=7e-5
            --reg_finetune_patience=8
            --reg_finetune_lr_patience=4
            --reg_finetune_min_delta=3e-5
            --tta_passes=5
        )
        ;;
    a5_v14_mse_anchor)
        # MSE-focused v14 anchor: stronger regression phase + wider attention heads.
        MODEL_NAME="attention_robust_v14"
        EPOCHS_DEFAULT="132"
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_UNCERT[@]}"
            "${ARGS_PRIOR[@]}"
            "${ARGS_REGFT[@]}"
            --hidden_dim=160
            --num_attention_heads=8
            --dropout=0.30
            --modality_dropout=0.11
            --modality_dropout_warmup=12
            --lr=3.4e-4
            --label_smoothing=0.06
            --emo_loss_weight=1.00
            --val_loss_weight=1.38
            --reg_stage2_start_epoch=28
            --huber_beta=0.65
            --feature_noise_std=0.010
            --feature_noise_prob=0.15
            --valence_consistency_weight=0.13
            --valence_center_reg_weight=0.0065
            --reg_finetune_epochs=28
            --reg_finetune_lr=6e-5
            --reg_finetune_patience=10
            --reg_finetune_lr_patience=5
            --reg_finetune_min_delta=2.5e-5
            --emotion_group_calibration_min_samples=8
            --valence_calibration_clip=3.0
            --tta_passes=5
        )
        ;;
    a6_v14_no_uncert_wide)
        # Remove uncertainty weighting and increase width for a cleaner valence fit.
        MODEL_NAME="attention_robust_v14"
        EPOCHS_DEFAULT="130"
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_PRIOR[@]}"
            "${ARGS_REGFT[@]}"
            --hidden_dim=192
            --num_attention_heads=6
            --dropout=0.30
            --modality_dropout=0.11
            --modality_dropout_warmup=12
            --lr=3.35e-4
            --label_smoothing=0.06
            --emo_loss_weight=1.02
            --val_loss_weight=1.34
            --reg_stage2_start_epoch=26
            --huber_beta=0.70
            --feature_noise_std=0.010
            --feature_noise_prob=0.14
            --valence_consistency_weight=0.12
            --valence_center_reg_weight=0.006
            --reg_finetune_epochs=24
            --reg_finetune_lr=7e-5
            --reg_finetune_patience=8
            --reg_finetune_lr_patience=4
            --reg_finetune_min_delta=3e-5
            --emotion_group_calibration_min_samples=8
            --valence_calibration_clip=3.0
            --tta_passes=5
        )
        ;;
    a7_v2_reg_hifi)
        # V2 architecture keeps strong F1; tune it to reduce MSE.
        MODEL_NAME="attention_robust_v2"
        EPOCHS_DEFAULT="132"
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_UNCERT[@]}"
            "${ARGS_PRIOR[@]}"
            "${ARGS_REGFT[@]}"
            --hidden_dim=128
            --num_attention_heads=4
            --dropout=0.33
            --modality_dropout=0.14
            --modality_dropout_warmup=18
            --lr=3.8e-4
            --label_smoothing=0.09
            --emo_loss_weight=1.12
            --val_loss_weight=1.16
            --reg_stage2_start_epoch=42
            --huber_beta=0.80
            --feature_noise_std=0.015
            --feature_noise_prob=0.20
            --feature_noise_warmup=12
            --valence_consistency_weight=0.08
            --valence_center_reg_weight=0.004
            --valence_prior_hidden_dim=96
            --valence_prior_gate_dropout=0.08
            --reg_finetune_epochs=18
            --reg_finetune_lr=8e-5
            --reg_finetune_patience=7
            --reg_finetune_lr_patience=3
            --reg_finetune_min_delta=4e-5
            --emotion_group_calibration_min_samples=12
            --valence_calibration_clip=3.0
            --tta_passes=5
        )
        ;;
    a8_v15_deep_mse)
        # V15 architecture with deeper hidden state and MSE-heavy schedule.
        MODEL_NAME="attention_robust_v15"
        EPOCHS_DEFAULT="128"
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_UNCERT[@]}"
            "${ARGS_PRIOR[@]}"
            "${ARGS_REGFT[@]}"
            --hidden_dim=192
            --num_attention_heads=6
            --dropout=0.29
            --modality_dropout=0.10
            --modality_dropout_warmup=12
            --lr=3.3e-4
            --label_smoothing=0.06
            --emo_loss_weight=1.02
            --val_loss_weight=1.34
            --reg_stage2_start_epoch=26
            --huber_beta=0.70
            --feature_noise_std=0.010
            --feature_noise_prob=0.15
            --valence_consistency_weight=0.11
            --valence_center_reg_weight=0.0055
            --reg_finetune_epochs=22
            --reg_finetune_lr=7e-5
            --reg_finetune_patience=8
            --reg_finetune_lr_patience=4
            --reg_finetune_min_delta=3e-5
            --emotion_group_calibration_min_samples=8
            --valence_calibration_clip=3.2
            --tta_passes=5
        )
        ;;
    a9_v14_g2_mseplus)
        # Round-2 anchor from g2: keep strong F1 while tightening valence fit.
        MODEL_NAME="attention_robust_v14"
        EPOCHS_DEFAULT="132"
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_UNCERT[@]}"
            "${ARGS_PRIOR[@]}"
            "${ARGS_REGFT[@]}"
            --hidden_dim=160
            --num_attention_heads=6
            --dropout=0.31
            --modality_dropout=0.10
            --modality_dropout_warmup=12
            --lr=3.45e-4
            --label_smoothing=0.055
            --emo_loss_weight=0.98
            --val_loss_weight=1.44
            --reg_stage2_start_epoch=24
            --huber_beta=0.68
            --feature_noise_std=0.009
            --feature_noise_prob=0.13
            --valence_consistency_weight=0.16
            --valence_center_reg_weight=0.008
            --reg_finetune_epochs=30
            --reg_finetune_lr=6e-5
            --reg_finetune_patience=10
            --reg_finetune_lr_patience=5
            --reg_finetune_min_delta=2e-5
            --emotion_group_calibration_min_samples=6
            --valence_calibration_clip=2.8
            --tta_passes=5
        )
        ;;
    a10_v2_s1_bridge)
        # Round-2 hybrid from s1: preserve strong discrete signal while reducing MSE.
        MODEL_NAME="attention_robust_v2"
        EPOCHS_DEFAULT="136"
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_UNCERT[@]}"
            "${ARGS_PRIOR[@]}"
            "${ARGS_REGFT[@]}"
            --hidden_dim=160
            --num_attention_heads=4
            --dropout=0.30
            --modality_dropout=0.12
            --modality_dropout_warmup=14
            --lr=3.45e-4
            --label_smoothing=0.08
            --emo_loss_weight=1.08
            --val_loss_weight=1.22
            --reg_stage2_start_epoch=52
            --huber_beta=0.75
            --feature_noise_std=0.010
            --feature_noise_prob=0.15
            --valence_consistency_weight=0.11
            --valence_center_reg_weight=0.005
            --valence_prior_hidden_dim=112
            --valence_prior_gate_dropout=0.07
            --reg_finetune_epochs=22
            --reg_finetune_lr=7e-5
            --reg_finetune_patience=8
            --reg_finetune_lr_patience=4
            --reg_finetune_min_delta=3e-5
            --emotion_group_calibration_min_samples=10
            --valence_calibration_clip=3.0
            --tta_passes=5
        )
        ;;
    a11_v15_regdeep)
        # Round-2 deep regression for v15: widen hidden state and tighten calibration.
        MODEL_NAME="attention_robust_v15"
        EPOCHS_DEFAULT="132"
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_UNCERT[@]}"
            "${ARGS_PRIOR[@]}"
            "${ARGS_REGFT[@]}"
            --hidden_dim=224
            --num_attention_heads=8
            --dropout=0.28
            --modality_dropout=0.10
            --modality_dropout_warmup=12
            --lr=3.2e-4
            --label_smoothing=0.05
            --emo_loss_weight=1.00
            --val_loss_weight=1.40
            --reg_stage2_start_epoch=22
            --huber_beta=0.65
            --feature_noise_std=0.009
            --feature_noise_prob=0.12
            --valence_consistency_weight=0.14
            --valence_center_reg_weight=0.007
            --reg_finetune_epochs=26
            --reg_finetune_lr=6e-5
            --reg_finetune_patience=9
            --reg_finetune_lr_patience=4
            --reg_finetune_min_delta=2.5e-5
            --emotion_group_calibration_min_samples=6
            --valence_calibration_clip=2.7
            --tta_passes=5
        )
        ;;
    a12_v14_mse_hard)
        # Round-2 hard MSE push without uncertainty balancing.
        MODEL_NAME="attention_robust_v14"
        EPOCHS_DEFAULT="134"
        EXTRA_ARGS=(
            "${COMMON_ARGS[@]}"
            "${ARGS_PRIOR[@]}"
            "${ARGS_REGFT[@]}"
            --hidden_dim=192
            --num_attention_heads=8
            --dropout=0.29
            --modality_dropout=0.10
            --modality_dropout_warmup=10
            --lr=3.25e-4
            --label_smoothing=0.05
            --emo_loss_weight=0.96
            --val_loss_weight=1.52
            --reg_stage2_start_epoch=20
            --huber_beta=0.60
            --feature_noise_std=0.008
            --feature_noise_prob=0.10
            --valence_consistency_weight=0.17
            --valence_center_reg_weight=0.009
            --reg_finetune_epochs=32
            --reg_finetune_lr=5.5e-5
            --reg_finetune_patience=10
            --reg_finetune_lr_patience=5
            --reg_finetune_min_delta=2e-5
            --emotion_group_calibration_min_samples=6
            --valence_calibration_clip=2.6
            --tta_passes=5
        )
        ;;
    *)
        echo "Unknown CASE_ID=${CASE_ID}, expected one of: a0_base a1_no_prior a2_no_uncertainty a3_no_regft a4_mse_push a5_v14_mse_anchor a6_v14_no_uncert_wide a7_v2_reg_hifi a8_v15_deep_mse a9_v14_g2_mseplus a10_v2_s1_bridge a11_v15_regdeep a12_v14_mse_hard"
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
