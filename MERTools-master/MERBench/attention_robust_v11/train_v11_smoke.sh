#!/bin/bash
set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v11/outputs"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-0}"
THREADS_PER_RUN="${THREADS_PER_RUN:-2}"
CPU_CORES="${CPU_CORES:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"

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

SEED="${SEED:-3407}"
RUN_TAG="v11_smoke_v2best"
SAVE_ROOT="${OUTPUT_DIR}/results/${RUN_TAG}/seed${SEED}"

cd "${MERBENCH_ROOT}"

echo "==== V11 smoke start ===="
echo "seed=${SEED} gpu=${GPU_ID} save_root=${SAVE_ROOT}"

run_cmd "${PYTHON_BIN}" attention_robust_v11/launch_main_robust_seeded.py \
    --seed="${SEED}" \
    --main_robust="${MERBENCH_ROOT}/main-robust.py" \
    --model=attention_robust_v2 \
    --dataset="${DATASET}" \
    --feat_type="${FEAT_TYPE}" \
    --audio_feature="${AUDIO_FEAT}" \
    --text_feature="${TEXT_FEAT}" \
    --video_feature="${VIDEO_FEAT}" \
    --save_root="${SAVE_ROOT}" \
    --hidden_dim=128 \
    --dropout=0.35 \
    --use_vae \
    --kl_weight=0.0005 \
    --recon_weight=0.01 \
    --cross_kl_weight=0.001 \
    --use_proxy_attention \
    --fusion_temperature=0.3 \
    --num_attention_heads=4 \
    --modality_dropout=0.2 \
    --use_modality_dropout \
    --modality_dropout_warmup=35 \
    --lr=5e-4 \
    --l2=5e-5 \
    --epochs=30 \
    --batch_size=32 \
    --num_workers="${NUM_WORKERS}" \
    --early_stopping_patience=8 \
    --lr_patience=4 \
    --emo_loss_weight=1.0 \
    --val_loss_weight=1.0 \
    --gpu="${GPU_ID}"

run_cmd "${PYTHON_BIN}" attention_robust_v11/summarize_v11.py \
    --root "${OUTPUT_DIR}/results" \
    --tag_prefix "v11_smoke_" \
    --out_csv "${OUTPUT_DIR}/summary/smoke_summary.csv"

echo "==== V11 smoke done ===="
