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

SHORT_CSV="${SHORT_CSV:-${OUTPUT_DIR}/summary/short_summary.csv}"
TOPK="${TOPK:-2}"
RANK_KEY="${RANK_KEY:-score_balanced_mean}"  # score_balanced_mean | multi_waf_mean
BASE_SEED="${BASE_SEED:-7407}"
REPEATS="${REPEATS:-6}"
FAILED_RUNS=0

cd "${MERBENCH_ROOT}"

if [[ ! -f "${SHORT_CSV}" ]]; then
    echo "Missing short summary: ${SHORT_CSV}"
    echo "Run: bash attention_robust_v11/train_v11_short.sh"
    exit 1
fi

SELECTED_TAGS=$("${PYTHON_BIN}" - <<PY
import csv
path = r"${SHORT_CSV}"
rank_key = r"${RANK_KEY}"
topk = int("${TOPK}")
rows = []
with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if not r.get("run_tag", "").startswith("c0"):
            continue
        try:
            score = float(r.get(rank_key, "") or "")
        except Exception:
            continue
        rows.append((score, r["run_tag"]))
rows.sort(reverse=True)
print(",".join([x[1] for x in rows[:topk]]))
PY
)

if [[ -z "${SELECTED_TAGS}" ]]; then
    echo "No tags selected from ${SHORT_CSV} with rank key ${RANK_KEY}"
    exit 1
fi

echo "Selected tags: ${SELECTED_TAGS}"

run_case() {
    local run_tag="$1"
    local seed="$2"
    shift 2

    local save_root="${OUTPUT_DIR}/results/promote_${run_tag}/seed${seed}"
    echo ""
    echo "==================================================================="
    echo "run_tag=promote_${run_tag} seed=${seed} gpu=${GPU_ID}"
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
        --batch_size=32 \
        --num_workers="${NUM_WORKERS}" \
        --epochs=140 \
        --gpu="${GPU_ID}" \
        "$@"; then
        return 0
    else
        local rc=$?
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "[WARN] run failed: run_tag=promote_${run_tag} seed=${seed} rc=${rc}, continue."
        return 0
    fi
}

run_by_tag() {
    local tag="$1"
    local seed="$2"
    case "${tag}" in
        c01_v2best_emoval)
            run_case "${tag}" "${seed}" \
                --dropout=0.35 --kl_weight=0.0005 --recon_weight=0.01 --cross_kl_weight=0.001 \
                --fusion_temperature=0.30 --modality_dropout=0.20 --use_modality_dropout --modality_dropout_warmup=35 \
                --lr=5e-4 --l2=5e-5 --early_stopping_patience=40 --lr_patience=10 \
                --emo_loss_weight=1.0 --val_loss_weight=1.0 --reg_loss_type=mse
            ;;
        c02_multi_focus)
            run_case "${tag}" "${seed}" \
                --dropout=0.35 --kl_weight=0.0005 --recon_weight=0.01 --cross_kl_weight=0.001 \
                --fusion_temperature=0.25 --modality_dropout=0.15 --use_modality_dropout --modality_dropout_warmup=40 \
                --lr=5e-4 --l2=5e-5 --early_stopping_patience=45 --lr_patience=12 \
                --emo_loss_weight=1.3 --val_loss_weight=0.7 --reg_loss_type=mse
            ;;
        c03_balanced_smoothl1)
            run_case "${tag}" "${seed}" \
                --dropout=0.35 --kl_weight=0.0005 --recon_weight=0.012 --cross_kl_weight=0.001 \
                --fusion_temperature=0.30 --modality_dropout=0.18 --use_modality_dropout --modality_dropout_warmup=35 \
                --lr=4e-4 --l2=5e-5 --early_stopping_patience=45 --lr_patience=12 \
                --emo_loss_weight=1.1 --val_loss_weight=1.0 --reg_loss_type=smoothl1 --huber_beta=0.8
            ;;
        c04_noise_guard)
            run_case "${tag}" "${seed}" \
                --dropout=0.35 --kl_weight=0.0007 --recon_weight=0.015 --cross_kl_weight=0.0012 \
                --fusion_temperature=0.35 --modality_dropout=0.25 --use_modality_dropout --modality_dropout_warmup=30 \
                --lr=4e-4 --l2=5e-5 --early_stopping_patience=45 --lr_patience=12 \
                --emo_loss_weight=1.0 --val_loss_weight=1.2 --reg_loss_type=smoothl1 --huber_beta=0.8
            ;;
        c05_clean_upper)
            run_case "${tag}" "${seed}" \
                --dropout=0.35 --kl_weight=0.0003 --recon_weight=0.008 --cross_kl_weight=0.0008 \
                --fusion_temperature=0.20 --no_modality_dropout \
                --lr=5e-4 --l2=5e-5 --early_stopping_patience=50 --lr_patience=12 \
                --emo_loss_weight=1.5 --val_loss_weight=0.5 --reg_loss_type=mse
            ;;
        *)
            echo "[WARN] unknown tag from short summary: ${tag}, skip."
            ;;
    esac
}

echo "==== V11 promote stage start (repeats=${REPEATS}, base_seed=${BASE_SEED}, topk=${TOPK}, rank=${RANK_KEY}) ===="

IFS=',' read -r -a TAG_ARRAY <<< "${SELECTED_TAGS}"
for ((i=0; i<REPEATS; i++)); do
    seed=$((BASE_SEED + i))
    for tag in "${TAG_ARRAY[@]}"; do
        run_by_tag "${tag}" "${seed}"
    done
done

run_cmd "${PYTHON_BIN}" attention_robust_v11/summarize_v11.py \
    --root "${OUTPUT_DIR}/results" \
    --tag_prefix "promote_" \
    --out_csv "${OUTPUT_DIR}/summary/promote_summary.csv"

if [[ "${FAILED_RUNS}" -gt 0 ]]; then
    echo "[WARN] promote stage finished with failed_runs=${FAILED_RUNS}"
fi
echo "==== V11 promote stage done ===="
