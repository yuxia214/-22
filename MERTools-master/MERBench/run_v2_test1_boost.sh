#!/bin/bash
set -euo pipefail

# Focus: maximize MER2023 test1 (MER-MULTI) classification accuracy.
# Trade-off: some settings below may reduce test2 robustness.

DATASET="MER2023"
AUDIO_FEATURE="chinese-hubert-large-UTT"
TEXT_FEATURE="Baichuan-13B-Base-UTT"
VIDEO_FEATURE="clip-vit-large-patch14-UTT"
GPU=0

run_exp() {
  local name="$1"
  shift
  echo ""
  echo "==================== ${name} ===================="
  python -u main-robust.py \
    --model='attention_robust_v2' \
    --feat_type='utt' \
    --dataset="${DATASET}" \
    --audio_feature="${AUDIO_FEATURE}" \
    --text_feature="${TEXT_FEATURE}" \
    --video_feature="${VIDEO_FEATURE}" \
    --save_root="./saved_test1_boost/${name}" \
    "$@"
}

# Baseline from your best historical test1 run (acc=0.8345).
run_exp "v2_best_baseline" \
  --hidden_dim=128 \
  --dropout=0.35 \
  --use_vae \
  --kl_weight=0.0005 \
  --recon_weight=0.01 \
  --cross_kl_weight=0.001 \
  --use_proxy_attention \
  --fusion_temperature=0.3 \
  --modality_dropout=0.2 \
  --use_modality_dropout \
  --modality_dropout_warmup=35 \
  --lr=5e-4 \
  --l2=5e-5 \
  --epochs=120 \
  --early_stopping_patience=40 \
  --early_stopping_min_delta=5e-4 \
  --lr_patience=10 \
  --batch_size=32 \
  --metric_name=emoval \
  --emo_loss_weight=1.0 \
  --val_loss_weight=1.0 \
  --gpu="${GPU}"

# Test1-first selection + stronger classification weight.
run_exp "v2_test1_focus_a" \
  --hidden_dim=128 \
  --dropout=0.35 \
  --use_vae \
  --kl_weight=0.0005 \
  --recon_weight=0.01 \
  --cross_kl_weight=0.001 \
  --use_proxy_attention \
  --fusion_temperature=0.25 \
  --modality_dropout=0.15 \
  --use_modality_dropout \
  --modality_dropout_warmup=40 \
  --lr=5e-4 \
  --l2=5e-5 \
  --epochs=120 \
  --early_stopping_patience=45 \
  --early_stopping_min_delta=3e-4 \
  --lr_patience=12 \
  --batch_size=32 \
  --metric_name=emo \
  --emo_loss_weight=1.3 \
  --val_loss_weight=0.7 \
  --gpu="${GPU}"

# More aggressive clean-data bias.
run_exp "v2_test1_focus_b" \
  --hidden_dim=128 \
  --dropout=0.35 \
  --use_vae \
  --kl_weight=0.0003 \
  --recon_weight=0.008 \
  --cross_kl_weight=0.0008 \
  --use_proxy_attention \
  --fusion_temperature=0.2 \
  --modality_dropout=0.1 \
  --use_modality_dropout \
  --modality_dropout_warmup=50 \
  --lr=5e-4 \
  --l2=5e-5 \
  --epochs=130 \
  --early_stopping_patience=50 \
  --early_stopping_min_delta=3e-4 \
  --lr_patience=12 \
  --batch_size=32 \
  --metric_name=emo \
  --emo_loss_weight=1.4 \
  --val_loss_weight=0.6 \
  --gpu="${GPU}"

# Upper bound check for clean-set optimization (often hurts test2).
run_exp "v2_test1_focus_no_md" \
  --hidden_dim=128 \
  --dropout=0.35 \
  --use_vae \
  --kl_weight=0.0003 \
  --recon_weight=0.008 \
  --cross_kl_weight=0.0008 \
  --use_proxy_attention \
  --fusion_temperature=0.2 \
  --no_modality_dropout \
  --lr=5e-4 \
  --l2=5e-5 \
  --epochs=130 \
  --early_stopping_patience=50 \
  --early_stopping_min_delta=3e-4 \
  --lr_patience=12 \
  --batch_size=32 \
  --metric_name=emo \
  --emo_loss_weight=1.5 \
  --val_loss_weight=0.5 \
  --gpu="${GPU}"

echo ""
echo "All test1-focused experiments finished."
