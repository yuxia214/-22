#!/bin/bash
# AttentionRobustV4 消融实验脚本
# 目的: 找出导致test2下降的模块

PYTHON=/root/miniconda3/bin/python

cd /root/autodl-tmp/MERTools-master/MERBench

echo "=========================================="
echo "实验1: 关闭对比学习 (--no_contrastive)"
echo "=========================================="
$PYTHON -u main-robust.py \
    --model='attention_robust_v4' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --use_vae --kl_weight=0.01 --recon_weight=0.1 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --no_contrastive \
    --use_gated_fusion --gate_alpha=0.5 \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee ablation_exp1_no_contrastive.log

echo ""
echo "=========================================="
echo "实验2: 关闭门控融合 (--no_gated_fusion)"
echo "=========================================="
$PYTHON -u main-robust.py \
    --model='attention_robust_v4' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --use_vae --kl_weight=0.01 --recon_weight=0.1 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_contrastive --contrastive_weight=0.1 \
    --no_gated_fusion \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee ablation_exp2_no_gated_fusion.log

echo ""
echo "=========================================="
echo "实验3: 两者都关闭 (纯VAE基线)"
echo "=========================================="
$PYTHON -u main-robust.py \
    --model='attention_robust_v4' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --use_vae --kl_weight=0.01 --recon_weight=0.1 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --no_contrastive \
    --no_gated_fusion \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee ablation_exp3_pure_vae.log

echo ""
echo "=========================================="
echo "所有消融实验完成!"
echo "=========================================="
echo "日志文件:"
echo "  - ablation_exp1_no_contrastive.log"
echo "  - ablation_exp2_no_gated_fusion.log"
echo "  - ablation_exp3_pure_vae.log"
