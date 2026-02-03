#!/bin/bash
# V5 系列实验脚本
# 基于V4消融实验结论（Pure VAE最佳），进行V5改进实验
# 使用screen在后台运行

PYTHON=/root/miniconda3/bin/python
cd /root/autodl-tmp/MERTools-master/MERBench

# 创建日志目录
mkdir -p logs/v5_experiments

# 实验时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "V5 系列实验 - 开始时间: $(date)"
echo "=========================================="

# ==================== 实验1: V5 基础版 ====================
# 深度编码器 + 自适应融合，无Mixup
echo ""
echo "=========================================="
echo "实验1: V5 基础版 (深度编码器 + 自适应融合)"
echo "=========================================="
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp1_v5_base_${TIMESTAMP}.log

# ==================== 实验2: V5 + Mixup ====================
echo ""
echo "=========================================="
echo "实验2: V5 + Mixup数据增强"
echo "=========================================="
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --use_mixup --mixup_alpha=0.4 \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp2_v5_mixup_${TIMESTAMP}.log

# ==================== 实验3: V5 + 更大hidden_dim ====================
echo ""
echo "=========================================="
echo "实验3: V5 + 更大hidden_dim (256)"
echo "=========================================="
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=256 --dropout=0.4 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=3e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp3_v5_hidden256_${TIMESTAMP}.log

# ==================== 实验4: V5 + 更强重建损失 ====================
echo ""
echo "=========================================="
echo "实验4: V5 + 更强重建损失 (recon_weight=0.2)"
echo "=========================================="
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.2 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp4_v5_recon02_${TIMESTAMP}.log

# ==================== 实验5: V5 + 无模态Dropout ====================
echo ""
echo "=========================================="
echo "实验5: V5 无模态Dropout (测试自适应融合效果)"
echo "=========================================="
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --no_modality_dropout \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp5_v5_no_mod_dropout_${TIMESTAMP}.log

# ==================== 实验6: V4 Pure VAE 对照组 ====================
echo ""
echo "=========================================="
echo "实验6: V4 Pure VAE 对照组 (复现最佳结果)"
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
    2>&1 | tee logs/v5_experiments/exp6_v4_pure_vae_${TIMESTAMP}.log

echo ""
echo "=========================================="
echo "所有V5实验完成!"
echo "结束时间: $(date)"
echo "=========================================="
echo ""
echo "日志文件位置: logs/v5_experiments/"
echo "请查看各实验日志获取详细结果"
