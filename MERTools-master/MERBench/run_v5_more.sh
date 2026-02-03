#!/bin/bash
# V5 更多实验 - 不同架构 + 超参数组合
# 充分利用GPU资源

PYTHON=/root/miniconda3/bin/python
cd /root/autodl-tmp/MERTools-master/MERBench

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "更多实验 - 开始时间: $(date)"
echo "=========================================="

# ==================== 实验13: V4 + 对比学习 (对照) ====================
echo "启动实验13: V4 + 对比学习..."
screen -dmS v5_exp13 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
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
    2>&1 | tee logs/v5_experiments/exp13_v4_contrastive_${TIMESTAMP}.log
exec bash
"

# ==================== 实验14: V4 + 门控融合 (对照) ====================
echo "启动实验14: V4 + 门控融合..."
screen -dmS v5_exp14 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
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
    2>&1 | tee logs/v5_experiments/exp14_v4_gated_${TIMESTAMP}.log
exec bash
"

# ==================== 实验15: V5 + hidden_dim=64 (更小模型) ====================
echo "启动实验15: V5 + hidden_dim=64..."
screen -dmS v5_exp15 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=64 --dropout=0.3 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp15_v5_hidden64_${TIMESTAMP}.log
exec bash
"

# ==================== 实验16: V5 + 无proxy attention ====================
echo "启动实验16: V5 无proxy attention..."
screen -dmS v5_exp16 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --no_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp16_v5_no_proxy_${TIMESTAMP}.log
exec bash
"

# ==================== 实验17: V5 + 更高温度 ====================
echo "启动实验17: V5 + temperature=2.0..."
screen -dmS v5_exp17 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=2.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp17_v5_temp2_${TIMESTAMP}.log
exec bash
"

# ==================== 实验18: V5 + 更低温度 ====================
echo "启动实验18: V5 + temperature=0.5..."
screen -dmS v5_exp18 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=0.5 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp18_v5_temp05_${TIMESTAMP}.log
exec bash
"

# ==================== 实验19: V5 + 更强模态dropout ====================
echo "启动实验19: V5 + modality_dropout=0.25..."
screen -dmS v5_exp19 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.25 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp19_v5_moddrop025_${TIMESTAMP}.log
exec bash
"

# ==================== 实验20: V5 + batch_size=64 ====================
echo "启动实验20: V5 + batch_size=64..."
screen -dmS v5_exp20 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
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
    --early_stopping_patience=30 --batch_size=64 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp20_v5_bs64_${TIMESTAMP}.log
exec bash
"

sleep 3

echo ""
echo "=========================================="
echo "新增8个实验! 现在共20个实验"
echo "=========================================="
screen -list | grep v5_exp | wc -l
echo "个screen会话运行中"
echo ""
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
