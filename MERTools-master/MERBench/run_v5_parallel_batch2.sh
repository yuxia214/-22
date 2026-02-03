#!/bin/bash
# V5 并行实验 - 第二批
# 在第一批实验完成后运行

PYTHON=/root/miniconda3/bin/python
cd /root/autodl-tmp/MERTools-master/MERBench

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "V5 第二批实验 - 开始时间: $(date)"
echo "=========================================="

# ==================== 实验5: V5 + 更强重建损失 ====================
echo "启动实验5: V5 + recon_weight=0.2..."
screen -dmS v5_exp5 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
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
    2>&1 | tee logs/v5_experiments/exp5_v5_recon02_${TIMESTAMP}.log
exec bash
"

# ==================== 实验6: V5 无模态Dropout ====================
echo "启动实验6: V5 无模态Dropout..."
screen -dmS v5_exp6 bash -c "
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
    --no_modality_dropout \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp6_v5_no_mod_dropout_${TIMESTAMP}.log
exec bash
"

sleep 2

echo ""
echo "第二批2个实验已启动!"
echo ""
screen -list
