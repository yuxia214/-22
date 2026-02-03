#!/bin/bash
# V5 并行实验脚本 - 最大化GPU利用率
# 同时运行4个实验，分别在不同的screen终端

PYTHON=/root/miniconda3/bin/python
cd /root/autodl-tmp/MERTools-master/MERBench

# 创建日志目录
mkdir -p logs/v5_experiments

# 实验时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "V5 并行实验 - 开始时间: $(date)"
echo "=========================================="
echo "将同时启动4个实验，充分利用GPU"
echo ""

# 清理旧的screen会话
for i in 1 2 3 4 5 6; do
    screen -X -S v5_exp$i quit 2>/dev/null
done
sleep 1

# ==================== 实验1: V5 基础版 ====================
echo "启动实验1: V5 基础版..."
screen -dmS v5_exp1 bash -c "
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
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp1_v5_base_${TIMESTAMP}.log
exec bash
"

# ==================== 实验2: V5 + Mixup ====================
echo "启动实验2: V5 + Mixup..."
screen -dmS v5_exp2 bash -c "
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
    --use_mixup --mixup_alpha=0.4 \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp2_v5_mixup_${TIMESTAMP}.log
exec bash
"

# ==================== 实验3: V5 + hidden_dim=256 ====================
echo "启动实验3: V5 + hidden_dim=256..."
screen -dmS v5_exp3 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
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
exec bash
"

# ==================== 实验4: V4 Pure VAE 对照组 ====================
echo "启动实验4: V4 Pure VAE 对照组..."
screen -dmS v5_exp4 bash -c "
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
    --no_gated_fusion \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp4_v4_pure_vae_${TIMESTAMP}.log
exec bash
"

sleep 3

echo ""
echo "=========================================="
echo "第一批4个实验已启动!"
echo "=========================================="
echo ""
echo "Screen会话列表:"
screen -list
echo ""
echo "GPU使用情况:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
echo ""
echo "查看各实验进度:"
echo "  screen -r v5_exp1  # V5基础版"
echo "  screen -r v5_exp2  # V5+Mixup"
echo "  screen -r v5_exp3  # V5+hidden256"
echo "  screen -r v5_exp4  # V4 Pure VAE对照"
echo ""
echo "查看所有日志:"
echo "  tail -f logs/v5_experiments/*.log"
echo ""
echo "=========================================="
echo "第一批实验完成后，运行以下命令启动第二批:"
echo "  bash run_v5_parallel_batch2.sh"
echo "=========================================="
