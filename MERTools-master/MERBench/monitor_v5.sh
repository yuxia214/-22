#!/bin/bash
# 实验监控脚本 - 查看所有实验状态

cd /root/autodl-tmp/MERTools-master/MERBench

echo "=========================================="
echo "V5 实验监控 - $(date)"
echo "=========================================="
echo ""

# GPU状态
echo "=== GPU状态 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv
echo ""

# 统计完成情况
completed=0
running=0
for i in $(seq 1 20); do
    log=$(ls logs/v5_experiments/exp${i}_*.log 2>/dev/null | head -1)
    if [ -n "$log" ]; then
        if grep -q "所有实验完成\|Prediction and Saving" "$log" 2>/dev/null; then
            ((completed++))
        else
            ((running++))
        fi
    fi
done
echo "=== 完成统计: $completed/20 完成, $running 运行中 ==="
echo ""

# 各实验详情
echo "=== 各实验进度 ==="
printf "%-6s %-35s %-10s %-10s\n" "实验" "配置" "Epoch" "Eval"
echo "--------------------------------------------------------------"

configs=(
    "V5基础版"
    "V5+Mixup"
    "V5+hidden256"
    "V4 Pure VAE"
    "V5+recon=0.2"
    "V5无模态Dropout"
    "V5+dropout=0.25"
    "V5+kl=0.05"
    "V5+warmup=40"
    "V5+Mixup+h256"
    "V5+lr=1e-4"
    "V5+l2=1e-4"
    "V4+对比学习"
    "V4+门控融合"
    "V5+hidden64"
    "V5无proxy"
    "V5+temp=2.0"
    "V5+temp=0.5"
    "V5+moddrop=0.25"
    "V5+batch=64"
)

for i in $(seq 1 20); do
    log=$(ls logs/v5_experiments/exp${i}_*.log 2>/dev/null | head -1)
    config="${configs[$((i-1))]}"

    if [ -n "$log" ]; then
        # 检查是否完成
        if grep -q "save results in" "$log" 2>/dev/null; then
            # 提取最终结果
            test2_line=$(grep "test2_" "$log" | tail -1)
            if [ -n "$test2_line" ]; then
                f1=$(echo "$test2_line" | grep -oP 'f1:\K[0-9.]+')
                printf "%-6s %-35s %-10s %-10s ✅\n" "Exp$i" "$config" "完成" "F1=$f1"
            else
                printf "%-6s %-35s %-10s %-10s ✅\n" "Exp$i" "$config" "完成" "-"
            fi
        else
            # 提取当前epoch
            last_line=$(grep "epoch:" "$log" | tail -1)
            if [ -n "$last_line" ]; then
                epoch=$(echo "$last_line" | grep -oP 'epoch:\K[0-9]+')
                eval_score=$(echo "$last_line" | grep -oP 'eval:\K[0-9.]+')
                printf "%-6s %-35s %-10s %-10s 🔄\n" "Exp$i" "$config" "$epoch/100" "$eval_score"
            else
                printf "%-6s %-35s %-10s %-10s ⏳\n" "Exp$i" "$config" "启动中" "-"
            fi
        fi
    else
        printf "%-6s %-35s %-10s %-10s ❌\n" "Exp$i" "$config" "无日志" "-"
    fi
done

echo ""
echo "=========================================="
