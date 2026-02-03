#!/bin/bash
# 使用screen在后台运行V5实验
# 用法: bash start_v5_screen.sh

cd /root/autodl-tmp/MERTools-master/MERBench

# 检查是否已有同名screen会话
if screen -list | grep -q "v5_exp"; then
    echo "警告: 已存在名为 'v5_exp' 的screen会话"
    echo "请先终止: screen -X -S v5_exp quit"
    echo "或者附加到现有会话: screen -r v5_exp"
    exit 1
fi

# 确保脚本可执行
chmod +x run_v5_experiments.sh

# 创建日志目录
mkdir -p logs/v5_experiments

# 启动screen会话并运行实验
echo "=========================================="
echo "启动V5实验 (screen后台运行)"
echo "=========================================="
echo ""
echo "Screen会话名称: v5_exp"
echo "查看实验进度: screen -r v5_exp"
echo "分离会话: Ctrl+A, D"
echo "终止实验: screen -X -S v5_exp quit"
echo ""
echo "实时查看日志:"
echo "  tail -f logs/v5_experiments/*.log"
echo ""

# 启动screen
screen -dmS v5_exp bash -c "bash run_v5_experiments.sh; exec bash"

echo "Screen会话已启动!"
echo ""
echo "使用以下命令查看进度:"
echo "  screen -r v5_exp"
