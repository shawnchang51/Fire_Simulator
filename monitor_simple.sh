#!/bin/bash
# 简化版监控 - 通过监控日志文件和结果文件

NTFY_TOPIC="fire-sim-progress"
NTFY_URL="https://ntfy.sh/$NTFY_TOPIC"
INTERVAL=300  # 5分钟
LOG_FILE="training.log"  # 训练日志文件
RESULTS_FILE="./training_data_v5/simulation_results.jsonl"

send_notification() {
    curl -s -X POST "$NTFY_URL" \
        -H "Title: $1" \
        -H "Tags: $2" \
        -d "$3" > /dev/null
    echo "[$(date '+%H:%M:%S')] 通知: $1"
}

echo "简化版监控启动 - 每 $((INTERVAL/60)) 分钟检查"
send_notification "监控启动" "rocket,eyes" "Fire Simulator 监控已启动"

start_time=$(date +%s)
last_count=0

while true; do
    # 检查进程
    if ! pgrep -f "generate_training_data_v5.py" > /dev/null; then
        send_notification "进程已停止" "warning,red_circle" "程序可能已完成或发生错误"
        break
    fi

    # 统计结果文件
    if [ -f "$RESULTS_FILE" ]; then
        current_count=$(wc -l < "$RESULTS_FILE")
        file_size=$(du -h "$RESULTS_FILE" | cut -f1)
    else
        current_count=0
        file_size="0"
    fi

    # 计算速度
    elapsed=$(( $(date +%s) - start_time ))
    rate=$(echo "scale=2; ($current_count - $last_count) / ($INTERVAL / 60)" | bc -l)

    # 获取最新日志
    if [ -f "$LOG_FILE" ]; then
        latest_log=$(tail -5 "$LOG_FILE" | grep -E "Completed|Phase|pairs" | tail -1)
    else
        latest_log="无日志"
    fi

    # 构建消息
    message="✓ 运行中

结果: ${current_count} 条
文件大小: ${file_size}
速度: ${rate} 条/分钟
运行时间: $((elapsed/3600))h $((elapsed%3600/60))m

最新: ${latest_log}
时间: $(date '+%H:%M:%S')"

    echo "$message"
    send_notification "Fire Sim 运行中" "hourglass,fire" "$message"

    last_count=$current_count
    sleep $INTERVAL
done

echo "监控结束"
