#!/bin/bash
# Fire Simulator 外部进度监控脚本
# 每5分钟检查进度并发送通知到手机

# ============ 配置区 ============
NTFY_TOPIC="fire-sim-progress"        # 你的 ntfy.sh 主题
NTFY_URL="https://ntfy.sh/$NTFY_TOPIC"
INTERVAL=300                          # 检查间隔(秒) - 5分钟
OUTPUT_DIR="./training_data_v5"       # 输出目录
PROCESS_NAME="generate_training_data_v5.py"  # 进程名称

# ============ 监控函数 ============

send_notification() {
    local title="$1"
    local message="$2"
    local tags="${3:-fire,computer}"
    local priority="${4:-default}"

    curl -X POST "$NTFY_URL" \
        -H "Title: $title" \
        -H "Tags: $tags" \
        -H "Priority: $priority" \
        -d "$message" \
        2>/dev/null

    echo "[$(date '+%H:%M:%S')] 已发送通知: $title"
}

check_process_running() {
    # 检查进程是否在运行
    if pgrep -f "$PROCESS_NAME" > /dev/null; then
        return 0  # 运行中
    else
        return 1  # 未运行
    fi
}

get_progress_from_checkpoint() {
    # 从 checkpoint.json 读取进度
    local checkpoint_file="$OUTPUT_DIR/checkpoint.json"

    if [ ! -f "$checkpoint_file" ]; then
        echo "0|0|0"
        return
    fi

    # 使用 Python 解析 JSON (如果有 jq 更好)
    if command -v jq &> /dev/null; then
        local completed=$(jq -r '.progress.configs_completed // 0' "$checkpoint_file")
        local total=$(jq -r '.progress.configs_total // 0' "$checkpoint_file")
        local percent=$(jq -r '.progress.completion_percent // 0' "$checkpoint_file")
        echo "$completed|$total|$percent"
    else
        # 简易解析（不完美但能用）
        local completed=$(grep -oP '"configs_completed":\s*\K\d+' "$checkpoint_file" | head -1)
        local total=$(grep -oP '"configs_total":\s*\K\d+' "$checkpoint_file" | head -1)
        local percent=$(grep -oP '"completion_percent":\s*\K[\d.]+' "$checkpoint_file" | head -1)
        echo "${completed:-0}|${total:-0}|${percent:-0}"
    fi
}

get_results_count() {
    # 统计 simulation_results.jsonl 的行数
    local results_file="$OUTPUT_DIR/simulation_results.jsonl"

    if [ -f "$results_file" ]; then
        wc -l < "$results_file"
    else
        echo "0"
    fi
}

get_file_size_mb() {
    local file="$1"
    if [ -f "$file" ]; then
        du -m "$file" | cut -f1
    else
        echo "0"
    fi
}

format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# ============ 主监控循环 ============

echo "=========================================="
echo "Fire Simulator 进度监控启动"
echo "=========================================="
echo "监控目录: $OUTPUT_DIR"
echo "进程名称: $PROCESS_NAME"
echo "检查间隔: ${INTERVAL}秒 ($(($INTERVAL / 60))分钟)"
echo "通知地址: $NTFY_URL"
echo "=========================================="

# 发送启动通知
send_notification \
    "监控启动" \
    "Fire Simulator 进度监控已启动
检查间隔: $(($INTERVAL / 60))分钟
时间: $(date '+%Y-%m-%d %H:%M:%S')" \
    "rocket,eyes" \
    "low"

start_time=$(date +%s)
last_completed=0
consecutive_no_progress=0

while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    # 检查进程状态
    if check_process_running; then
        process_status="✓ 运行中"
        process_tags="green_circle"
    else
        process_status="✗ 已停止"
        process_tags="red_circle,warning"

        send_notification \
            "进程已停止!" \
            "Fire Simulator 进程未检测到
时间: $(date '+%Y-%m-%d %H:%M:%S')
可能已完成或发生错误" \
            "$process_tags" \
            "high"

        echo "[$(date '+%H:%M:%S')] 进程未检测到，退出监控"
        exit 1
    fi

    # 获取进度信息
    IFS='|' read -r completed total percent <<< "$(get_progress_from_checkpoint)"
    results_count=$(get_results_count)
    results_size=$(get_file_size_mb "$OUTPUT_DIR/simulation_results.jsonl")

    # 计算进度
    if [ "$total" -gt 0 ]; then
        progress_bar_width=20
        filled=$((progress_bar_width * completed / total))
        empty=$((progress_bar_width - filled))
        bar=$(printf "%${filled}s" | tr ' ' '█')$(printf "%${empty}s" | tr ' ' '░')

        # 估算剩余时间
        if [ "$completed" -gt "$last_completed" ]; then
            rate=$(echo "scale=2; ($completed - $last_completed) / $INTERVAL" | bc -l)
            remaining=$((total - completed))
            if [ "$(echo "$rate > 0" | bc -l)" -eq 1 ]; then
                eta_seconds=$(echo "scale=0; $remaining / $rate" | bc -l)
                eta=$(format_duration $eta_seconds)
            else
                eta="计算中..."
            fi
            consecutive_no_progress=0
        else
            eta="无新进度"
            consecutive_no_progress=$((consecutive_no_progress + 1))
        fi
    else
        bar="等待中..."
        eta="未知"
    fi

    # 构建状态消息
    status_message="进程: $process_status
进度: $completed/$total ($(printf "%.1f" $percent)%)
[$bar]

已完成任务: $completed
总任务数: $total
结果文件: ${results_count}行 (${results_size}MB)

运行时间: $(format_duration $elapsed)
预计剩余: $eta
更新时间: $(date '+%H:%M:%S')"

    # 打印到控制台
    echo ""
    echo "=========================================="
    echo "$status_message"
    echo "=========================================="

    # 发送通知
    if [ "$(echo "$percent >= 100" | bc -l)" -eq 1 ]; then
        # 完成通知
        send_notification \
            "任务完成! 🎉" \
            "$status_message" \
            "tada,white_check_mark,fire" \
            "high"
        echo "[$(date '+%H:%M:%S')] 任务完成，退出监控"
        break
    elif [ "$consecutive_no_progress" -ge 3 ]; then
        # 无进度警告 (15分钟无进度)
        send_notification \
            "警告: 进度停滞" \
            "$status_message

⚠️ 已有 $((consecutive_no_progress * INTERVAL / 60)) 分钟无新进度" \
            "warning,hourglass" \
            "high"
    else
        # 正常进度通知
        local tags="hourglass_flowing_sand,fire"
        if [ "$(echo "$percent >= 75" | bc -l)" -eq 1 ]; then
            tags="rocket,fire"
        elif [ "$(echo "$percent >= 50" | bc -l)" -eq 1 ]; then
            tags="hourglass,fire"
        fi

        send_notification \
            "Fire Sim ($(printf "%.0f" $percent)%)" \
            "$status_message" \
            "$tags" \
            "default"
    fi

    last_completed=$completed

    # 等待下一次检查
    echo "[$(date '+%H:%M:%S')] 下次检查: $(date -d "@$((current_time + INTERVAL))" '+%H:%M:%S')"
    sleep $INTERVAL
done

echo ""
echo "监控结束"
