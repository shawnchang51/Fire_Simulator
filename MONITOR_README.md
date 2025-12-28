# Fire Simulator 进度监控使用说明

## 快速开始

### 1. 在手机上安装 ntfy app
- **Android**: [Google Play](https://play.google.com/store/apps/details?id=io.heckel.ntfy) 或 [F-Droid](https://f-droid.org/en/packages/io.heckel.ntfy/)
- **iOS**: [App Store](https://apps.apple.com/us/app/ntfy/id1625396347)

### 2. 订阅你的主题
在 ntfy app 中订阅主题: `fire-sim-progress`（或你自定义的主题名称）

### 3. 在 Linux 服务器上启动监控

```bash
# 给脚本添加执行权限
chmod +x monitor_progress.sh

# 后台运行监控脚本
nohup ./monitor_progress.sh > monitor.log 2>&1 &

# 或使用 screen/tmux
screen -S fire-monitor
./monitor_progress.sh
# Ctrl+A+D 脱离
```

### 4. 查看监控日志

```bash
# 实时查看监控日志
tail -f monitor.log

# 查看后台任务
jobs
ps aux | grep monitor_progress.sh
```

### 5. 停止监控

```bash
# 查找监控进程
ps aux | grep monitor_progress.sh

# 杀死进程
kill <PID>

# 或如果在 screen 中
screen -r fire-monitor
# Ctrl+C
```

## 配置选项

编辑 `monitor_progress.sh` 顶部的配置：

```bash
NTFY_TOPIC="fire-sim-progress"        # ntfy.sh 主题名称
INTERVAL=300                          # 检查间隔(秒) - 默认5分钟
OUTPUT_DIR="./training_data_v5"       # 训练数据输出目录
```

## 监控内容

脚本会监控以下信息：
- ✓ 进程运行状态
- ✓ 任务完成进度 (从 checkpoint.json)
- ✓ 结果文件大小和行数
- ✓ 预计剩余时间
- ✓ 运行总时间

## 通知类型

| 事件 | 优先级 | 标签 |
|------|--------|------|
| 监控启动 | Low | 🚀 eyes |
| 正常进度 (0-50%) | Default | ⏳ fire |
| 正常进度 (50-75%) | Default | ⌛ fire |
| 正常进度 (75-100%) | Default | 🚀 fire |
| 进度停滞警告 | High | ⚠️ warning |
| 任务完成 | High | 🎉 ✅ fire |
| 进程停止 | High | 🔴 warning |

## 一键启动命令

```bash
# 启动训练 + 监控（推荐）
nohup python generate_training_data_v5.py > training.log 2>&1 &
sleep 5  # 等待程序启动
nohup ./monitor_progress.sh > monitor.log 2>&1 &

# 查看所有日志
tail -f training.log monitor.log
```

## 测试通知

在启动监控前，可以先测试通知是否正常工作：

```bash
# 测试发送通知
curl -X POST https://ntfy.sh/fire-sim-progress \
  -H "Title: 测试通知" \
  -H "Tags: white_check_mark" \
  -d "通知系统测试成功!"
```

检查手机是否收到通知。

## 故障排查

### 1. 没收到通知
- 检查 ntfy app 是否正确订阅了主题
- 检查服务器网络是否能访问 ntfy.sh
- 检查监控日志: `tail -f monitor.log`

### 2. 进度不更新
- 确认 `checkpoint.json` 文件存在
- 检查训练程序是否正常运行: `ps aux | grep generate_training_data_v5.py`
- 查看训练日志: `tail -f training.log`

### 3. 监控脚本报错
- 确保 `bc` 命令已安装: `sudo apt-get install bc`
- 确保 `curl` 命令已安装: `sudo apt-get install curl`

## 进阶：使用 systemd 服务

创建 `/etc/systemd/system/fire-monitor.service`:

```ini
[Unit]
Description=Fire Simulator Progress Monitor
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/Fire_Simulator
ExecStart=/path/to/Fire_Simulator/monitor_progress.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl start fire-monitor
sudo systemctl enable fire-monitor  # 开机自启
sudo systemctl status fire-monitor  # 查看状态
```

## 自定义通知主题

为了安全和隐私，建议使用随机主题名称：

```bash
# 生成随机主题名
TOPIC="fire-sim-$(openssl rand -hex 4)"
echo "你的主题: $TOPIC"

# 在脚本中使用
NTFY_TOPIC="$TOPIC"
```

然后在手机 app 中订阅这个随机主题。
