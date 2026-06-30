#!/bin/bash
# WSL优化脚本
# - 配置.wslconfig
# - 设置定时任务（日志清理、容器清理）
# - Docker日志轮转配置

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "🔧 WSL优化配置"
echo "项目路径: $PROJECT_ROOT"

# ========== 1. .wslconfig ==========
WSLCONFIG="$HOME/.wslconfig"

echo ""
echo "【1/4】配置 .wslconfig"

cat > "$WSLCONFIG" << 'EOF'
[wsl2]
# 内存与CPU限制（根据你的硬件调整）
memory=15GB
processors=14
swap=8GB
swapFile=C:\\temp\\wsl-swap.vhdx

# 网络（使用NAT模式，兼容Docker端口映射）
localhostForwarding=true
dnsTunneling=true
firewall=true
autoProxy=false   # 避免与Docker端口映射冲突

# 自动回收空间（需要WSL 2.0.0+）
sparseVhd=true
EOF

echo "✅ .wslconfig 已写入: $WSLCONFIG"
echo "   内存: 15GB | 处理器: 14核 | Swap: 8GB"

# ========== 2. Docker配置优化 ==========
echo ""
echo "【2/4】Docker配置优化（日志轮转）"

DOCKER_CONFIG="$HOME/.docker/daemon.json"
mkdir -p "$(dirname "$DOCKER_CONFIG")"

cat > "$DOCKER_CONFIG" << 'EOF'
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

echo "✅ Docker日志轮转已配置（单文件10MB，保留3个）"
echo "   ⚠️ 配置生效需重启Docker: sudo service docker restart"

# ========== 3. 日志清理脚本 ==========
echo ""
echo "【3/4】创建日志清理脚本"

CLEAN_SCRIPT="$PROJECT_ROOT/scripts/clean_logs.sh"
mkdir -p "$PROJECT_ROOT/scripts"
mkdir -p "$PROJECT_ROOT/logs"

cat > "$CLEAN_SCRIPT" << 'EOF'
#!/bin/bash
# 日志清理脚本
# 每周运行一次

LOG_DIR="$(cd "$(dirname "$0")/.." && pwd)/logs"
echo "🧹 清理日志: $LOG_DIR"

# 删除超过7天的日志
find "$LOG_DIR" -name "*.log" -mtime +7 -delete
find "$LOG_DIR" -name "metrics.jsonl" -mtime +7 -delete

# 清空过大的日志文件（>100MB）
find "$LOG_DIR" -name "*.log" -size +100M -exec truncate -s 0 {} \;

# 压缩旧评测报告（超过3天）
if command -v gzip &> /dev/null; then
    find "$LOG_DIR" -name "benchmark_*.json" -mtime +3 -exec gzip {} \;
fi

# 清理journal日志（保留最近3天）
if command -v journalctl &> /dev/null; then
    sudo journalctl --vacuum-time=3d
fi

echo "✅ 日志清理完成"
EOF

chmod +x "$CLEAN_SCRIPT"
echo "✅ 日志清理脚本: $CLEAN_SCRIPT"

# ========== 4. 定时任务 ==========
echo ""
echo "【4/4】设置定时任务（crontab）"

CRON_JOBS="# AI客服系统定时任务
# 每周一凌晨3点清理日志
0 3 * * 1 $CLEAN_SCRIPT >> $PROJECT_ROOT/logs/cron.log 2>&1
# 每天凌晨4点清理已停止的Docker容器（不删除镜像）
0 4 * * * /usr/bin/docker container prune -f >> $PROJECT_ROOT/logs/docker_prune.log 2>&1
"

# 检查是否已存在相同任务
if crontab -l 2>/dev/null | grep -q "AI客服系统"; then
    echo "⏭️ 定时任务已存在，跳过"
else
    (crontab -l 2>/dev/null; echo "$CRON_JOBS") | crontab -
    echo "✅ 定时任务已添加"
fi

# ========== 完成 ==========
echo ""
echo "=========================================="
echo "✅ WSL优化配置完成"
echo "=========================================="
echo ""
echo "【生效方式】"
echo "  1. 重启WSL: wsl --shutdown"
echo "  2. 重新打开WSL终端"
echo "  3. 重启Docker服务: sudo service docker restart"
echo ""
echo "【监控命令】"
echo "  实时资源: python scripts/monitor.py"
echo "  手动清理: bash scripts/clean_logs.sh"
echo "  容器清理: docker container prune -f"
echo ""
echo "【注意事项】"
echo "  - .wslconfig 修改后必须 wsl --shutdown 生效"
echo "  - 定时任务每天4:00清理已停止的容器（不影响镜像）"
echo "  - 建议每周检查一次 logs/ 目录大小"
echo "=========================================="