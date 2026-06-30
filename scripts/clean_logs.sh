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
