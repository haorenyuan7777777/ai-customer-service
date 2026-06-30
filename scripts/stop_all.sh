#!/bin/bash
# 一键停止AI客服系统全部服务

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🛑 停止AI客服系统..."

# ========== 1. 停止 Python 进程 ==========
echo ""
echo "【1/3】停止 Python 服务..."

# 停止 Streamlit（通过 PID 文件）
if [ -f logs/streamlit.pid ]; then
    kill "$(cat logs/streamlit.pid)" 2>/dev/null && echo "  ✅ Streamlit (PID: $(cat logs/streamlit.pid)) 已停止"
    rm -f logs/streamlit.pid
else
    # 备用：通过进程名停止
    if pkill -f "streamlit run src/ui/streamlit_app.py" 2>/dev/null; then
        echo "  ✅ Streamlit 已停止（通过进程名）"
    fi
fi

# 停止 FastAPI（通过 PID 文件）
if [ -f logs/api.pid ]; then
    kill "$(cat logs/api.pid)" 2>/dev/null && echo "  ✅ FastAPI (PID: $(cat logs/api.pid)) 已停止"
    rm -f logs/api.pid
else
    if pkill -f "uvicorn src.api.main:app" 2>/dev/null; then
        echo "  ✅ FastAPI 已停止（通过进程名）"
    fi
fi

# 停止监控进程（如果有）
pkill -f "python.*scripts/monitor.py" 2>/dev/null && echo "  ✅ 监控进程已停止"

# ========== 2. 停止 Docker 容器 ==========
echo ""
echo "【2/3】Docker 容器..."

# 询问是否停止 Docker 容器
read -p "是否停止 Docker 容器（vLLM + Milvus 全家桶）？(y/N): " stop_docker
if [[ "$stop_docker" =~ ^[Yy]$ ]]; then
    if command -v docker &> /dev/null; then
        # 使用 docker compose 停止所有容器（会移除容器，但保留卷）
        docker compose -f docker-compose.yml down 2>/dev/null && echo "  ✅ Docker 容器已停止并移除"
        # 如果 Compose 项目名称不同，可以指定 -p 参数，但通常默认即可
    else
        echo "  ❌ Docker 未安装或未运行"
    fi
else
    echo "  ⏭️ Docker 容器保持运行"
fi

# ========== 3. 清理可能残留的容器（安全兜底） ==========
echo ""
echo "【3/3】清理残留..."

# 可选：停止并移除可能单独运行的 vLLM 或 Milvus 容器（防止遗漏）
# 但不强制，因为用户可能手动启动过其他容器

echo "  ✅ 清理完成"

# ========== 完成 ==========
echo ""
echo "✅ 系统已停止"