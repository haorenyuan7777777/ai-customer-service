#!/bin/bash
# 一键启动AI客服系统全部服务（修复版 - 自动处理容器冲突）

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 AI客服系统一键启动"
echo "项目路径: $PROJECT_ROOT"
echo ""

# 创建日志目录
mkdir -p logs

# ========== 检查依赖 ==========
echo "【1/5】检查依赖..."

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ 未找到: $1，请先安装"
        exit 1
    fi
    echo "  ✅ $1"
}

check_command python3
check_command docker

# ========== 启动Docker服务 ==========
echo ""
echo "【2/5】启动Docker服务..."

# 检查是否全部已运行（避免重复启动）
if docker ps --format '{{.Names}}' | grep -q "milvus-standalone" && \
   docker ps --format '{{.Names}}' | grep -q "vllm-qwen"; then
    echo "  ✅ 容器服务已在运行"
else
    echo "  ▶️ 启动容器服务（vLLM + Milvus 全家桶）..."
    # 清理可能存在的冲突容器（防止名称占用）
    echo "    清理可能存在的旧容器..."
    docker rm -f milvus-etcd milvus-minio milvus-standalone vllm-qwen attu 2>/dev/null || true
    # 使用 --remove-orphans 清理孤儿容器并启动
    docker compose -f docker-compose.yml up -d --remove-orphans
    echo "  ⏳ 等待服务就绪..."
    # 等待 vLLM (8000)
    for i in {1..20}; do
        if nc -z localhost 8000 2>/dev/null; then
            echo "  ✅ vLLM 就绪"
            break
        fi
        sleep 2
        echo "    vLLM 等待中... ($i/20)"
    done
    # 等待 Milvus (19530)
    for i in {1..30}; do
        if nc -z localhost 19530 2>/dev/null; then
            echo "  ✅ Milvus 就绪"
            break
        fi
        sleep 2
        echo "    Milvus 等待中... ($i/30)"
    done
fi

# ========== 检查端口 ==========
echo ""
echo "【3/5】检查服务端口..."

check_port() {
    if nc -z localhost "$1" 2>/dev/null; then
        echo "  ✅ 端口 $1 ($2)"
        return 0
    else
        echo "  ❌ 端口 $1 ($2) 未连通"
        return 1
    fi
}

check_port 8000 "vLLM"
check_port 19530 "Milvus gRPC"

# ========== 预热模型 ==========
echo ""
echo "【4/5】预热模型..."

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

try:
    print("  ▶️  Embedding模型...")
    from src.models.embedding_model import get_embedding_model
    _ = get_embedding_model()
    print("  ✅ Embedding模型就绪")
except Exception as e:
    print(f"  ⚠️ Embedding预热失败: {e}")

try:
    print("  ▶️  Intent模型...")
    from src.models.intent_model import get_intent_model
    _ = get_intent_model()
    print("  ✅ Intent模型就绪")
except Exception as e:
    print(f"  ⚠️ Intent预热失败: {e}")

print("  ✅ 模型预热完成")
PYEOF

# ========== 启动服务 ==========
echo ""
echo "【5/5】启动Web服务..."

# 先停止旧进程
pkill -f "streamlit run src/ui/streamlit_app.py" 2>/dev/null || true
pkill -f "uvicorn src.api.main:app" 2>/dev/null || true
sleep 1

# 启动Streamlit
echo "  ▶️  启动 Streamlit (端口8501)..."
nohup python3 -m streamlit run src/ui/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > logs/streamlit.log 2>&1 &

STREAMLIT_PID=$!
echo $STREAMLIT_PID > logs/streamlit.pid
sleep 3

if kill -0 $STREAMLIT_PID 2>/dev/null; then
    echo "  ✅ Streamlit已启动 (PID: $STREAMLIT_PID)"
else
    echo "  ❌ Streamlit启动失败，查看日志: logs/streamlit.log"
fi

# 启动FastAPI
echo "  ▶️  启动 FastAPI (端口8080)..."
nohup python3 -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 1 \
    > logs/api.log 2>&1 &

API_PID=$!
echo $API_PID > logs/api.pid
sleep 3

if kill -0 $API_PID 2>/dev/null; then
    echo "  ✅ FastAPI已启动 (PID: $API_PID)"
else
    echo "  ❌ FastAPI启动失败，查看日志: logs/api.log"
    echo "  🔍 运行诊断: python3 scripts/diagnose_api.py"
fi

# ========== 完成 ==========
echo ""
echo "=========================================="
echo "✅ AI客服系统启动完成"
echo "=========================================="
echo ""
echo "【访问地址】"
echo "  管理后台:    http://localhost:8501"
echo "  API服务:     http://localhost:8080"
echo "  API文档:     http://localhost:8080/docs"
echo "  健康检查:    http://localhost:8080/health"
echo ""
echo "【验证命令】"
echo "  健康检查:    curl http://localhost:8080/health"
echo "  聊天测试:    curl -X POST http://localhost:8080/chat -H 'Content-Type: application/json' -d '{\"message\":\"你好\"}'"
echo ""
echo "【日志查看】"
echo "  Streamlit:   tail -f logs/streamlit.log"
echo "  FastAPI:     tail -f logs/api.log"
echo ""
echo "【停止命令】"
echo "  bash scripts/stop_all.sh"
echo "=========================================="