"""
FastAPI主应用
- 启动时异步预热关键模型（不阻塞）
- 请求时等待预热完成（或超时降级）
- 保留延迟加载兜底
"""

import asyncio
import os
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import API

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ Pydantic模型 ============
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = "default"
    use_flow: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    intent: str = "标准客服"
    category: str = "general"
    context_used: Optional[str] = None
    execution_time_ms: float = 0.0


# ============ 全局状态 ============
_agent = None
_rag = None
_flow = None
_ready = False
_ready_event = asyncio.Event()
_preheat_task = None


async def _preheat_models():
    """异步预热所有关键模型"""
    global _agent, _rag, _ready
    try:
        logger.info("🔄 开始后台预热模型...")
        
        # 预热 Embedding（轻量，优先）
        from src.models.embedding_model import get_embedding_model
        get_embedding_model()
        logger.info("   ✅ Embedding模型就绪")
        
        # 预热 RAG（较重）
        from src.rag.llama_index_rag import get_rag_engine
        _rag = get_rag_engine()
        logger.info("   ✅ RAG引擎就绪")
        
        # 预热 Agent（依赖上述组件）
        from src.agent.agent_core import get_agent
        _agent = get_agent()
        logger.info("   ✅ Agent就绪")
        
        # 预热 Promptflow（可选，非关键）
        try:
            from src.promptflow.flow_engine import get_flow_engine
            _flow = get_flow_engine()
            logger.info("   ✅ Promptflow引擎就绪")
        except Exception as e:
            logger.warning(f"   ⚠️ Promptflow预热失败（非关键）: {e}")
        
        _ready = True
        _ready_event.set()
        logger.info("✅ 所有模型预热完成")
    except Exception as e:
        logger.error(f"❌ 预热失败: {e}")
        # 即使预热失败，也设置事件，后续请求可尝试延迟加载
        _ready_event.set()


def get_agent_smart():
    """智能获取Agent：优先使用预热实例，否则延迟加载"""
    global _agent
    if _agent is not None:
        return _agent
    # 如果尚未加载，延迟加载（兜底）
    from src.agent.agent_core import get_agent
    _agent = get_agent()
    return _agent


def get_rag_smart():
    """智能获取RAG"""
    global _rag
    if _rag is not None:
        return _rag
    from src.rag.llama_index_rag import get_rag_engine
    _rag = get_rag_engine()
    return _rag


# ============ 生命周期 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _preheat_task
    logger.info("🚀 FastAPI启动中...")
    # 启动后台预热任务
    _preheat_task = asyncio.create_task(_preheat_models())
    logger.info("   后台预热已启动，服务已就绪（可接收请求）")
    yield
    logger.info("👋 FastAPI关闭中...")


# ============ 创建应用 ============
app = FastAPI(
    title="AI客服系统API",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 路由 ============
@app.get("/")
async def root():
    return {"name": "AI客服系统API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health():
    """健康检查：返回服务状态和预热进度"""
    return {
        "status": "ready" if _ready else "warming_up",
        "services": {
            "vllm": _check_port("localhost", 8000),
            "milvus_grpc": _check_port("localhost", 19530),
            "milvus_http": _check_port("localhost", 9091),
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.perf_counter()
    
    # 等待预热完成（最多30秒）
    if not _ready:
        try:
            await asyncio.wait_for(_ready_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("⏰ 预热超时，尝试延迟加载")
    
    try:
        agent = get_agent_smart()
        result = agent.chat(
            user_message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            use_flow=request.use_flow
        )
        elapsed = (time.perf_counter() - start) * 1000
        return ChatResponse(
            response=result.get("response", ""),
            intent=result.get("intent", "标准客服"),
            category=result.get("category", "general"),
            context_used=result.get("context_used", "")[:200] if result.get("context_used") else None,
            execution_time_ms=round(elapsed, 2)
        )
    except Exception as e:
        logger.error(f"聊天接口错误: {e}")
        # 降级响应
        return ChatResponse(
            response=f"系统暂时繁忙，请稍后重试。（{str(e)[:50]}）",
            execution_time_ms=round((time.perf_counter() - start) * 1000, 2)
        )


@app.get("/metrics")
async def metrics():
    import psutil
    try:
        import torch
        gpu_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None
    except:
        gpu_alloc = gpu_total = None
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_allocated_gb": round(gpu_alloc, 2) if gpu_alloc else None,
        "gpu_total_gb": round(gpu_total, 2) if gpu_total else None,
        "models_ready": _ready
    }


def _check_port(host, port):
    import socket
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except:
        return False


# ============ 子路由（可选） ============
try:
    from src.api.routers import admin, chat as chat_router, knowledge
    app.include_router(admin.router, prefix="/api/v1")
    app.include_router(chat_router.router, prefix="/api/v1")
    app.include_router(knowledge.router, prefix="/api/v1")
    logger.info("✅ 子路由加载成功")
except Exception as e:
    logger.warning(f"⚠️ 子路由加载失败: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=API.get("host", "0.0.0.0"),
        port=API.get("port", 8080),
        workers=1,
        reload=False
    )