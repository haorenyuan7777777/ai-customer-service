"""
聊天API路由
- 单轮对话
- 流式对话（SSE）
- 健康检查
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.agent_core import get_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["聊天"])


# ============ 数据模型 ============

class ChatRequest(BaseModel):
    user_message: str
    user_id: str = "anonymous"
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    intent: str
    category: str
    context_used: str
    execution_time_ms: float


# ============ API端点 ============

@router.post("/send", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """单轮对话"""
    try:
        agent = get_agent()
        result = agent.chat(
            user_message=request.user_message,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        return ChatResponse(
            response=result.get("response", ""),
            intent=result.get("intent", "标准客服"),
            category=result.get("category", "general"),
            context_used=result.get("context_used", ""),
            execution_time_ms=result.get("execution_time_ms", 0)
        )
    except Exception as e:
        logger.error(f"对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话（SSE）
    
    【演示实现】：逐字返回
    【生产目标】：对接vLLM stream接口
    """
    async def generate():
        try:
            agent = get_agent()
            result = agent.chat(
                user_message=request.user_message,
                user_id=request.user_id,
                session_id=request.session_id
            )
            
            response = result.get("response", "")
            
            # SSE格式
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            
            for char in response:
                yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
            
            yield f"data: {json.dumps({'type': 'end', 'intent': result.get('intent')})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.get("/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "ok",
        "service": "chat",
        "timestamp": logging.Formatter().formatTime(logging.LogRecord(None, None, None, None, None, None, None))
    }