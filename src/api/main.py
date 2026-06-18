"""
FastAPI主服务
提供对话、知识库、管理接口
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json

from src.agent.agent_core import CustomerServiceAgent
from src.agent.memory import ChatMemory
from src.rag.milvus_store import MilvusKnowledgeStore
from src.models.embedding_model import BGEEmbedding
from src.config import API, get_gpu_memory_info


# 创建FastAPI应用
app = FastAPI(
    title="AI客服系统",
    description="基于Milvus + LlamaIndex + Promptflow的智能客服",
    version="1.0.0"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例（延迟初始化）
agent = None
memory = None
knowledge_store = None
embedding_model = None


def get_agent():
    global agent
    if agent is None:
        agent = CustomerServiceAgent()
    return agent


def get_memory():
    global memory
    if memory is None:
        memory = ChatMemory()
    return memory


def get_knowledge_store():
    global knowledge_store
    if knowledge_store is None:
        knowledge_store = MilvusKnowledgeStore()
    return knowledge_store


def get_embedding():
    global embedding_model
    if embedding_model is None:
        embedding_model = BGEEmbedding()
    return embedding_model


# ========== 数据模型 ==========
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    response: str
    intent: str
    intent_confidence: float
    sources: List[dict]
    session_id: str


class KnowledgeItem(BaseModel):
    instruction: str
    output: str
    category: str = "general"
    intent: str = "general_query"


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = 5


# ========== 对话接口 ==========
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口 - 核心入口
    
    处理流程:
    1. 意图识别
    2. 记忆加载
    3. RAG检索
    4. LLM生成
    5. 记忆保存
    """
    try:
        service_agent = get_agent()
        result = service_agent.process(request.query, request.session_id)
        
        return ChatResponse(
            response=result["response"],
            intent=result["intent"],
            intent_confidence=result["intent_confidence"],
            sources=result["sources"],
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 知识库接口 ==========
@app.post("/api/v1/knowledge")
async def add_knowledge(item: KnowledgeItem):
    """添加知识到知识库"""
    try:
        store = get_knowledge_store()
        emb = get_embedding()
        
        # 向量化
        vector = emb.encode([item.instruction], show_progress=False)[0]
        
        # 生成ID
        import uuid
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        # 插入Milvus
        store.insert(
            ids=[doc_id],
            vectors=vector.reshape(1, -1),
            instructions=[item.instruction],
            outputs=[item.output],
            categories=[item.category],
            intents=[item.intent]
        )
        
        return {"status": "success", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/knowledge/search")
async def search_knowledge(query: str, top_k: int = 5):
    """检索知识库"""
    try:
        store = get_knowledge_store()
        emb = get_embedding()
        
        query_vector = emb.encode_queries([query])[0]
        results = store.search(query_vector, top_k=top_k)
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str):
    """删除知识"""
    try:
        store = get_knowledge_store()
        store.delete_by_ids([doc_id])
        return {"status": "success", "deleted_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 管理接口 ==========
@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    gpu_info = get_gpu_memory_info()
    return {
        "status": "healthy",
        "gpu": gpu_info,
        "timestamp": json.dumps({"now": "ok"})
    }


@app.get("/api/v1/stats")
async def get_stats():
    """系统统计信息"""
    try:
        store = get_knowledge_store()
        mem = get_memory()
        gpu_info = get_gpu_memory_info()
        
        return {
            "milvus": store.get_stats(),
            "memory": mem.get_stats(),
            "gpu": gpu_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversations")
async def list_conversations(limit: int = 100):
    """列出所有会话"""
    try:
        mem = get_memory()
        sessions = mem.get_sessions(limit=limit)
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversations/{session_id}")
async def get_conversation(session_id: str):
    """获取指定会话的历史"""
    try:
        mem = get_memory()
        history = mem.get_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversations/export")
async def export_conversations():
    """导出对话记录为Excel"""
    try:
        import pandas as pd
        from io import BytesIO
        from fastapi.responses import StreamingResponse
        
        mem = get_memory()
        
        # 获取所有会话
        sessions = mem.get_sessions(limit=1000)
        
        # 收集所有对话记录
        all_records = []
        for session_id in sessions:
            history = mem.get_history(session_id, limit=1000)
            for entry in history:
                all_records.append({
                    "session_id": session_id,
                    "role": entry["role"],
                    "content": entry["content"],
                    "intent": entry.get("intent", ""),
                    "timestamp": entry["timestamp"]
                })
        
        # 创建Excel
        df = pd.DataFrame(all_records)
        output = BytesIO()
        df.to_excel(output, index=False, sheet_name="对话记录")
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=conversations.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 启动入口 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=API["host"],
        port=API["port"],
        workers=API["workers"],
        reload=False
    )