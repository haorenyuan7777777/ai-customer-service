"""
知识库管理API
- CRUD操作
- 批量向量化
- 检索测试
"""

import os
import time
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.rag.milvus_store import get_milvus_store
from src.models.embedding_model import get_embedding_model
from src.rag.data_loader import load_alpaca_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["知识库管理"])


# ============ 数据模型 ============

class KnowledgeItem(BaseModel):
    id: int
    instruction: str
    output: str
    category: str = "标准客服"
    intent: str = "标准客服"


class KnowledgeCreate(BaseModel):
    instruction: str
    output: str
    category: str = "标准客服"
    intent: str = "标准客服"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    category: Optional[str] = None


# ============ API端点 ============

@router.get("/stats")
async def get_stats():
    """获取知识库统计"""
    try:
        store = get_milvus_store()
        stats = store.get_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_knowledge(request: SearchRequest):
    """检索知识库"""
    try:
        store = get_milvus_store()
        emb_model = get_embedding_model()
        
        vector = emb_model.get_text_embedding(request.query)
        results = store.search(
            query_vector=vector,
            top_k=request.top_k,
            category=request.category
        )
        
        return {
            "success": True,
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add")
async def add_knowledge(item: KnowledgeCreate):
    """添加单条知识"""
    try:
        emb_model = get_embedding_model()
        store = get_milvus_store()
        
        # 编码
        vector = emb_model.get_text_embedding(item.instruction)
        
        # 生成ID
        new_id = int(time.time() * 1000)
        
        # 插入
        from pymilvus import Collection
        collection = Collection(store.config.collection_name)
        collection.insert([{
            "id": new_id,
            "vector": vector,
            "instruction": item.instruction,
            "output": item.output,
            "category": item.category,
            "intent": item.intent
        }])
        
        return {
            "success": True,
            "data": {"id": new_id}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch_import")
async def batch_import(file: UploadFile = File(...)):
    """批量导入JSON"""
    try:
        content = await file.read()
        data = json.loads(content)
        
        if not isinstance(data, list):
            raise ValueError("JSON必须是数组格式")
        
        emb_model = get_embedding_model()
        store = get_milvus_store()
        
        batch_size = 100
        total = len(data)
        inserted = 0
        
        for i in range(0, total, batch_size):
            batch = data[i:i+batch_size]
            instructions = [item["instruction"] for item in batch]
            vectors = emb_model.get_text_embedding_batch(instructions)
            
            entities = []
            for j, item in enumerate(batch):
                entities.append({
                    "id": item.get("id", int(time.time() * 1000) + j),
                    "vector": vectors[j],
                    "instruction": item["instruction"],
                    "output": item["output"],
                    "category": item.get("category", "标准客服"),
                    "intent": item.get("intent", "标准客服")
                })
            
            from pymilvus import Collection
            collection = Collection(store.config.collection_name)
            collection.insert(entities)
            inserted += len(batch)
        
        return {
            "success": True,
            "data": {"total": total, "inserted": inserted}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{item_id}")
async def delete_knowledge(item_id: int):
    """删除知识条目"""
    try:
        store = get_milvus_store()
        from pymilvus import Collection
        collection = Collection(store.config.collection_name)
        collection.delete(f"id == {item_id}")
        
        return {
            "success": True,
            "data": {"deleted_id": item_id}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebuild")
async def rebuild_index():
    """重建索引（数据量大时慎用）"""
    try:
        store = get_milvus_store()
        collection = store.collection
        
        # 释放并重新加载
        collection.release()
        collection.load()
        
        return {
            "success": True,
            "message": "索引重建完成"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))