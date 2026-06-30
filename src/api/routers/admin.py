"""
FastAPI管理后台路由
- 供Streamlit内部调用或外部API访问
- 知识库统计、对话记录查询、系统状态
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import sqlite3
from typing import List, Dict, Any
from datetime import datetime

from src.config import DATA_PATHS, DATABASE, MILVUS

router = APIRouter(prefix="/admin", tags=["管理后台"])


@router.get("/stats")
def get_system_stats() -> Dict[str, Any]:
    """获取系统统计概览"""
    # 知识库数量
    raw_path = Path(DATA_PATHS["raw"])
    knowledge_count = 0
    if raw_path.exists():
        with open(raw_path, 'r', encoding='utf-8') as f:
            knowledge_count = len(json.load(f))
    
    # 对话数量
    db_path = Path(DATABASE["path"])
    conversation_count = 0
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM chat_history")
        conversation_count = cursor.fetchone()[0]
        conn.close()
    
    return {
        "knowledge_count": knowledge_count,
        "conversation_count": conversation_count,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/conversations")
def get_conversations(
    user_id: str = None,
    session_id: str = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """查询对话记录"""
    db_path = Path(DATABASE["path"])
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    query = "SELECT * FROM chat_history WHERE 1=1"
    params = []
    
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


@router.delete("/conversations/clear")
def clear_conversations():
    """清空对话记录"""
    db_path = Path(DATABASE["path"])
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="数据库不存在")
    
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "对话记录已清空"}