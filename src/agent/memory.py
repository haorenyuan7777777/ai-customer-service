"""
多轮对话记忆模块 - SQLite实现
支持多会话隔离、历史限制、摘要压缩
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from src.config import DATABASE


class ChatMemory:
    """SQLite多轮记忆管理"""
    
    def __init__(self, db_path: str = None, max_history: int = None):
        self.db_path = db_path or DATABASE["path"]
        self.max_history = max_history or DATABASE["max_history_rounds"]
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    intent TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # 创建索引加速查询
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session 
                ON chat_history(session_id, timestamp)
            """)
            conn.commit()
    
    def save(self, session_id: str, role: str, content: str, 
             intent: str = None, metadata: dict = None):
        """
        保存对话记录
        
        Args:
            session_id: 会话ID
            role: user 或 assistant
            content: 对话内容
            intent: 意图标签（可选）
            metadata: 额外元数据（可选）
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO chat_history 
                   (session_id, role, content, intent, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, content, intent, 
                 json.dumps(metadata) if metadata else None)
            )
            conn.commit()
    
    def get_history(self, session_id: str, limit: int = None) -> List[Dict]:
        """
        获取对话历史
        
        Args:
            session_id: 会话ID
            limit: 返回最近N条，None则使用默认限制
        
        Returns:
            对话历史列表，每条包含 role/content/timestamp
        """
        limit = limit or self.max_history
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT role, content, intent, timestamp, metadata
                   FROM chat_history
                   WHERE session_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (session_id, limit)
            )
            
            rows = cursor.fetchall()
            # 反转回时间顺序
            rows = list(reversed(rows))
            
            history = []
            for row in rows:
                entry = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                }
                if row["intent"]:
                    entry["intent"] = row["intent"]
                if row["metadata"]:
                    entry["metadata"] = json.loads(row["metadata"])
                history.append(entry)
            
            return history
    
    def get_sessions(self, limit: int = 100) -> List[str]:
        """获取所有会话ID列表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT DISTINCT session_id 
                   FROM chat_history 
                   ORDER BY MAX(timestamp) DESC
                   LIMIT ?""",
                (limit,)
            )
            return [row[0] for row in cursor.fetchall()]
    
    def clear_session(self, session_id: str):
        """清空指定会话的历史"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
    
    def get_stats(self) -> Dict:
        """获取记忆统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM chat_history"
            ).fetchone()[0]
            
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM chat_history"
            ).fetchone()[0]
            
            return {
                "total_messages": total,
                "total_sessions": sessions,
                "db_path": self.db_path
            }
    
    def compress_history(self, session_id: str) -> str:
        """
        历史摘要压缩（当历史超过限制时）
        生产目标：调用LLM生成摘要
        演示实现：简单拼接前N条关键信息
        """
        history = self.get_history(session_id, limit=10)
        if len(history) <= 3:
            return None
        
        # 演示实现：提取关键信息拼接
        summary_parts = []
        for entry in history:
            if entry["role"] == "user":
                summary_parts.append(f"用户问: {entry['content'][:30]}...")
        
        summary = "历史摘要: " + "; ".join(summary_parts[:3])
        return summary