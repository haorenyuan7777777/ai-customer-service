"""
多轮记忆管理
- SQLite持久化
- 限制3轮（受2048上下文限制）
- 支持用户级、会话级隔离
"""

import os
import sqlite3
import json
import logging
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    """单轮对话"""
    user_message: str
    assistant_message: str
    timestamp: str
    intent: str = "标准客服"


class MemoryStore:
    """
    SQLite多轮记忆存储
    
    Schema:
    - id: 自增主键
    - user_id: 用户标识
    - session_id: 会话标识
    - user_message: 用户消息
    - assistant_message: 助手回复
    - intent: 意图标签
    - created_at: 创建时间
    """
    
    def __init__(self, db_path: str = "data/chat_memory.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """获取线程安全的连接"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """初始化数据库表"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL,
                intent TEXT DEFAULT '标准客服',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session 
            ON chat_history(user_id, session_id, created_at)
        """)
        
        conn.commit()
        logger.info(f"✅ 记忆数据库初始化完成: {self.db_path}")
    
    def save_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        intent: str = "标准客服"
    ):
        """保存单轮对话"""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO chat_history 
            (user_id, session_id, user_message, assistant_message, intent)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, session_id, user_message, assistant_message, intent))
        conn.commit()
    
    def get_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 3
    ) -> List[Dict]:
        """
        获取最近N轮对话历史
        """
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT user_message, assistant_message, intent, created_at
            FROM chat_history
            WHERE user_id = ? AND session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, session_id, limit))
        
        rows = cursor.fetchall()
        # 反转顺序（从早到晚）
        history = []
        for row in reversed(rows):
            history.append({
                "user_message": row["user_message"],
                "assistant_message": row["assistant_message"],
                "intent": row["intent"],
                "timestamp": row["created_at"]
            })
        
        return history
    
    def get_formatted_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 3
    ) -> str:
        """
        获取格式化的历史对话字符串（直接用于Prompt）
        """
        history = self.get_history(user_id, session_id, limit)
        
        if not history:
            return ""
        
        lines = []
        for turn in history:
            lines.append(f"用户：{turn['user_message']}")
            lines.append(f"助手：{turn['assistant_message']}")
        
        return "\n".join(lines)
    
    def clear_history(self, user_id: str, session_id: str):
        """清空指定会话历史"""
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM chat_history WHERE user_id = ? AND session_id = ?",
            (user_id, session_id)
        )
        conn.commit()
        logger.info(f"🧹 清空记忆: user={user_id}, session={session_id}")
    
    def get_stats(self) -> Dict:
        """获取记忆统计"""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_turns,
                COUNT(DISTINCT user_id) as total_users,
                COUNT(DISTINCT session_id) as total_sessions
            FROM chat_history
        """)
        row = cursor.fetchone()
        return {
            "total_turns": row["total_turns"],
            "total_users": row["total_users"],
            "total_sessions": row["total_sessions"]
        }
    
    def close(self):
        """关闭连接"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ========== 单例 ==========

_memory_store: Optional[MemoryStore] = None

def get_memory_store() -> MemoryStore:
    """获取记忆存储单例"""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


# ========== 兼容 Promptflow 节点入口 ==========

def run(user_id: str, session_id: str = "default", max_turns: int = 3) -> str:
    """Promptflow load_memory 节点入口"""
    store = get_memory_store()
    return store.get_formatted_history(user_id, session_id, max_turns)


def save(user_id: str, session_id: str, user_message: str, assistant_message: str):
    """Promptflow save_memory 节点入口"""
    store = get_memory_store()
    store.save_turn(user_id, session_id, user_message, assistant_message)