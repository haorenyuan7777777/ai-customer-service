"""
记忆保存节点
- 将当前轮对话存入SQLite
- 异步执行，不阻塞响应返回
"""

import os
import sys
import threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agent.memory import get_memory_store


def run(user_id: str, session_id: str, user_message: str, assistant_message: str):
    """
    保存对话记忆（异步执行）
    """
    def _save():
        try:
            memory_store = get_memory_store()
            memory_store.save_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message
            )
        except Exception:
            pass
    
    thread = threading.Thread(target=_save)
    thread.daemon = True
    thread.start()
    
    return None