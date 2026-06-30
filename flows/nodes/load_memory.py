"""
记忆加载节点
- 从SQLite加载多轮对话历史
- 限制3轮（受2048上下文限制）
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agent.memory import get_memory_store


def run(user_id: str, session_id: str = "default", max_turns: int = 3) -> str:
    """
    加载用户历史记忆
    
    Returns:
        格式化记忆字符串
    """
    try:
        memory_store = get_memory_store()
        return memory_store.get_formatted_history(user_id, session_id, max_turns)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"加载记忆失败: {e}")
        return ""