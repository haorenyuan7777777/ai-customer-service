"""
Promptflow节点：加载对话记忆
代理文件，调用 src.agent.memory
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent.memory import ChatMemory


def load_memory(session_id: str, limit: int = 3) -> list:
    """
    加载对话历史
    
    Args:
        session_id: 会话ID
        limit: 返回最近N条
    
    Returns:
        对话历史列表
    """
    memory = ChatMemory()
    return memory.get_history(session_id, limit=limit)


try:
    from promptflow.core import tool
    
    @tool
    def main(session_id: str, limit: int = 3) -> list:
        return load_memory(session_id, limit)
except ImportError:
    pass