"""
Promptflow节点：知识检索
代理文件，调用 src.rag.llama_index_rag
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.llama_index_rag import retrieve


def retrieve_knowledge(query: str, intent: str = None, top_k: int = 5) -> list:
    """
    检索相关知识
    
    Args:
        query: 查询文本
        intent: 意图过滤
        top_k: 返回Top-K结果
    
    Returns:
        检索结果列表
    """
    return retrieve(query, intent=intent, top_k=top_k)


try:
    from promptflow.core import tool
    
    @tool
    def main(query: str, intent: str = None, top_k: int = 5) -> list:
        return retrieve_knowledge(query, intent, top_k)
except ImportError:
    pass