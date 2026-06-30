"""
知识检索节点（RAG）
- 调用LlamaIndex RAG引擎
- 支持按意图分类过滤
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rag.llama_index_rag import get_rag_engine


def run(query: str, intent: str = "标准客服", category: str = None) -> str:
    """
    检索相关知识库内容
    
    Args:
        query: 用户查询
        intent: 检测到的意图（用于选择检索策略）
        category: 业务分类过滤（可选）
    
    Returns:
        格式化上下文字符串
    """
    try:
        rag = get_rag_engine()
        
        # 根据意图调整检索参数
        top_k = 3
        if intent in ["投诉处理"]:
            top_k = 5  # 投诉需要更多上下文
        
        # 执行检索
        context = rag.get_context_string(query, category=category)
        
        return context
        
    except Exception as e:
        return f"知识检索失败：{str(e)}"
    

# """
# Promptflow节点：知识检索
# 代理文件，调用 src.rag.llama_index_rag
# """
# import sys
# from pathlib import Path

# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

# from src.rag.llama_index_rag import retrieve


# def retrieve_knowledge(query: str, intent: str = None, top_k: int = 5) -> list:
#     """
#     检索相关知识
    
#     Args:
#         query: 查询文本
#         intent: 意图过滤
#         top_k: 返回Top-K结果
    
#     Returns:
#         检索结果列表
#     """
#     return retrieve(query, intent=intent, top_k=top_k)


# try:
#     from promptflow.core import tool
    
#     @tool
#     def main(query: str, intent: str = None, top_k: int = 5) -> list:
#         return retrieve_knowledge(query, intent, top_k)
# except ImportError:
#     pass