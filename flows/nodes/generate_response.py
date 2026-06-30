"""
LLM响应生成节点
- 调用vLLM（端口8000）
- 控制生成长度（受2048上下文限制）
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.llm_client import get_llm_client


def run(prompt: str, max_tokens: int = 512) -> str:
    """
    调用LLM生成响应
    
    Args:
        prompt: 完整Prompt
        max_tokens: 最大生成token数（受2048上下文限制）
    
    Returns:
        生成的回复文本
    """
    try:
        llm = get_llm_client()
        
        # 估算当前prompt长度（粗略：1token≈1.5中文字符）
        prompt_tokens = len(prompt) // 1.5
        
        # 动态调整max_tokens，确保不超过2048
        available_tokens = 2048 - int(prompt_tokens) - 50  # 留50token缓冲
        max_tokens = min(max_tokens, max(available_tokens, 100))  # 至少100token
        
        response = llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["用户：", "助手：", "\n\n"]  # 停止词，防止生成过多
        )
        
        return response
        
    except Exception as e:
        return f"抱歉，系统暂时无法响应，请稍后重试。（错误：{str(e)[:50]}）"


# """
# Promptflow节点：LLM回复生成
# 代理文件，调用 src.models.llm_client
# """
# import sys
# from pathlib import Path

# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

# from src.models.llm_client import LLMClient


# def generate_response(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
#     """
#     调用LLM生成回复
    
#     Args:
#         prompt: 提示词
#         max_tokens: 最大token数
#         temperature: 温度参数
    
#     Returns:
#         生成的文本
#     """
#     client = LLMClient()
#     return client.generate(prompt, max_tokens=max_tokens, temperature=temperature)


# try:
#     from promptflow.core import tool
    
#     @tool
#     def main(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
#         return generate_response(prompt, max_tokens, temperature)
# except ImportError:
#     pass