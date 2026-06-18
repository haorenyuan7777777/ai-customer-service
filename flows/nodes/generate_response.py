"""
Promptflow节点：LLM回复生成
代理文件，调用 src.models.llm_client
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.llm_client import LLMClient


def generate_response(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """
    调用LLM生成回复
    
    Args:
        prompt: 提示词
        max_tokens: 最大token数
        temperature: 温度参数
    
    Returns:
        生成的文本
    """
    client = LLMClient()
    return client.generate(prompt, max_tokens=max_tokens, temperature=temperature)


try:
    from promptflow.core import tool
    
    @tool
    def main(prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        return generate_response(prompt, max_tokens, temperature)
except ImportError:
    pass