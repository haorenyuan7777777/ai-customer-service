"""
Promptflow节点：意图识别
代理文件，调用 src.agent.intent_classifier
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent.intent_classifier import classify_intent


def detect_intent(query: str) -> dict:
    """
    检测用户意图
    
    Args:
        query: 用户输入
    
    Returns:
        意图识别结果字典
    """
    return classify_intent(query)


# Promptflow兼容入口（如果未来迁移到Azure Promptflow）
try:
    from promptflow.core import tool
    
    @tool
    def main(query: str) -> dict:
        return detect_intent(query)
except ImportError:
    pass