"""
意图识别模块 - 供Promptflow节点调用
封装规则+模型混合分类逻辑
"""
from typing import Dict
from src.models.intent_model import IntentClassifier

# 全局分类器实例
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def classify_intent(query: str) -> Dict:
    """
    分类用户意图（Promptflow节点入口）
    
    Args:
        query: 用户输入文本
    
    Returns:
        {
            "intent": str,          # 意图标签
            "confidence": float,    # 置信度
            "method": str,          # "rule" 或 "model"
            "probabilities": dict   # 各类别概率
        }
    """
    classifier = _get_classifier()
    return classifier.predict(query, use_rule=True)