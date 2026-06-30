"""
意图检测节点
- 关键词规则 + bert-base-chinese模型（GPU）
- 输出：intent（业务意图）、category（分类）、priority（优先级）
"""

import os
import sys
import yaml
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.intent_model import get_intent_model


def _load_intent_routing():
    """从 prompts.yaml 加载意图路由配置"""
    prompts_path = Path(__file__).parent.parent.parent / "configs" / "prompts.yaml"
    if prompts_path.exists():
        with open(prompts_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('intent_routing', {})
    # 兜底配置
    return {
        "销售转化": {"keywords": ["价格", "优惠", "活动", "推荐", "对比", "买", "下单"], "priority": 2},
        "技术支持": {"keywords": ["故障", "无法", "报错", "怎么修", "安装", "使用"], "priority": 3},
        "投诉处理": {"keywords": ["投诉", "不满", "差评", "退款", "赔偿", "垃圾"], "priority": 1},
        "标准客服": {"keywords": [], "priority": 4},
    }


def run(message: str) -> dict:
    """检测用户意图"""
    routing = _load_intent_routing()
    
    detected_keywords = []
    matched_intent = None
    highest_priority = 999
    
    for intent_name, intent_config in routing.items():
        keywords = intent_config.get('keywords', [])
        for kw in keywords:
            if kw in message:
                detected_keywords.append(kw)
                if intent_config.get('priority', 4) < highest_priority:
                    highest_priority = intent_config.get('priority', 4)
                    matched_intent = intent_name
    
    if matched_intent and matched_intent != "标准客服":
        return {
            "intent": matched_intent,
            "category": matched_intent.lower().replace('处理', '').replace('转化', ''),
            "priority": highest_priority,
            "keywords": detected_keywords,
            "confidence": 0.85
        }
    
    # 关键词未命中，调用模型
    try:
        model = get_intent_model()
        prediction = model.predict(message)
        
        intent_mapping = {
            "general_query": "标准客服",
            "price_inquiry": "销售转化",
            "purchase_intent": "销售转化",
            "technical_issue": "技术支持",
            "complaint": "投诉处理",
        }
        
        model_intent = prediction.get('intent', 'general_query')
        mapped_intent = intent_mapping.get(model_intent, "标准客服")
        
        return {
            "intent": mapped_intent,
            "category": model_intent,
            "priority": routing.get(mapped_intent, {}).get('priority', 4),
            "keywords": [],
            "confidence": prediction.get('confidence', 0.5)
        }
    except Exception as e:
        return {
            "intent": "标准客服",
            "category": "general",
            "priority": 4,
            "keywords": [],
            "confidence": 1.0,
            "error": str(e)
        }
    
# """
# Promptflow节点：意图识别
# 代理文件，调用 src.agent.intent_classifier
# """
# import sys
# from pathlib import Path

# # 添加项目根目录到Python路径
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

# from src.agent.intent_classifier import classify_intent


# def detect_intent(query: str) -> dict:
#     """
#     检测用户意图
    
#     Args:
#         query: 用户输入
    
#     Returns:
#         意图识别结果字典
#     """
#     return classify_intent(query)


# # Promptflow兼容入口（如果未来迁移到Azure Promptflow）
# try:
#     from promptflow.core import tool
    
#     @tool
#     def main(query: str) -> dict:
#         return detect_intent(query)
# except ImportError:
#     pass