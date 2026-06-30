"""
Prompt组装节点
- 根据意图选择模板
- 使用Jinja2渲染最终Prompt
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.promptflow.flow_engine import get_flow_engine


def run(
    user_message: str,
    intent: dict,
    memory: str,
    context: str
) -> str:
    """
    组装最终Prompt
    
    Args:
        user_message: 用户原始消息
        intent: 意图检测结果（含intent、category、keywords等）
        memory: 格式化记忆字符串
        context: 检索到的知识上下文
    
    Returns:
        完整Prompt字符串（直接送入LLM）
    """
    try:
        engine = get_flow_engine()
        
        # 选择模板
        template_name = intent.get('intent', '标准客服').lower()
        
        # 映射到模板名
        template_map = {
            '销售转化': 'sales',
            '技术支持': 'technical_support',
            '投诉处理': 'complaint',
            '标准客服': 'customer_service'
        }
        template_name = template_map.get(intent.get('intent'), 'customer_service')
        
        # 准备模板变量
        template_vars = {
            'user_message': user_message,
            'memory': memory,
            'context': context,
            'keywords': ', '.join(intent.get('keywords', [])),
            'issue_type': intent.get('category', 'general')
        }
        
        # 渲染模板
        prompt = engine.render_template(template_name, **template_vars)
        
        return prompt
        
    except Exception as e:
        # 兜底：直接拼接
        return f"""请回答用户问题。

用户问题：{user_message}

相关知识：{context}

记忆：{memory}

请给出专业回答："""