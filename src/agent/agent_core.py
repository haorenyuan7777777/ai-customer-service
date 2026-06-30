"""
Agent核心逻辑
- 整合：意图识别 + 多轮记忆 + RAG检索 + 工具调用 + LLM生成
- 对接 Promptflow 流程引擎
- 支持流式响应（SSE）

【演示实现】：同步调用，单轮处理
【生产目标】：异步流式响应，多Agent协作
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List, Generator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agent.intent_classifier import get_intent_classifier
from src.agent.memory import get_memory_store
from src.agent.tools import get_tool_registry
from src.rag.llama_index_rag import get_rag_engine
from src.models.llm_client import get_llm_client
from src.promptflow.flow_engine import get_flow_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerServiceAgent:
    """
    智能客服Agent
    
    核心流程（对齐 flow.dag.yaml）：
    1. detect_intent: 意图识别
    2. load_memory: 加载多轮记忆
    3. retrieve_knowledge: RAG知识检索
    4. assemble_prompt: 组装Prompt
    5. generate_response: LLM生成
    6. save_memory: 保存对话
    """
    
    def __init__(self):
        self.intent_clf = get_intent_classifier()
        self.memory = get_memory_store()
        self.tools = get_tool_registry()
        self.rag = get_rag_engine()
        self.llm = get_llm_client()
        self.flow = get_flow_engine()
        
        logger.info("🤖 Agent初始化完成")
    
    def chat(
        self,
        user_message: str,
        user_id: str = "anonymous",
        session_id: str = "default",
        use_flow: bool = True
    ) -> Dict[str, Any]:
        """
        单轮对话（非流式）
        
        Args:
            user_message: 用户消息
            user_id: 用户标识
            session_id: 会话标识
            use_flow: 是否使用Promptflow引擎（True=完整流程，False=简化流程）
        
        Returns:
            {
                "response": "助手回复",
                "intent": "销售转化",
                "category": "sales",
                "context_used": "检索到的知识...",
                "execution_time_ms": 1234
            }
        """
        if use_flow:
            # 使用Promptflow完整流程
            return self.flow.run({
                "user_message": user_message,
                "user_id": user_id,
                "session_id": session_id
            })
        else:
            # 简化流程（直接调用，跳过YAML解析开销）
            return self._chat_direct(user_message, user_id, session_id)
    
    def _chat_direct(
        self,
        user_message: str,
        user_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """简化流程（直接调用各组件，无YAML开销）"""
        import time
        start = time.perf_counter()
        
        # 1. 意图识别
        intent_result = self.intent_clf.classify(user_message)
        
        # 2. 加载记忆
        memory_str = self.memory.get_formatted_history(user_id, session_id, limit=3)
        
        # 3. 知识检索
        category = intent_result.get("category")
        context = self.rag.get_context_string(user_message, category=category)
        
        # 4. 组装Prompt（简化版）
        prompt = self._build_prompt(
            user_message=user_message,
            intent=intent_result,
            memory=memory_str,
            context=context
        )
        
        # 5. LLM生成
        response = self.llm.generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7
        )
        
        # 6. 保存记忆（异步）
        self.memory.save_turn(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=response,
            intent=intent_result.get("intent", "标准客服")
        )
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return {
            "response": response,
            "intent": intent_result.get("intent"),
            "category": intent_result.get("category"),
            "context_used": context,
            "execution_time_ms": round(elapsed, 2)
        }
    
    def _build_prompt(
        self,
        user_message: str,
        intent: Dict,
        memory: str,
        context: str
    ) -> str:
        """简化版Prompt组装（无Jinja2依赖）"""
        intent_name = intent.get("intent", "标准客服")
        
        # 场景化前缀
        prefixes = {
            "销售转化": "【当前场景】销售咨询\n【处理策略】突出产品优势，引导成交\n",
            "技术支持": "【当前场景】技术支持\n【处理策略】分步骤排查，提供解决方案\n",
            "投诉处理": "【当前场景】投诉处理（高优先级）\n【处理策略】先安抚情绪，记录投诉，提供解决方案\n",
            "标准客服": "【当前场景】一般咨询\n【处理策略】直接回答，必要时引导至相关知识点\n"
        }
        
        prefix = prefixes.get(intent_name, prefixes["标准客服"])
        
        prompt = f"""你是一名专业的AI客服助手。

{prefix}
回答要求：
1. 基于提供的知识库上下文回答，不编造信息
2. 语气友好、专业、简洁
3. 如涉及价格/成交等敏感信息，标注"仅供参考，以实际为准"
4. 当前上下文限制：2048 tokens，请控制回复长度

"""

        if memory:
            prompt += f"""【历史对话】
{memory}

"""

        prompt += f"""【检索到的相关知识】
{context}

【用户问题】
{user_message}

请基于以上信息，给出专业回答："""

        return prompt
    
    def stream_chat(
        self,
        user_message: str,
        user_id: str = "anonymous",
        session_id: str = "default"
    ) -> Generator[str, None, None]:
        """
        流式对话（SSE）
        
        【演示实现】：逐字返回（模拟流式）
        【生产目标】：对接vLLM stream接口
        """
        result = self.chat(user_message, user_id, session_id)
        response = result["response"]
        
        # 模拟流式：逐字返回
        for char in response:
            yield char
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """执行工具调用"""
        return self.tools.call(tool_name, **kwargs)


# 单例
_agent: Optional[CustomerServiceAgent] = None

def get_agent() -> CustomerServiceAgent:
    """获取Agent单例"""
    global _agent
    if _agent is None:
        _agent = CustomerServiceAgent()
    return _agent


# 便捷入口
def chat(message: str, user_id: str = "anonymous", session_id: str = "default") -> str:
    """最简对话入口"""
    agent = get_agent()
    result = agent.chat(message, user_id, session_id)
    return result.get("response", "")