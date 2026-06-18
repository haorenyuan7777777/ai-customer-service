"""
AI Agent核心编排
整合意图识别、记忆、RAG检索、工具调用、LLM生成
"""
import json
from typing import Dict, List, Optional
from src.models.intent_model import IntentClassifier
from src.models.llm_client import LLMClient
from src.agent.memory import ChatMemory
from src.agent.tools import ToolRegistry
from src.rag.milvus_store import MilvusKnowledgeStore
from src.models.embedding_model import BGEEmbedding


class CustomerServiceAgent:
    """
    智能客服AI Agent
    
    处理流程:
    1. 意图识别 (bert-base-chinese)
    2. 加载记忆 (SQLite)
    3. RAG检索 (LlamaIndex + Milvus + BGE)
    4. 工具调用 (Python函数)
    5. LLM生成 (vLLM Qwen1.5B)
    6. 保存记忆
    """
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.memory = ChatMemory()
        self.tools = ToolRegistry()
        self.llm = LLMClient()
        self.embedding = BGEEmbedding()
        self.knowledge_store = MilvusKnowledgeStore()
    
    def process(self, query: str, session_id: str = "default") -> Dict:
        """处理用户查询 - 阶段3增强版"""
        
        # 1. 意图识别
        intent_result = self.intent_classifier.predict(query)
        intent = intent_result["intent"]
        
        # 2. 加载历史记忆
        history = self.memory.get_history(session_id)
        
        # 3. RAG检索（自适应上下文）
        rag_result = self.rag.adaptive_retrieve(
            query,
            max_context_tokens=1500
        )
        knowledge_results = rag_result["results"]
        context = rag_result["context"]
        
        # 4. 工具调用
        tool_result = None
        if intent in ["price_inquiry", "purchase_intent", "technical_issue"]:
            tool_result = self.tools.execute_by_intent(intent, {"product": query})
        
        # 5. 构建提示词并生成回复
        prompt = self._build_prompt(query, context, intent, history, tool_result)
        response = self.llm.generate(prompt, max_tokens=1024, temperature=0.7)
        
        # 6. 保存对话记忆
        self.memory.save(session_id, "user", query, intent=intent)
        self.memory.save(session_id, "assistant", response, intent=intent)
        
        return {
            "response": response,
            "intent": intent,
            "intent_confidence": intent_result["confidence"],
            "sources": knowledge_results,
            "tool_result": tool_result,
            "history_length": len(history),
            "context_tokens": rag_result["used_tokens"],
        }
    
    def _build_prompt(self, query: str, context: str, intent: str, 
                      history: List[Dict], tool_result: Optional[Dict]) -> str:
        """构建LLM提示词 - 阶段3增强版"""
        
        # 历史摘要（压缩）
        history_text = ""
        if history:
            history_parts = []
            for h in history[-2:]:
                role = "用户" if h["role"] == "user" else "助手"
                history_parts.append(f"{role}: {h['content'][:50]}...")
            history_text = "\n".join(history_parts)
        
        # 工具结果
        tool_text = ""
        if tool_result and tool_result.get("status") == "success":
            tool_text = f"工具查询结果: {json.dumps(tool_result['result'], ensure_ascii=False)}\n"
        
        return f"""你是专业的AI客服助手。请基于以下信息回答用户问题。
                ## 对话历史
                {history_text or "（无历史对话）"}

                ## 用户意图
                {intent}

                ## 相关知识
                {context}

                {tool_text}## 用户问题
                {query}

                请给出专业、简洁的回答（200字以内）：
                """