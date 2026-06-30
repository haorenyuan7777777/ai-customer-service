"""
LlamaIndex完整RAG链路
- 直接对接Milvus Standalone（绕过MilvusVectorStore兼容性bug）
- 支持重排序、多文档组合、相似度阈值过滤
- 与Agent层集成，提供检索上下文
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# 项目内导入
from src.models.embedding_model import get_embedding_model
from src.rag.milvus_store import get_milvus_store, MilvusStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """RAG检索配置"""
    top_k: int = 5              # 初始检索数量
    similarity_cutoff: float = 0.7  # 相似度阈值（COSINE距离，越接近1越相似）
    rerank_top_n: int = 3       # 重排序后保留数量
    use_reranker: bool = True   # 是否启用重排序（内存充裕，启用）


class LlamaIndexRAG:
    """
    LlamaIndex RAG引擎（MilvusStore直连版）
    
    流程：
    1. 用户查询 → BGE-small-zh编码
    2. Milvus检索Top-K
    3. 相似度过滤（<0.7丢弃）
    4. 重排序（可选，提升相关性）
    5. 格式化上下文 → 返回Agent
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.embedding_model = get_embedding_model()  # BGE-small-zh GPU常驻
        self.milvus_store = get_milvus_store()

        # 预热Milvus（确保索引加载完成）
        self.milvus_store.warmup()
        logger.info("✅ Milvus预热完成")
        
        # 验证embedding_model类型（保留原检查逻辑）
        from llama_index.core.embeddings import BaseEmbedding as CoreBaseEmbedding
        if not isinstance(self.embedding_model, CoreBaseEmbedding):
            raise TypeError(
                f"Invalid embedding model: {type(self.embedding_model)}. "
                "Please check model loading logic."
            )
        
        logger.info("✅ LlamaIndex RAG引擎初始化完成")

    def retrieve(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        检索知识库，返回结构化上下文
        
        Args:
            query: 用户查询
            category: 业务分类过滤（如"售后支持"）
        
        Returns:
            检索结果列表，每项包含：
            - instruction: 匹配的问题
            - output: 标准答案
            - score: 相似度分数
            - category: 分类
        """
        # 统一走MilvusStore检索（绕过MilvusVectorStore bug）
        query_vector = self.embedding_model.get_text_embedding(query)
        raw_results = self.milvus_store.search(
            query_vector=query_vector,
            top_k=self.config.top_k,
            category=category
        )
        return self._format_milvus_results(raw_results)
    
    def _format_milvus_results(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """格式化Milvus原始结果"""
        formatted = []
        for r in results:
            score = r.get("distance", 0)
            if score < self.config.similarity_cutoff:
                continue
                
            formatted.append({
                "instruction": r.get("instruction", ""),
                "output": r.get("output", ""),
                "score": round(score, 4),
                "category": r.get("category", "unknown"),
                "source": r.get("intent", r.get("source", "")),  # 兼容 intent/source 两种字段名
                "id": r.get("id", 0)
            })
        return formatted
    
    def retrieve_with_rerank(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        带重排序的检索（内存充裕，提升准确性）
        
        实现：先检索Top-5，再用cross-encoder重排序，取Top-3
        【演示实现】：使用简单加权重排序（无额外模型）
        【生产目标】：集成bge-reranker-large模型
        """
        # 基础检索
        results = self.retrieve(query, category=category)
        
        if not self.config.use_reranker or len(results) <= self.config.rerank_top_n:
            return results[:self.config.rerank_top_n]
        
        # 【演示实现】简单重排序：按score降序 + 去重
        # 【生产目标】加载bge-reranker-large（需1.5G显存），交叉编码重排序
        seen_outputs = set()
        reranked = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            if r["output"] not in seen_outputs:
                seen_outputs.add(r["output"])
                reranked.append(r)
            if len(reranked) >= self.config.rerank_top_n:
                break
                
        return reranked
    
    def get_context_string(self, query: str, category: Optional[str] = None) -> str:
        """
        获取格式化上下文字符串，直接注入Prompt
        
        Returns:
            拼接的上下文文本，用于LLM生成
        """
        results = self.retrieve_with_rerank(query, category=category)
        
        if not results:
            return "未检索到相关知识。"
        
        contexts = []
        for i, r in enumerate(results, 1):
            contexts.append(
                f"[知识{i}] 问题：{r['instruction']}\n"
                f"答案：{r['output'][:500]}{'...' if len(r['output']) > 500 else ''}\n"
                f"相关度：{r['score']}"
            )
            
        return "\n\n".join(contexts)
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        category: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        批量检索（评测用）
        
        利用GPU批量编码加速
        """
        # 批量编码
        embeddings = self.embedding_model.get_text_embedding_batch(queries)
        
        results = []
        for emb in embeddings:
            raw = self.milvus_store.search(
                query_vector=emb,
                top_k=self.config.top_k,
                category=category
            )
            results.append(self._format_milvus_results(raw))
            
        return results


# 单例模式
_rag_engine: Optional[LlamaIndexRAG] = None

def get_rag_engine() -> LlamaIndexRAG:
    """获取RAG引擎单例"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = LlamaIndexRAG()
    return _rag_engine