"""
LlamaIndex RAG链封装 - 阶段3增强版
整合：VectorStoreIndex + BGE Embedding + Milvus + 重排序 + 上下文压缩
"""
from typing import List, Dict, Optional
import numpy as np
import time

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.embeddings import BaseEmbedding

from src.models.embedding_model import BGEEmbedding
from src.config import MILVUS


class BGEEmbeddingAdapter(BaseEmbedding):
    """
    LlamaIndex Embedding适配器
    将我们的BGEEmbedding包装为LlamaIndex兼容的Embedding类
    """
    
    def __init__(self, bge_model: BGEEmbedding = None):
        self._bge = bge_model or BGEEmbedding()
        super().__init__()
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询嵌入"""
        embedding = self._bge.encode_queries([query])[0]
        return embedding.tolist()
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        embedding = self._bge.encode([text], show_progress=False)[0]
        return embedding.tolist()
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入"""
        embeddings = self._bge.encode(texts, show_progress=False)
        return embeddings.tolist()
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


class LlamaIndexRAG:
    """
    LlamaIndex官方RAG检索链
    """
    
    def __init__(self, top_k: int = 5, rerank_top_k: int = 3):
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        # 初始化Embedding模型（适配器）
        self.embedding = BGEEmbedding()
        self.embed_adapter = BGEEmbeddingAdapter(self.embedding)
        
        # 关键修复：设置LlamaIndex全局Embedding为本地BGE
        Settings.embed_model = self.embed_adapter
        
        # 初始化LlamaIndex Milvus向量存储
        self.vector_store = MilvusVectorStore(
            uri=f"http://{MILVUS['host']}:{MILVUS['port']}",
            collection_name=MILVUS["collection_name"],
            dim=MILVUS["dim"],
            overwrite=False,
        )
        
        # 构建VectorStoreIndex
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
        )
        
        # 配置检索器
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k * 2,
        )
        
        # 相似度过滤
        self.similarity_processor = SimilarityPostprocessor(similarity_cutoff=0.5)
        
        print(f"[RAG] LlamaIndex索引构建完成")
        print(f"[RAG] 初检索Top-{top_k*2} → 重排序Top-{rerank_top_k}")
    
    def retrieve(self, query: str, intent_filter: str = None) -> List[Dict]:
        """检索相关知识"""
        start_time = time.time()
        
        # 1. LlamaIndex检索
        nodes = self.retriever.retrieve(query)
        
        # 2. 相似度过滤
        nodes = self.similarity_processor.postprocess_nodes(nodes)
        
        # 3. 转换为标准格式
        results = []
        for node in nodes:
            results.append({
                "id": node.node_id,
                "score": float(node.score) if hasattr(node, 'score') else 0.0,
                "instruction": node.text,
                "output": node.metadata.get("output", ""),
                "intent": node.metadata.get("intent", "general_query"),
            })
        
        # 4. 意图过滤
        if intent_filter and intent_filter != "general_query":
            results = [r for r in results if r["intent"] == intent_filter]
        
        # 5. 重排序
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        results = results[:self.rerank_top_k]
        
        elapsed = time.time() - start_time
        print(f"[RAG] 检索完成: {len(results)}条 / {elapsed*1000:.1f}ms")
        
        return results
    
    def format_context(self, results: List[Dict], max_length: int = 200) -> str:
        """格式化检索结果为上下文字符串"""
        if not results:
            return "暂无相关知识"
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            output = doc["output"]
            if len(output) > max_length:
                output = output[:max_length] + "..."
            
            context_parts.append(
                f"[来源{i}][相似度{doc['score']:.3f}]\n"
                f"问题: {doc['instruction']}\n"
                f"答案: {output}"
            )
        
        return "\n\n".join(context_parts)
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本token数"""
        return int(len(text) / 1.5)
    
    def adaptive_retrieve(self, query: str, max_context_tokens: int = 1500) -> Dict:
        """自适应检索"""
        all_results = self.retrieve(query, intent_filter=None)
        
        selected = []
        total_tokens = 0
        
        for doc in all_results:
            doc_text = f"问题: {doc['instruction']}\n答案: {doc['output']}"
            doc_tokens = self.estimate_tokens(doc_text)
            
            if total_tokens + doc_tokens > max_context_tokens:
                break
            
            selected.append(doc)
            total_tokens += doc_tokens
        
        context = self.format_context(selected)
        
        return {
            "context": context,
            "results": selected,
            "used_tokens": total_tokens,
            "total_available": len(all_results)
        }


# 便捷函数
def retrieve(query: str, intent: str = None, top_k: int = 5) -> List[Dict]:
    rag = LlamaIndexRAG(top_k=top_k)
    return rag.retrieve(query, intent_filter=intent)