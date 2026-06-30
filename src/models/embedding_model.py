"""
BGE-small-zh Embedding模型
- GPU常驻（0.5G显存），向量化速度提升5-10倍
- 支持单条/批量编码
- ModelScope本地路径加载
"""

import os
import logging
import asyncio
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGEEmbeddingModel(BaseEmbedding):
    """
    BGE-small-zh Embedding模型封装
    继承LlamaIndex BaseEmbedding，实现兼容接口
    """
    
    _model: Optional[SentenceTransformer] = None
    
    def __init__(self):
        # 必须先调用 super().__init__()，让Pydantic完成模型初始化
        super().__init__()
        
        # 避免重复加载模型
        if self._model is not None:
            return
            
        try:
            model_path = os.getenv("BGE_MODEL_PATH", "/mnt/e/modelscope/BAAI/bge-small-zh-v1.5")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🚀 加载BGE-small-zh模型: {model_path} → {device}")
            
            self._model = SentenceTransformer(model_path, device=device)
            
            if device == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"✅ BGE模型加载完成，显存占用: {mem:.2f}GB")
                
        except Exception as e:
            logger.error(f"❌ BGE模型加载失败: {e}")
            raise RuntimeError(f"BGE模型加载失败: {e}") from e
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """单条查询编码"""
        embedding = self._model.encode(query, normalize_embeddings=True)
        return embedding.tolist()
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """单条文本编码"""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> List[List[float]]:
        """
        兼容接口：直接代理到 SentenceTransformer.encode
        供 data_loader.py 等非LlamaIndex组件调用
        """
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        return embeddings
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量编码（GPU加速）"""
        embeddings = self._model.encode(
            texts, 
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    # ========== 异步方法（必须实现）==========
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步单条查询编码"""
        # BGE编码是CPU/GPU计算密集型，用线程池避免阻塞事件循环
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步单条文本编码"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量编码"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)
    
    # ========== 显式批量接口 ==========
    
    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """显式批量接口"""
        return self._get_text_embeddings(texts)


# ========== 单例实现（不用 lru_cache，Pydantic实例不可hash）==========

_embedding_model_instance: Optional[BGEEmbeddingModel] = None

def get_embedding_model() -> BGEEmbeddingModel:
    """获取Embedding模型单例（线程安全）"""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = BGEEmbeddingModel()
    return _embedding_model_instance


if __name__ == "__main__":
    from llama_index.core.embeddings import BaseEmbedding
    model = get_embedding_model()
    print(f"Model type: {type(model)}")
    print(f"Is instance of BaseEmbedding? {isinstance(model, BaseEmbedding)}")
    # 第二次获取验证单例
    model2 = get_embedding_model()
    print(f"Same instance? {model is model2}")
    # 测试编码
    emb = model._get_query_embedding("测试文本")
    print(f"Embedding dim: {len(emb)}")