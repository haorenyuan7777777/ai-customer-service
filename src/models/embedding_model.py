"""
BGE-small-zh Embedding模型封装 - ModelScope本地路径版
"""
import torch
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from pathlib import Path

from src.config import MODELS


class BGEEmbedding:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, device: str = "cuda"):
        if self._initialized:
            return
        
        model_path = MODELS["embedding"]["local_path"]
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # 检查本地路径是否存在
        if Path(model_path).exists():
            print(f"[BGE] 从ModelScope本地路径加载: {model_path}")
            self.model = SentenceTransformer(model_path, device=self.device)
        else:
            # 回退到在线加载（需网络）
            model_name = MODELS["embedding"]["name"]
            print(f"[BGE] 本地路径不存在，从HuggingFace下载: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
        
        self.dim = self.model.get_sentence_embedding_dimension()
        
        # 预热
        self.model.encode(["预热"], convert_to_tensor=True)
        
        print(f"[BGE] 加载完成 | 维度: {self.dim} | 设备: {self.device}")
        print(f"[BGE] 当前显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        self._initialized = True
    
    def encode(self, texts: List[str], batch_size: int = 500, show_progress: bool = True) -> np.ndarray:
        instruction = "为这个句子生成表示："
        texts = [instruction + t for t in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        instruction = "为这个句子生成表示："
        queries = [instruction + q for q in queries]
        
        return self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    @property
    def dimension(self) -> int:
        return self.dim


# 全局实例
embedding_model = BGEEmbedding()