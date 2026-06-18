"""
Milvus向量存储封装
- 单Collection设计（11157条规模）
- Schema: id, vector, instruction, output, category, intent
"""
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
import numpy as np
from typing import List, Dict, Optional
import json

class MilvusKnowledgeStore:
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "knowledge_base",
        dim: int = 512
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        
        self._connect()
        self._ensure_collection()
    
    def _connect(self):
        connections.connect("default", host=self.host, port=self.port)
        print(f"[Milvus] 已连接: {self.host}:{self.port}")
    
    def _ensure_collection(self):
        """确保Collection存在，不存在则创建"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            print(f"[Milvus] Collection已存在: {self.collection_name}")
            print(f"[Milvus] 实体数量: {self.collection.num_entities}")
            return
        
        # 定义Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="instruction", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="output", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=64),
        ]
        
        schema = CollectionSchema(fields, description="AI客服知识库")
        self.collection = Collection(self.collection_name, schema)
        
        # 创建HNSW索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 128}
        }
        self.collection.create_index("vector", index_params)
        print(f"[Milvus] Collection创建完成: {self.collection_name}")
    
    def insert(
        self,
        ids: List[str],
        vectors: np.ndarray,
        instructions: List[str],
        outputs: List[str],
        categories: List[str] = None,
        intents: List[str] = None
    ):
        """批量插入数据"""
        if categories is None:
            categories = ["general"] * len(ids)
        if intents is None:
            intents = ["general_query"] * len(ids)
        
        entities = [
            ids,
            vectors.tolist(),
            instructions,
            outputs,
            categories,
            intents
        ]
        
        self.collection.insert(entities)
        self.collection.flush()
        print(f"[Milvus] 插入 {len(ids)} 条数据，当前总数: {self.collection.num_entities}")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        intent_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        向量相似度检索
        
        Args:
            query_vector: 查询向量 (dim,)
            top_k: 返回Top-K结果
            intent_filter: 按意图过滤（可选）
        """
        self.collection.load()
        
        # 构建过滤表达式
        expr = None
        if intent_filter:
            expr = f'intent == "{intent_filter}"'
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["instruction", "output", "category", "intent"]
        )
        
        # 格式化结果
        hits = []
        for hit in results[0]:
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "instruction": hit.entity.get("instruction"),
                "output": hit.entity.get("output"),
                "category": hit.entity.get("category"),
                "intent": hit.entity.get("intent"),
            })
        return hits
    
    def delete_by_ids(self, ids: List[str]):
        """按ID删除"""
        expr = f'id in {json.dumps(ids)}'
        self.collection.delete(expr)
        self.collection.flush()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "collection_name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "dim": self.dim,
            "index_type": "HNSW"
        }
    
    def clear_collection(self):
        """清空Collection中的所有数据（重建）"""
        from pymilvus import utility
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"[Milvus] 已删除旧Collection: {self.collection_name}")
        self.collection = None
        self._ensure_collection()
        print(f"[Milvus] Collection已重建")


# 全局实例（延迟初始化）
milvus_store = None

def get_milvus_store():
    global milvus_store
    if milvus_store is None:
        from src.config import MILVUS
        milvus_store = MilvusKnowledgeStore(
            host=MILVUS["host"],
            port=MILVUS["port"],
            collection_name=MILVUS["collection_name"],
            dim=MILVUS["dim"]
        )
    return milvus_store