"""
Milvus向量存储封装
- 对接Milvus Standalone（Docker部署）
- 支持HNSW索引、分区查询、批量检索
- 与LlamaIndex VectorStoreIndex兼容
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, 
    DataType, utility, MilvusException
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MilvusConfig:
    """Milvus连接配置"""
    host: str = "localhost"
    port: str = "19530"
    collection_name: str = "knowledge_base"
    dim: int = 512  # BGE-small-zh输出维度
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    
    # HNSW参数（内存充裕，追求速度）
    M: int = 16
    efConstruction: int = 200
    ef: int = 128
    
    # 连接别名 - 注意：MilvusVectorStore内部用MilvusClient，不用这个
    alias: str = "default"


class MilvusStore:
    """
    Milvus向量存储管理器
    功能：Collection管理、索引构建、向量检索、与LlamaIndex集成
    """
    
    def __init__(self, config: Optional[MilvusConfig] = None):
        self.config = config or MilvusConfig()
        self.collection: Optional[Collection] = None
        self._client = None  # MilvusClient 实例（用于LlamaIndex兼容）
        self._connect()
        
    def _connect(self):
        """建立Milvus连接（旧API，用于自定义search/insert）"""
        try:
            connections.connect(
                alias=self.config.alias,
                host=self.config.host,
                port=self.config.port
            )
            logger.info(f"✅ Milvus连接成功: {self.config.host}:{self.config.port}")
        except MilvusException as e:
            logger.error(f"❌ Milvus连接失败: {e}")
            raise
            
    def create_collection(self, drop_existing: bool = False) -> Collection:
        """
        创建Collection（阶段2已完成，此处用于重建）
        """
        collection_name = self.config.collection_name
        
        if utility.has_collection(collection_name):
            if drop_existing:
                logger.warning(f"删除已存在Collection: {collection_name}")
                utility.drop_collection(collection_name)
            else:
                logger.info(f"Collection已存在: {collection_name}")
                self.collection = Collection(collection_name)
                return self.collection
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.dim),
            FieldSchema(name="instruction", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="output", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="intent", dtype=DataType.VARCHAR, max_length=256),
        ]
        
        schema = CollectionSchema(fields, description="AI客服知识库")
        self.collection = Collection(collection_name, schema)
        
        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": {"M": self.config.M, "efConstruction": self.config.efConstruction}
        }
        self.collection.create_index(field_name="vector", index_params=index_params)
        logger.info(f"✅ Collection创建成功: {collection_name}, 索引: {self.config.index_type}")
        
        return self.collection
    
    def warmup(self):
        """预热：加载Collection到内存，避免首次查询延迟"""
        self.load_collection()
        print("✅ Milvus预热完成")

    def clear_collection(self):
        """清空Collection（删除后重建空表）"""
        collection_name = self.config.collection_name
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"🗑️ Collection已清空: {collection_name}")
        # 重建空Collection
        self.create_collection(drop_existing=False)
    
    def load_collection(self):
        """加载Collection到内存"""
        if self.collection is None:
            self.collection = Collection(self.config.collection_name)
        self.collection.load()
        logger.info(f"✅ Collection已加载: {self.config.collection_name}")
    
    def insert(
        self,
        ids: List[int],
        vectors: List[List[float]],
        instructions: List[str],
        outputs: List[str],
        intents: List[str],
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        批量插入数据到Milvus
        entities顺序必须与fields定义一致: [id, vector, instruction, output, category, intent]
        """
        if categories is None:
            categories = intents
        
        entities = [
            ids,
            vectors,
            instructions,
            outputs,
            categories,
            intents,
        ]
        
        self.load_collection()
        insert_result = self.collection.insert(entities)
        self.collection.flush()
        
        logger.info(f"✅ 插入完成: {len(ids)}条, 总计: {self.collection.num_entities}条")
        
        return {
            "insert_count": len(ids),
            "total_entities": self.collection.num_entities
        }
    
    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 3,
        category: Optional[str] = None,
        output_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        向量检索
        """
        # 【修复】schema实际字段是 intent 而非 source
        if output_fields is None:
            output_fields = ["id", "instruction", "output", "category", "intent"]
            
        self.load_collection()
        
        expr = f'category == "{category}"' if category else ""
        
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"ef": self.config.ef}
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields
        )
        
        hits = []
        for result in results[0]:
            hit = {
                "id": result.id,
                "distance": round(result.distance, 4),
                **{field: result.entity.get(field) for field in output_fields if field != "id"}
            }
            hits.append(hit)
            
        return hits
    
    def get_llama_index_vector_store(self) -> MilvusVectorStore:
        """
        获取LlamaIndex兼容的VectorStore
        
        【修复】MilvusVectorStore内部混用MilvusClient和connections API导致连接丢失
        解决方案：提前用connections.connect建立连接，确保Collection初始化时能拿到连接
        """
        from pymilvus import connections
        
        # 关键修复：先用旧API建立连接，确保MilvusVectorStore内部的Collection能复用
        # 注意：MilvusVectorStore会再次connect，但pymilvus允许重复connect同一alias
        try:
            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port
            )
        except Exception:
            pass  # 已连接或连接失败都继续
        
        return MilvusVectorStore(
            uri=f"http://{self.config.host}:{self.config.port}",
            collection_name=self.config.collection_name,
            dim=self.config.dim,
            overwrite=False,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Collection统计信息"""
        self.load_collection()
        stats = {
            "collection_name": self.config.collection_name,
            "total_entities": self.collection.num_entities,
            "num_entities": self.collection.num_entities,
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "dim": self.config.dim,
        }
        return stats
    
    def release(self):
        """释放Collection内存"""
        if self.collection:
            self.collection.release()
            logger.info("✅ Collection已释放")
    
    def close(self):
        """关闭连接"""
        connections.disconnect(self.config.alias)
        logger.info("✅ Milvus连接已关闭")


# 单例模式
_milvus_store: Optional[MilvusStore] = None

def get_milvus_store() -> MilvusStore:
    """获取MilvusStore单例"""
    global _milvus_store
    if _milvus_store is None:
        _milvus_store = MilvusStore()
    return _milvus_store