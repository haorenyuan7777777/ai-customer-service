"""
Milvus连接与向量存储测试
"""
import pytest
import numpy as np
from src.rag.milvus_store import MilvusKnowledgeStore
from src.config import MILVUS


class TestMilvusConnection:
    """测试Milvus连接与基础操作"""
    
    @pytest.fixture(scope="class")
    def store(self):
        """创建测试用的Milvus存储实例"""
        return MilvusKnowledgeStore(
            host=MILVUS["host"],
            port=MILVUS["port"],
            collection_name="test_knowledge_base",
            dim=MILVUS["dim"]
        )
    
    def test_connection(self, store):
        """测试Milvus连接是否成功"""
        stats = store.get_stats()
        assert stats["collection_name"] == "test_knowledge_base"
        assert stats["dim"] == 512
        print(f"✅ Milvus连接成功: {stats}")
    
    def test_insert_and_search(self, store):
        """测试插入和检索"""
        # 清理测试数据
        from pymilvus import utility
        if utility.has_collection("test_knowledge_base"):
            from pymilvus import Collection
            c = Collection("test_knowledge_base")
            c.drop()
            store._ensure_collection()
        
        # 插入测试数据
        test_ids = ["test_001", "test_002", "test_003"]
        test_vectors = np.random.randn(3, 512).astype(np.float32)
        test_vectors = test_vectors / np.linalg.norm(test_vectors, axis=1, keepdims=True)
        test_instructions = ["测试问题1", "测试问题2", "测试问题3"]
        test_outputs = ["测试答案1", "测试答案2", "测试答案3"]
        
        store.insert(
            ids=test_ids,
            vectors=test_vectors,
            instructions=test_instructions,
            outputs=test_outputs
        )
        
        # 检索测试
        results = store.search(test_vectors[0], top_k=3)
        assert len(results) == 3
        assert results[0]["id"] == "test_001"
        assert results[0]["score"] > 0.99  # 自身检索相似度应接近1
        print(f"✅ 插入和检索测试通过，Top1相似度: {results[0]['score']:.4f}")
    
    def test_intent_filter(self, store):
        """测试按意图过滤检索"""
        test_ids = ["filter_001", "filter_002"]
        test_vectors = np.random.randn(2, 512).astype(np.float32)
        test_vectors = test_vectors / np.linalg.norm(test_vectors, axis=1, keepdims=True)
        
        store.insert(
            ids=test_ids,
            vectors=test_vectors,
            instructions=["价格问题", "技术问题"],
            outputs=["价格答案", "技术答案"],
            intents=["price_inquiry", "technical_issue"]
        )
        
        # 按意图过滤
        results = store.search(test_vectors[0], top_k=10, intent_filter="price_inquiry")
        assert all(r["intent"] == "price_inquiry" for r in results)
        print(f"✅ 意图过滤测试通过，返回{len(results)}条结果")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])