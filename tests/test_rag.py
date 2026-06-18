"""
RAG链路测试 - LlamaIndex + Milvus + BGE
"""
import pytest
import numpy as np
from src.models.embedding_model import BGEEmbedding
from src.rag.milvus_store import MilvusKnowledgeStore
from src.config import MILVUS


class TestRAGPipeline:
    """测试完整RAG链路"""
    
    @pytest.fixture(scope="class")
    def embedding_model(self):
        """BGE模型实例"""
        return BGEEmbedding()
    
    @pytest.fixture(scope="class")
    def store(self):
        """Milvus存储实例"""
        return MilvusKnowledgeStore(
            host=MILVUS["host"],
            port=MILVUS["port"],
            collection_name="test_rag",
            dim=MILVUS["dim"]
        )
    
    def test_embedding_consistency(self, embedding_model):
        """测试Embedding编码一致性"""
        texts = ["铅酸蓄电池", "铅酸蓄电池"]
        embeddings = embedding_model.encode(texts, batch_size=2, show_progress=False)
        
        # 相同文本应产生相同向量
        similarity = np.dot(embeddings[0], embeddings[1])
        assert similarity > 0.999, f"相同文本相似度: {similarity}"
        
        # 向量应已归一化
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 0.001, f"向量未归一化: {norm}"
        
        print(f"✅ Embedding一致性测试通过，维度: {embeddings.shape}")
    
    def test_end_to_end_retrieval(self, embedding_model, store):
        """测试端到端检索链路"""
        # 准备测试知识
        knowledge = [
            {"id": "rag_001", "instruction": "铅酸蓄电池正确使用的注意事项有哪些？", "output": "铅酸蓄电池使用时应注意..."},
            {"id": "rag_002", "instruction": "锂电池和铅酸电池哪个好？", "output": "锂电池能量密度高..."},
            {"id": "rag_003", "instruction": "电池充电需要多长时间？", "output": "一般充电8-10小时..."},
        ]
        
        # 向量化
        texts = [k["instruction"] for k in knowledge]
        vectors = embedding_model.encode(texts, batch_size=3, show_progress=False)
        
        # 插入Milvus
        store.insert(
            ids=[k["id"] for k in knowledge],
            vectors=vectors,
            instructions=[k["instruction"] for k in knowledge],
            outputs=[k["output"] for k in knowledge]
        )
        
        # 模拟查询
        query = "铅酸电池使用注意事项"
        query_vector = embedding_model.encode_queries([query])[0]
        
        # 检索
        results = store.search(query_vector, top_k=3)
        
        assert len(results) > 0
        assert results[0]["score"] > 0.5  # 应能检索到相关内容
        print(f"✅ 端到端检索测试通过")
        print(f"   查询: {query}")
        print(f"   Top1: {results[0]['instruction']} (相似度: {results[0]['score']:.4f})")
    
    def test_batch_embedding_performance(self, embedding_model):
        """测试批量向量化性能"""
        import time
        
        batch_sizes = [100, 200, 500]
        texts = [f"测试文本{i}" for i in range(500)]
        
        for bs in batch_sizes:
            start = time.time()
            embeddings = embedding_model.encode(texts[:bs], batch_size=bs, show_progress=False)
            elapsed = time.time() - start
            
            assert embeddings.shape == (bs, 512)
            print(f"✅ 批次{bs}: {elapsed:.3f}s ({bs/elapsed:.1f}条/秒)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])