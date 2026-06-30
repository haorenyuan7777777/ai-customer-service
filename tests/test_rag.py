"""
RAG检索测试
- Top-K准确率测试（测试集2,232条）
- 响应延迟测试（P50/P95/P99）
- 显存稳定性测试
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict
from statistics import mean, median
import random

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.llama_index_rag import LlamaIndexRAG, RAGConfig
from src.models.embedding_model import get_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGTester:
    """RAG评测器"""
    
    def __init__(self, test_data_path: str = "data/processed/test.json"):
        self.rag = LlamaIndexRAG(RAGConfig(top_k=5, similarity_cutoff=0.7))
        self.test_data = self._load_test_data(test_data_path)
        
    def _load_test_data(self, path: str) -> List[Dict]:
        """加载测试集"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"📊 加载测试集: {len(data)}条")
        return data
    
    def test_top_k_accuracy(self, k: int = 3, sample_size: int = 500) -> Dict:
        """
        Top-K准确率测试
        
        判定标准：检索结果中是否包含与查询语义匹配的答案
        【演示实现】：用instruction精确匹配判定（简化）
        【生产目标】：用LLM判定语义等价性
        """
        import random
        sample = random.sample(self.test_data, min(sample_size, len(self.test_data)))
        
        correct = 0
        total = 0
        
        for item in sample:
            query = item["instruction"]
            expected_output = item["output"]
            
            # 检索
            results = self.rag.retrieve(query)
            top_k_outputs = [r["output"] for r in results[:k]]
            
            # 判定：预期答案是否在Top-K中（简化：包含关键短语）
            # 【演示实现】简单包含判定
            is_hit = any(
                expected_output[:100] in out or out[:100] in expected_output
                for out in top_k_outputs
            )
            
            if is_hit:
                correct += 1
            total += 1
            
            if total % 100 == 0:
                logger.info(f"  进度: {total}/{len(sample)}, 当前准确率: {correct/total:.2%}")
        
        accuracy = correct / total if total > 0 else 0
        return {
            "metric": f"Top-{k} Accuracy",
            "sample_size": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "status": "✅ PASS" if accuracy > 0.6 else "⚠️ NEED IMPROVE"
        }
    
    def test_latency(self, num_queries: int = 100) -> Dict:
        """
        响应延迟测试
        
        测试端到端检索延迟（编码+检索+格式化）
        """
        import random
        sample = random.sample(self.test_data, min(num_queries, len(self.test_data)))
        
        latencies = []
        
        for item in sample:
            query = item["instruction"]
            
            start = time.perf_counter()
            _ = self.rag.retrieve(query)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # 转毫秒
        
        latencies.sort()
        
        p50 = latencies[len(latencies)//2]
        p95 = latencies[int(len(latencies)*0.95)]
        p99 = latencies[int(len(latencies)*0.99)] if len(latencies) >= 100 else max(latencies)
        
        return {
            "metric": "Retrieval Latency",
            "sample_size": len(latencies),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "mean_ms": round(mean(latencies), 2),
            "status": "✅ PASS" if p95 < 500 else "⚠️ NEED OPTIMIZE"
        }
    
    def test_gpu_memory_stability(self, duration_sec: int = 60) -> Dict:
        """
        GPU显存稳定性测试
        
        持续运行检索，监控显存波动
        """
        import torch
        
        initial_mem = torch.cuda.memory_allocated() / 1024**3
        peak_mem = initial_mem
        
        start_time = time.time()
        query_count = 0
        
        while time.time() - start_time < duration_sec:
            query = random.choice(self.test_data)["instruction"]
            _ = self.rag.retrieve(query)
            query_count += 1
            
            current = torch.cuda.memory_allocated() / 1024**3
            peak_mem = max(peak_mem, current)
        
        return {
            "metric": "GPU Memory Stability",
            "duration_sec": duration_sec,
            "queries_executed": query_count,
            "initial_gb": round(initial_mem, 2),
            "peak_gb": round(peak_mem, 2),
            "growth_gb": round(peak_mem - initial_mem, 2),
            "status": "✅ PASS" if (peak_mem - initial_mem) < 0.5 else "⚠️ LEAK DETECTED"
        }
    
    def run_all_tests(self) -> Dict:
        """运行全部测试"""
        logger.info("=" * 50)
        logger.info("🧪 RAG检索评测开始")
        logger.info("=" * 50)
        
        results = {
            "top3_accuracy": self.test_top_k_accuracy(k=3, sample_size=500),
            "latency": self.test_latency(num_queries=100),
            "gpu_stability": self.test_gpu_memory_stability(duration_sec=60)
        }
        
        # 输出报告
        logger.info("\n" + "=" * 50)
        logger.info("📋 RAG评测报告")
        logger.info("=" * 50)
        for name, result in results.items():
            logger.info(f"\n【{name}】")
            for k, v in result.items():
                logger.info(f"  {k}: {v}")
        
        # 总评
        all_pass = all(r.get("status", "").startswith("✅") for r in results.values())
        logger.info(f"\n{'=' * 50}")
        logger.info(f"总评: {'✅ 全部通过' if all_pass else '⚠️ 存在待优化项'}")
        logger.info(f"{'=' * 50}")
        
        return results


if __name__ == "__main__":
    tester = RAGTester()
    tester.run_all_tests()

    # scripts/diagnose_rag.py
    # import time
    # import sys, os
    # sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


    # from tests.test_rag import RAGTester
    # tester = RAGTester()
    # # 单独运行每项测试，查看详细结果
    # print("=" * 50)
    # print("【1/3】Top-3 准确率诊断")
    # acc = tester.test_top_k_accuracy(k=3, sample_size=100)  # 先用100条快速诊断
    # print(f"准确率: {acc['accuracy']}, 状态: {acc['status']}")
    # time.sleep(5)
    # print("\n" + "=" * 50)
    # print("【2/3】延迟诊断")
    # lat = tester.test_latency(num_queries=20)
    # print(f"P95: {lat['p95_ms']}ms, 状态: {lat['status']}")
    # time.sleep(5)
    # print("\n" + "=" * 50)
    # print("【3/3】显存稳定性诊断")
    # gpu = tester.test_gpu_memory_stability(duration_sec=10)
    # print(f"增长: {gpu['growth_gb']}GB, 状态: {gpu['status']}")


    # from src.rag.milvus_store import get_milvus_store
    # store = get_milvus_store()
    # stats = store.get_stats()
    # print(f"Collection: {stats['collection_name']}")
    # print(f"总数据量: {stats['total_entities']}")
    # print(f"索引类型: {stats['index_type']}")


    # # 保存为 test_single.py
    # from src.rag.llama_index_rag import get_rag_engine

    # rag = get_rag_engine()

    # # 用测试集第一条数据测试
    # test_query = "铅酸蓄电池正确使用的注意事项有哪些？"
    # results = rag.retrieve(test_query)

    # print(f"检索到 {len(results)} 条结果")
    # for i, r in enumerate(results[:3]):
    #     print(f"\n[{i}] score={r['score']:.4f}")
    #     print(f"    instruction: {r['instruction'][:60]}...")
    #     print(f"    output: {r['output'][:60]}...")


    # from src.models.embedding_model import get_embedding_model
    # model = get_embedding_model()
    # vec = model.get_text_embedding("测试")
    # norm = sum(x**2 for x in vec) ** 0.5
    # print(f"向量模长: {norm:.6f} (应为 1.0)")


    # from src.rag.llama_index_rag import LlamaIndexRAG, RAGConfig
    # # 测试不同阈值
    # for cutoff in [0.99, 0.9, 0.7, 0.5, 0.3, 0.0]:
    #     rag = LlamaIndexRAG(RAGConfig(similarity_cutoff=cutoff, top_k=5))
    #     results = rag.retrieve("铅酸蓄电池正确使用的注意事项有哪些？")
    #     print(f"cutoff={cutoff:4.2f} -> {len(results)} 条结果")