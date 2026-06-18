"""
RAG检索评测脚本
计算：Top-K准确率、召回率、MRR、平均检索时间
"""
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np

from src.rag.llama_index_rag import LlamaIndexRAG


class RAGEvaluator:
    """RAG检索评测器"""
    
    def __init__(self, rag: LlamaIndexRAG = None):
        self.rag = rag or LlamaIndexRAG()
    
    def evaluate_on_testset(
        self,
        test_data_path: str,
        top_k_values: List[int] = [1, 3, 5],
        max_samples: int = 200
    ) -> Dict:
        """
        在测试集上评测RAG检索质量
        
        评测方法：
        - 用测试集的instruction作为查询
        - 检查正确答案是否出现在Top-K结果中
        """
        print("=" * 60)
        print("RAG检索评测")
        print("=" * 60)
        
        # 加载测试集
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 限制样本数加速评测
        test_data = test_data[:max_samples]
        print(f"[评测] 测试集: {len(test_data)}条 (限制{max_samples}条)")
        
        # 评测指标
        hits = {k: 0 for k in top_k_values}
        rr_sum = 0
        retrieval_times = []
        
        for i, item in enumerate(test_data):
            query = item["instruction"]
            ground_truth = item["output"]
            
            # 执行检索
            start = time.time()
            results = self.rag.retrieve(query, intent_filter=None)
            elapsed = time.time() - start
            retrieval_times.append(elapsed)
            
            # 检查Hit@K
            result_outputs = [r["output"] for r in results]
            
            for k in top_k_values:
                if any(self._is_match(ground_truth, r) for r in result_outputs[:k]):
                    hits[k] += 1
            
            # 计算RR
            for rank, r in enumerate(result_outputs, 1):
                if self._is_match(ground_truth, r):
                    rr_sum += 1.0 / rank
                    break
            
            if (i + 1) % 50 == 0:
                print(f"[评测] 已处理 {i+1}/{len(test_data)}条")
        
        # 计算指标
        total = len(test_data)
        metrics = {
            "total_queries": total,
            "avg_retrieval_time_ms": np.mean(retrieval_times) * 1000,
            "p99_retrieval_time_ms": np.percentile(retrieval_times, 99) * 1000,
        }
        
        for k in top_k_values:
            metrics[f"hit@{k}"] = hits[k] / total
        
        metrics["mrr"] = rr_sum / total
        
        self._print_metrics(metrics)
        
        return metrics
    
    def _is_match(self, ground_truth: str, retrieved: str) -> bool:
        """判断检索结果是否匹配正确答案"""
        gt = ground_truth.strip()
        ret = retrieved.strip()
        
        if gt in ret or ret in gt:
            return True
        
        # Jaccard相似度
        gt_set = set(gt)
        ret_set = set(ret)
        if len(gt_set) == 0:
            return False
        jaccard = len(gt_set & ret_set) / len(gt_set | ret_set)
        return jaccard > 0.6
    
    def _print_metrics(self, metrics: Dict):
        """打印评测结果"""
        print(f"\n{'='*60}")
        print("评测结果")
        print(f"{'='*60}")
        print(f"查询总数: {metrics['total_queries']}")
        print(f"平均检索时间: {metrics['avg_retrieval_time_ms']:.1f}ms")
        print(f"P99检索时间: {metrics['p99_retrieval_time_ms']:.1f}ms")
        print()
        print("Hit@K（正确答案出现在Top-K中的比例）:")
        for k in [1, 3, 5]:
            if f"hit@{k}" in metrics:
                print(f"  Hit@{k}: {metrics[f'hit@{k}']:.3f} ({metrics[f'hit@{k}']*100:.1f}%)")
        print(f"\nMRR（平均倒数排名）: {metrics['mrr']:.3f}")
        print(f"{'='*60}")


def run_evaluation():
    """运行评测"""
    evaluator = RAGEvaluator()
    
    metrics = evaluator.evaluate_on_testset(
        test_data_path="data/train_test_split/test.json",
        top_k_values=[1, 3, 5],
        max_samples=200
    )
    
    # 保存结果
    output_path = Path("data/processed/rag_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\n[评测] 结果已保存: {output_path}")


if __name__ == "__main__":
    run_evaluation()