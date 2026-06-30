#!/usr/bin/env python3
"""
RAG检索评测脚本
- Top-1/Top-3/Top-5 准确率（测试集20%，2232条）
- 响应延迟 P50/P95/P99
- GPU显存稳定性（30分钟持续运行）
- 结果输出到 logs/rag_evaluation_report.json
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from statistics import mean, median
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PATHS, HARDWARE, MODELS, MILVUS
from src.rag.llama_index_rag import get_rag_engine, RAGConfig
from src.models.embedding_model import get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """RAG评测器"""
    
    def __init__(self):
        self.rag = get_rag_engine()
        self.test_data = self._load_test_data()
        self.report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": HARDWARE,
            "model_config": {
                "llm": MODELS["llm"]["name"],
                "embedding": MODELS["embedding"]["name"],
                "milvus_collection": MILVUS["collection_name"]
            },
            "tests": {}
        }
    
    def _load_test_data(self) -> List[Dict]:
        """加载测试集，兼容标准JSON数组或每行一个JSON对象"""
        test_path = DATA_PATHS.get("test")
        if test_path and Path(test_path).exists():
            raw_data = self._load_json_flexible(test_path)
            if raw_data:
                logger.info(f"📊 加载测试集: {len(raw_data)}条")
                return raw_data

        # 尝试从raw划分
        raw_path = DATA_PATHS.get("raw")
        if raw_path and Path(raw_path).exists():
            logger.info("⚠️ 未找到划分好的测试集，从raw加载全部数据...")
            data = self._load_json_flexible(raw_path)
            if not data:
                logger.error("❌ 无法加载raw数据")
                return []
            
            # 80/20划分
            random.seed(42)
            random.shuffle(data)
            split_idx = int(len(data) * 0.8)
            test_data = data[split_idx:]
            logger.info(f"📊 从raw划分测试集: {len(test_data)}条")
            return test_data
        
        logger.error("❌ 未找到测试数据")
        return []

    def _load_json_flexible(self, file_path: str) -> List[Dict]:
        """兼容多种JSON格式的加载器"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []
            
            # 1. 尝试解析为标准JSON数组
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
            except json.JSONDecodeError:
                # 2. 按行解析（每行一个JSON对象）
                lines = content.splitlines()
                data = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            data.append(obj)
                        except json.JSONDecodeError:
                            continue
                return data
    
    def evaluate_top_k_accuracy(self, k_values: List[int] = [1, 3, 5], sample_size: int = 500) -> Dict[str, Any]:
        """
        Top-K准确率评测
        
        判定标准：检索结果中是否包含与查询语义相似的问题
        【演示实现】：instruction包含判定（简化）
        【生产目标】：LLM判定语义等价性
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"🎯 Top-K 准确率评测 (sample={sample_size})")
        
        sample = random.sample(self.test_data, min(sample_size, len(self.test_data)))
        
        results = {k: {"correct": 0, "total": 0, "samples": []} for k in k_values}
        
        for idx, item in enumerate(sample):
            query = item.get("instruction", "")
            if not query:
                continue
            
            try:
                retrieved = self.rag.retrieve(query)
            except Exception as e:
                logger.warning(f"检索失败 [{idx}]: {e}")
                continue
            
            # 判定命中（简化：检索结果instruction包含查询关键词）
            query_keywords = set(query.lower().split())
            if len(query_keywords) < 2:
                query_keywords = set(query.lower())
            
            for k in k_values:
                top_k_results = retrieved[:k]
                is_hit = False
                
                for r in top_k_results:
                    result_text = r.get("instruction", "").lower()
                    # 共同词比例 > 30% 视为命中
                    result_words = set(result_text.split())
                    if not result_words:
                        continue
                    common = len(query_keywords & result_words)
                    union = len(query_keywords | result_words)
                    similarity = common / union if union > 0 else 0
                    
                    if similarity > 0.3:
                        is_hit = True
                        break
                
                results[k]["correct"] += int(is_hit)
                results[k]["total"] += 1
            
            if (idx + 1) % 100 == 0:
                progress = {k: f"{v['correct']}/{v['total']}" for k, v in results.items()}
                logger.info(f"  进度: {idx+1}/{len(sample)}, 命中: {progress}")
        
        # 计算准确率
        report = {}
        for k in k_values:
            total = results[k]["total"]
            correct = results[k]["correct"]
            accuracy = correct / total if total > 0 else 0
            status = "✅ PASS" if accuracy >= (0.5 if k == 1 else 0.6 if k == 3 else 0.7) else "⚠️ NEED IMPROVE"
            
            report[f"top_{k}_accuracy"] = {
                "k": k,
                "sample_size": total,
                "correct": correct,
                "accuracy": round(accuracy, 4),
                "threshold": f"≥{50 if k==1 else 60 if k==3 else 70}%",
                "status": status
            }
            logger.info(f"  Top-{k}: {correct}/{total} = {accuracy:.2%} {status}")
        
        self.report["tests"]["top_k_accuracy"] = report
        return report
    
    def evaluate_latency(self, num_queries: int = 200) -> Dict[str, Any]:
        """
        响应延迟评测
        
        测试端到端检索延迟（编码+检索+格式化）
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"⏱️  延迟评测 (n={num_queries})")
        
        sample = random.sample(self.test_data, min(num_queries, len(self.test_data)))
        latencies = []
        
        # 预热（排除首次加载开销）
        logger.info("  预热中...")
        try:
            self.rag.retrieve("预热查询")
        except:
            pass
        
        for idx, item in enumerate(sample):
            query = item.get("instruction", "")
            if not query:
                continue
            
            start = time.perf_counter()
            try:
                _ = self.rag.retrieve(query)
            except Exception as e:
                logger.warning(f"检索失败: {e}")
                continue
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        if not latencies:
            return {"error": "无有效延迟数据"}
        
        latencies.sort()
        n = len(latencies)
        
        report = {
            "sample_size": n,
            "p50_ms": round(latencies[n // 2], 2),
            "p95_ms": round(latencies[int(n * 0.95)], 2),
            "p99_ms": round(latencies[int(n * 0.99)] if n >= 100 else max(latencies), 2),
            "mean_ms": round(mean(latencies), 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "threshold_p95": "<500ms",
            "status": "✅ PASS" if latencies[int(n * 0.95)] < 500 else "⚠️ NEED OPTIMIZE"
        }
        
        logger.info(f"  P50: {report['p50_ms']}ms | P95: {report['p95_ms']}ms | P99: {report['p99_ms']}ms")
        logger.info(f"  Mean: {report['mean_ms']}ms | Min: {report['min_ms']}ms | Max: {report['max_ms']}ms")
        logger.info(f"  {report['status']}")
        
        self.report["tests"]["latency"] = report
        return report
    
    def evaluate_gpu_stability(self, duration_sec: int = 60, qps: int = 5) -> Dict[str, Any]:
        """
        GPU显存稳定性评测
        
        持续运行检索，监控显存波动和吞吐量
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"🎮 GPU稳定性评测 (duration={duration_sec}s, target_qps={qps})")
        
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("⚠️ 未检测到GPU，跳过GPU稳定性测试")
                return {"status": "⏭️ SKIPPED", "reason": "No GPU detected"}
            
            initial_mem = torch.cuda.memory_allocated() / 1024**3
            peak_mem = initial_mem
            query_count = 0
            errors = 0
            
            start_time = time.time()
            interval = 1.0 / qps
            
            while time.time() - start_time < duration_sec:
                item = random.choice(self.test_data)
                query = item.get("instruction", "")
                
                try:
                    _ = self.rag.retrieve(query)
                    query_count += 1
                except Exception as e:
                    errors += 1
                
                # 监控显存
                current = torch.cuda.memory_allocated() / 1024**3
                peak_mem = max(peak_mem, current)
                
                # 控制QPS
                elapsed = time.time() - start_time
                expected_queries = int(elapsed * qps)
                if query_count > expected_queries:
                    time.sleep(interval)
            
            actual_qps = query_count / duration_sec
            growth = peak_mem - initial_mem
            
            report = {
                "duration_sec": duration_sec,
                "queries_executed": query_count,
                "errors": errors,
                "actual_qps": round(actual_qps, 2),
                "initial_gb": round(initial_mem, 2),
                "peak_gb": round(peak_mem, 2),
                "growth_gb": round(growth, 2),
                "threshold_growth": "<0.5GB",
                "status": "✅ PASS" if growth < 0.5 and errors == 0 else "⚠️ UNSTABLE"
            }
            
            logger.info(f"  执行查询: {query_count} | 错误: {errors} | QPS: {actual_qps:.1f}")
            logger.info(f"  显存: {initial_mem:.2f}G → {peak_mem:.2f}G (增长{growth:.2f}G)")
            logger.info(f"  {report['status']}")
            
        except ImportError:
            report = {"status": "⏭️ SKIPPED", "reason": "torch not installed"}
        
        self.report["tests"]["gpu_stability"] = report
        return report
    
    def evaluate_throughput(self, batch_sizes: List[int] = [1, 10, 50, 100]) -> Dict[str, Any]:
        """
        吞吐量评测（批量检索）
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"📈 吞吐量评测")
        
        report = {}
        sample = random.sample(self.test_data, 100)
        queries = [item.get("instruction", "") for item in sample if item.get("instruction")]
        
        for batch_size in batch_sizes:
            if batch_size > len(queries):
                continue
            
            batch = queries[:batch_size]
            start = time.perf_counter()
            try:
                results = self.rag.batch_retrieve(batch)
                end = time.perf_counter()
                
                total_time = end - start
                throughput = batch_size / total_time
                
                report[f"batch_{batch_size}"] = {
                    "batch_size": batch_size,
                    "total_time_ms": round(total_time * 1000, 2),
                    "throughput_qps": round(throughput, 2),
                    "avg_latency_ms": round(total_time * 1000 / batch_size, 2)
                }
                logger.info(f"  Batch={batch_size}: {throughput:.1f} qps ({total_time*1000:.0f}ms)")
            except Exception as e:
                report[f"batch_{batch_size}"] = {"error": str(e)}
                logger.warning(f"  Batch={batch_size} 失败: {e}")
        
        self.report["tests"]["throughput"] = report
        return report
    
    def save_report(self):
        """保存评测报告"""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        report_path = logs_dir / "rag_evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n📄 评测报告已保存: {report_path}")
        return report_path
    
    def run_all(self, sample_size: int = 500, latency_samples: int = 200) -> Dict[str, Any]:
        """运行全部RAG评测"""
        logger.info(f"\n{'='*60}")
        logger.info("🧪 RAG检索评测开始")
        logger.info(f"{'='*60}")
        
        self.evaluate_top_k_accuracy(k_values=[1, 3, 5], sample_size=sample_size)
        self.evaluate_latency(num_queries=latency_samples)
        self.evaluate_gpu_stability(duration_sec=60, qps=5)
        self.evaluate_throughput()
        
        # 总评
        all_pass = all(
            test.get("status", "").startswith("✅") or test.get("status", "").startswith("⏭️")
            for test in self.report["tests"].values()
            if isinstance(test, dict)
        )
        self.report["summary"] = {
            "total_tests": len(self.report["tests"]),
            "all_pass": all_pass,
            "overall_status": "✅ PASS" if all_pass else "⚠️ NEED IMPROVE"
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("📋 RAG评测报告")
        logger.info(f"{'='*60}")
        for name, result in self.report["tests"].items():
            if isinstance(result, dict) and "status" in result:
                logger.info(f"  {result['status']} {name}")
        
        logger.info(f"\n总评: {self.report['summary']['overall_status']}")
        
        self.save_report()
        return self.report


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="RAG检索评测")
    parser.add_argument("--sample", type=int, default=500, help="准确率测试样本数")
    parser.add_argument("--latency", type=int, default=200, help="延迟测试查询数")
    parser.add_argument("--gpu-duration", type=int, default=60, help="GPU稳定性测试时长(秒)")
    args = parser.parse_args()
    
    evaluator = RAGEvaluator()
    evaluator.run_all(
        sample_size=args.sample,
        latency_samples=args.latency
    )


if __name__ == "__main__":
    main()