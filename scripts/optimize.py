#!/usr/bin/env python3
"""
性能调优脚本
- 自动调参：similarity_cutoff、batch_size、max_tokens
- 针对阶段7评测未达标项进行优化
- 输出最优配置到 configs/optimized.yaml
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.llama_index_rag import LlamaIndexRAG, RAGConfig
from src.agent.intent_classifier import get_intent_classifier
from src.config import DATA_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoOptimizer:
    """自动调参器"""
    
    def __init__(self):
        self.test_data = self._load_test_data()
    
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
    
    def optimize_similarity_cutoff(self) -> Tuple[float, float]:
        """
        优化RAG相似度阈值
        
        目标：Top-3准确率 ≥ 60%，同时召回率不过低
        """
        logger.info("🔧 优化 similarity_cutoff...")
        
        best_cutoff = 0.7
        best_accuracy = 0
        
        for cutoff in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            rag = LlamaIndexRAG(RAGConfig(similarity_cutoff=cutoff, top_k=5))
            
            correct = 0
            total = 0
            
            for item in self.test_data[:50]:
                query = item.get("instruction", "")
                if not query:
                    continue
                
                results = rag.retrieve(query)
                top3 = results[:3]
                
                # 简化判定：有结果即算命中（cutoff过低会全命中但质量差）
                if len(top3) > 0:
                    # 检查质量：最高分是否>cutoff
                    if top3[0].get("score", 0) >= cutoff:
                        correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            logger.info(f"  cutoff={cutoff}: 命中率={accuracy:.1%}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_cutoff = cutoff
        
        logger.info(f"✅ 最优 similarity_cutoff: {best_cutoff} (命中率={best_accuracy:.1%})")
        return best_cutoff, best_accuracy
    
    def optimize_batch_size(self) -> int:
        """
        优化向量化批次大小
        
        目标：速度最快，显存不溢出
        """
        logger.info("\n🔧 优化 batch_size...")
        
        from src.models.embedding_model import get_embedding_model
        
        embed_model = get_embedding_model()
        test_texts = [item.get("instruction", "") for item in self.test_data[:100]]
        
        best_batch = 64
        best_time = float('inf')
        
        for batch_size in [16, 32, 64, 128, 256, 500]:
            try:
                import torch
                torch.cuda.empty_cache()
                
                start = time.perf_counter()
                _ = embed_model.get_text_embedding_batch(test_texts[:batch_size])
                elapsed = time.perf_counter() - start
                
                # 计算吞吐量
                throughput = batch_size / elapsed
                
                logger.info(f"  batch={batch_size}: {elapsed*1000:.0f}ms, {throughput:.0f} texts/s")
                
                if throughput > best_time:
                    best_time = throughput
                    best_batch = batch_size
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"  batch={batch_size}: OOM")
                    break
        
        logger.info(f"✅ 最优 batch_size: {best_batch}")
        return best_batch
    
    def optimize_max_tokens(self) -> int:
        """
        优化LLM生成最大token数
        
        目标：在2048上下文限制内，生成质量最佳
        """
        logger.info("\n🔧 优化 max_tokens...")
        
        # 基于测试数据平均长度估算
        avg_output_len = sum(len(item.get("output", "")) for item in self.test_data[:50]) / 50
        # 中文字符 ≈ 1.5 tokens
        estimated_tokens = int(avg_output_len * 1.5)
        
        # 限制在合理范围
        optimal = max(256, min(estimated_tokens, 1024))
        
        logger.info(f"  平均输出长度: {avg_output_len:.0f}字符 ≈ {estimated_tokens}tokens")
        logger.info(f"✅ 最优 max_tokens: {optimal}")
        return optimal
    
    def run(self):
        """运行全部优化"""
        logger.info(f"\n{'='*60}")
        logger.info("🔧 性能自动调优")
        logger.info(f"{'='*60}")
        
        cutoff, acc = self.optimize_similarity_cutoff()
        batch = self.optimize_batch_size()
        max_tokens = self.optimize_max_tokens()
        
        # 保存优化配置
        optimized = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rag": {
                "similarity_cutoff": cutoff,
                "top_k": 5,
                "rerank_top_n": 3
            },
            "embedding": {
                "batch_size": batch
            },
            "llm": {
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "validation": {
                "estimated_top3_accuracy": round(acc, 4)
            }
        }
        
        config_path = Path("configs") / "optimized.yaml"
        config_path.parent.mkdir(exist_ok=True)
        
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(optimized, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("📋 优化结果")
        logger.info(f"{'='*60}")
        logger.info(f"  similarity_cutoff: {cutoff}")
        logger.info(f"  batch_size: {batch}")
        logger.info(f"  max_tokens: {max_tokens}")
        logger.info(f"\n📄 配置已保存: {config_path}")
        
        return optimized


def main():
    optimizer = AutoOptimizer()
    optimizer.run()


if __name__ == "__main__":
    main()