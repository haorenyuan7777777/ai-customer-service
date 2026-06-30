#!/usr/bin/env python3
"""
Agent意图识别评测脚本
- 关键词规则准确率
- BERT模型准确率（如有标注数据）
- 端到端延迟
- 结果追加到 logs/rag_evaluation_report.json
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.intent_classifier import get_intent_classifier
from src.agent.agent_core import get_agent
from src.config import DATA_PATHS, INTENT_LABELS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Agent评测器"""
    
    def __init__(self):
        self.intent_clf = get_intent_classifier()
        self.agent = get_agent()
        self.report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {}
        }
    
    def evaluate_keyword_intent(self) -> Dict[str, Any]:
        """
        关键词意图识别评测
        使用预定义测试用例
        """
        logger.info(f"\n{'='*50}")
        logger.info("🎯 关键词意图识别评测")
        
        test_cases = [
            # (输入, 预期意图)
            ("这个电池多少钱？", "销售转化"),
            ("有优惠活动吗？想批量购买", "销售转化"),
            ("怎么下单？可以开发票吗？", "销售转化"),
            ("电池充不进去电怎么办？", "技术支持"),
            ("产品坏了，怎么修？", "技术支持"),
            ("安装步骤是什么？", "技术支持"),
            ("你们这是骗人的，我要投诉！", "投诉处理"),
            ("产品质量太差，要求退款赔偿", "投诉处理"),
            ("我要差评，虚假宣传", "投诉处理"),
            ("铅酸蓄电池正确使用的注意事项有哪些？", "标准客服"),
            ("你们公司在哪里？", "标准客服"),
            ("谢谢，再见", "标准客服"),
        ]
        
        correct = 0
        details = []
        
        for msg, expected in test_cases:
            result = self.intent_clf.classify(msg)
            actual = result.get("intent", "标准客服")
            is_correct = actual == expected
            
            details.append({
                "input": msg[:30],
                "expected": expected,
                "actual": actual,
                "correct": is_correct,
                "source": result.get("source", "unknown")
            })
            
            correct += int(is_correct)
            icon = "✅" if is_correct else "❌"
            logger.info(f"  {icon} [{result.get('source', '?')}] {msg[:25]}... → {actual}")
        
        accuracy = correct / len(test_cases)
        
        report = {
            "test_type": "keyword_intent",
            "total_cases": len(test_cases),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "details": details,
            "threshold": "≥90%",
            "status": "✅ PASS" if accuracy >= 0.9 else "⚠️ NEED IMPROVE"
        }
        
        logger.info(f"  准确率: {accuracy:.1%} ({correct}/{len(test_cases)}) {report['status']}")
        self.report["tests"]["keyword_intent"] = report
        return report
    
    def evaluate_bert_intent(self, sample_size: int = 200) -> Dict[str, Any]:
        """
        BERT模型意图识别评测
        【演示实现】：使用测试集instruction，无标注时随机验证
        【生产目标】：使用人工标注的测试集
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"🤖 BERT意图识别评测 (sample={sample_size})")
        
        # 加载数据
        test_path = DATA_PATHS.get("test")
        if not test_path or not Path(test_path).exists():
            logger.warning("⚠️ 未找到测试集，跳过BERT评测")
            return {"status": "⏭️ SKIPPED", "reason": "No test data"}
        
        with open(test_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sample = random.sample(data, min(sample_size, len(data)))
        
        # 【演示实现】：无标注标签，仅验证模型能输出且不报错
        # 统计输出分布
        intent_counts = {}
        latencies = []
        errors = 0
        
        for item in sample:
            query = item.get("instruction", "")
            if not query:
                continue
            
            start = time.perf_counter()
            try:
                result = self.intent_clf.classify(query)
                intent = result.get("intent", "标准客服")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                
                # 验证置信度合理
                conf = result.get("confidence", 0)
                if conf < 0.3:
                    logger.warning(f"  低置信度: {query[:30]}... → {intent} ({conf:.2f})")
                    
            except Exception as e:
                errors += 1
                logger.error(f"  预测失败: {e}")
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        if not latencies:
            return {"status": "❌ FAIL", "reason": "All predictions failed"}
        
        report = {
            "test_type": "bert_intent",
            "sample_size": len(latencies),
            "errors": errors,
            "intent_distribution": intent_counts,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies)*0.95)], 2),
            "status": "✅ PASS" if errors == 0 else "⚠️ UNSTABLE"
        }
        
        logger.info(f"  样本: {report['sample_size']} | 错误: {errors}")
        logger.info(f"  意图分布: {intent_counts}")
        logger.info(f"  平均延迟: {report['avg_latency_ms']}ms")
        logger.info(f"  {report['status']}")
        
        self.report["tests"]["bert_intent"] = report
        return report
    
    def evaluate_end_to_end(self, num_queries: int = 50) -> Dict[str, Any]:
        """
        Agent端到端延迟评测
        完整流程：意图→记忆→检索→生成
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"⏱️  Agent端到端延迟评测 (n={num_queries})")
        
        test_queries = [
            "这个电池多少钱？",
            "电池充不进去电怎么办？",
            "我要投诉你们的产品",
            "铅酸蓄电池正确使用的注意事项有哪些？",
            "怎么下单购买？",
        ] * (num_queries // 5 + 1)
        test_queries = test_queries[:num_queries]
        
        latencies = []
        intent_distribution = {}
        
        for idx, query in enumerate(test_queries):
            start = time.perf_counter()
            try:
                result = self.agent.chat(query, user_id=f"eval_user_{idx}", session_id=f"eval_session_{idx}")
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
                
                intent = result.get("intent", "标准客服")
                intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
                
            except Exception as e:
                logger.error(f"  端到端失败: {e}")
        
        if not latencies:
            return {"status": "❌ FAIL", "reason": "All queries failed"}
        
        latencies.sort()
        n = len(latencies)
        
        report = {
            "test_type": "end_to_end",
            "sample_size": n,
            "p50_ms": round(latencies[n // 2], 2),
            "p95_ms": round(latencies[int(n * 0.95)], 2),
            "p99_ms": round(latencies[int(n * 0.99)] if n >= 100 else max(latencies), 2),
            "mean_ms": round(sum(latencies) / n, 2),
            "intent_distribution": intent_distribution,
            "threshold_p95": "<3000ms",
            "status": "✅ PASS" if latencies[int(n * 0.95)] < 3000 else "⚠️ SLOW"
        }
        
        logger.info(f"  P50: {report['p50_ms']}ms | P95: {report['p95_ms']}ms | P99: {report['p99_ms']}ms")
        logger.info(f"  Mean: {report['mean_ms']}ms")
        logger.info(f"  意图分布: {intent_distribution}")
        logger.info(f"  {report['status']}")
        
        self.report["tests"]["end_to_end"] = report
        return report
    
    def save_report(self):
        """追加保存到统一报告文件"""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        report_path = logs_dir / "rag_evaluation_report.json"
        
        # 如果已有报告，合并
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            
            # 合并Agent测试到现有报告
            if "tests" not in existing:
                existing["tests"] = {}
            existing["tests"].update(self.report["tests"])
            existing["agent_summary"] = {
                "timestamp": self.report["timestamp"],
                "tests_count": len(self.report["tests"])
            }
            merged = existing
        else:
            merged = self.report
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n📄 Agent评测报告已追加: {report_path}")
        return report_path
    
    def run_all(self) -> Dict[str, Any]:
        """运行全部Agent评测"""
        logger.info(f"\n{'='*60}")
        logger.info("🤖 Agent评测开始")
        logger.info(f"{'='*60}")
        
        self.evaluate_keyword_intent()
        self.evaluate_bert_intent()
        self.evaluate_end_to_end()
        
        # 总评
        all_pass = all(
            test.get("status", "").startswith("✅") or test.get("status", "").startswith("⏭️")
            for test in self.report["tests"].values()
        )
        self.report["summary"] = {
            "total_tests": len(self.report["tests"]),
            "all_pass": all_pass,
            "overall_status": "✅ PASS" if all_pass else "⚠️ NEED IMPROVE"
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("📋 Agent评测报告")
        logger.info(f"{'='*60}")
        for name, result in self.report["tests"].items():
            if isinstance(result, dict) and "status" in result:
                logger.info(f"  {result['status']} {name}")
        
        logger.info(f"\n总评: {self.report['summary']['overall_status']}")
        
        self.save_report()
        return self.report


def main():
    evaluator = AgentEvaluator()
    evaluator.run_all()


if __name__ == "__main__":
    main()