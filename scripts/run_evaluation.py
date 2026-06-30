#!/usr/bin/env python3
"""
综合评测入口
- 一键运行全部评测（RAG + Agent + Benchmark）
- 生成统一报告到 logs/rag_evaluation_report.json
- 供Streamlit管理后台展示
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_all_evaluations():
    """运行全部评测"""
    
    logger.info(f"\n{'='*70}")
    logger.info("🧪 AI客服系统综合评测")
    logger.info(f"{'='*70}")
    logger.info("阶段: 7/8 | 模块: 评测脚本")
    logger.info(f"{'='*70}\n")
    
    # 1. RAG评测
    logger.info("【1/3】RAG检索评测...")
    try:
        from scripts.evaluate_rag import RAGEvaluator
        rag_eval = RAGEvaluator()
        rag_report = rag_eval.run_all(sample_size=500, latency_samples=200)
    except Exception as e:
        logger.error(f"RAG评测失败: {e}")
        rag_report = {"error": str(e)}
    
    # 2. Agent评测
    logger.info("\n【2/3】Agent意图识别评测...")
    try:
        from scripts.evaluate_agent import AgentEvaluator
        agent_eval = AgentEvaluator()
        agent_report = agent_eval.run_all()
    except Exception as e:
        logger.error(f"Agent评测失败: {e}")
        agent_report = {"error": str(e)}
    
    # 3. 压测
    logger.info("\n【3/3】并发压测...")
    try:
        from scripts.benchmark import Benchmark
        benchmark = Benchmark()
        bench_report = benchmark.run(concurrent_users=3, total_requests=30)
    except Exception as e:
        logger.error(f"压测失败: {e}")
        bench_report = {"error": str(e)}
    
    # 合并报告
    unified_report = {
        "report_type": "comprehensive_evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "stage": "7/8",
            "modules_evaluated": ["RAG", "Agent", "Benchmark"]
        },
        "rag_evaluation": rag_report.get("tests", {}),
        "agent_evaluation": agent_report.get("tests", {}),
        "benchmark": bench_report,
        "summary": {
            "rag_status": rag_report.get("summary", {}).get("overall_status", "UNKNOWN"),
            "agent_status": agent_report.get("summary", {}).get("overall_status", "UNKNOWN"),
            "benchmark_status": bench_report.get("status", "UNKNOWN")
        }
    }
    
    # 判定总评
    all_pass = all(
        "✅" in str(v) or "⏭️" in str(v)
        for v in unified_report["summary"].values()
    )
    unified_report["summary"]["overall_status"] = "✅ ALL PASS" if all_pass else "⚠️ NEED IMPROVE"
    
    # 保存
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    report_path = logs_dir / "rag_evaluation_report.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(unified_report, f, ensure_ascii=False, indent=2)
    
    # 输出摘要
    logger.info(f"\n{'='*70}")
    logger.info("📋 综合评测报告摘要")
    logger.info(f"{'='*70}")
    logger.info(f"  RAG评测:      {unified_report['summary']['rag_status']}")
    logger.info(f"  Agent评测:    {unified_report['summary']['agent_status']}")
    logger.info(f"  压测:         {unified_report['summary']['benchmark_status']}")
    logger.info(f"  总评:         {unified_report['summary']['overall_status']}")
    logger.info(f"\n📄 完整报告: {report_path}")
    logger.info(f"{'='*70}")
    
    return unified_report


if __name__ == "__main__":
    run_all_evaluations()