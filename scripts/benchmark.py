#!/usr/bin/env python3
"""
并发压测脚本
- 模拟多用户并发请求
- 测试系统吞吐量、响应延迟、错误率
- 监控GPU/内存峰值
"""

import os
import sys
import json
import time
import random
import threading
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.agent_core import get_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Benchmark:
    """压测器"""
    
    def __init__(self):
        self.agent = get_agent()
        self.results: List[Dict] = []
        self.lock = threading.Lock()
        
        self.test_queries = [
            "这个电池多少钱？",
            "电池充不进去电怎么办？",
            "我要投诉你们的产品",
            "铅酸蓄电池正确使用的注意事项有哪些？",
            "怎么下单购买？",
            "有优惠活动吗？",
            "产品坏了怎么修？",
            "你们公司在哪里？",
            "谢谢，再见",
            "退款流程是什么？",
        ]
    
    def _single_request(self, user_id: str, query: str) -> Dict:
        """单次请求"""
        start = time.perf_counter()
        
        try:
            result = self.agent.chat(
                query,
                user_id=user_id,
                session_id=f"bench_{threading.current_thread().name}"
            )
            success = True
            error = None
            response = result.get("response", "")[:50]
        except Exception as e:
            success = False
            error = str(e)
            response = ""
        
        latency = (time.perf_counter() - start) * 1000
        
        return {
            "success": success,
            "latency_ms": latency,
            "error": error,
            "response_preview": response
        }
    
    def run(
        self,
        concurrent_users: int = 3,
        total_requests: int = 30,
        ramp_up_sec: float = 5.0
    ) -> Dict[str, Any]:
        """
        运行压测
        
        Args:
            concurrent_users: 并发用户数（建议≤3，显存限制）
            total_requests: 总请求数
            ramp_up_sec: 预热时间
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔥 并发压测开始")
        logger.info(f"   并发用户: {concurrent_users} | 总请求: {total_requests}")
        logger.info(f"{'='*60}")
        
        # 预热
        logger.info("预热中...")
        for _ in range(3):
            self._single_request("warmup", "预热查询")
        
        # 记录初始资源
        try:
            import psutil
            import torch
            initial_cpu = psutil.cpu_percent()
            initial_mem = psutil.virtual_memory().percent
            initial_gpu = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        except:
            initial_cpu = initial_mem = initial_gpu = 0
        
        # 执行压测
        start_time = time.time()
        self.results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            for i in range(total_requests):
                query = random.choice(self.test_queries)
                user_id = f"user_{i % concurrent_users}"
                future = executor.submit(self._single_request, user_id, query)
                futures.append(future)
                
                # 渐进加载
                if ramp_up_sec > 0 and i < concurrent_users:
                    time.sleep(ramp_up_sec / concurrent_users)
            
            for future in as_completed(futures):
                result = future.result()
                with self.lock:
                    self.results.append(result)
                
                completed = len(self.results)
                if completed % 10 == 0:
                    logger.info(f"  进度: {completed}/{total_requests}")
        
        total_time = time.time() - start_time
        
        # 分析结果
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        latencies = [r["latency_ms"] for r in successful]
        
        if not latencies:
            return {"status": "❌ FAIL", "reason": "All requests failed"}
        
        latencies.sort()
        n = len(latencies)
        
        # 峰值资源
        try:
            peak_cpu = psutil.cpu_percent()
            peak_mem = psutil.virtual_memory().percent
            peak_gpu = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        except:
            peak_cpu = peak_mem = peak_gpu = 0
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "concurrent_users": concurrent_users,
                "total_requests": total_requests,
                "ramp_up_sec": ramp_up_sec
            },
            "results": {
                "total_time_sec": round(total_time, 2),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": round(len(successful) / total_requests, 4),
                "throughput_rps": round(total_requests / total_time, 2),
                "latency": {
                    "min_ms": round(min(latencies), 2),
                    "p50_ms": round(latencies[n // 2], 2),
                    "p95_ms": round(latencies[int(n * 0.95)], 2),
                    "p99_ms": round(latencies[int(n * 0.99)] if n >= 100 else max(latencies), 2),
                    "max_ms": round(max(latencies), 2),
                    "mean_ms": round(mean(latencies), 2),
                }
            },
            "resource_peak": {
                "cpu_percent": peak_cpu,
                "memory_percent": peak_mem,
                "gpu_gb": round(peak_gpu, 2)
            },
            "errors": [r["error"] for r in failed[:5]] if failed else []
        }
        
        # 判定
        report["status"] = "✅ PASS" if (
            report["results"]["success_rate"] >= 0.95 and
            report["results"]["latency"]["p95_ms"] < 5000
        ) else "⚠️ NEED OPTIMIZE"
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 压测报告")
        logger.info(f"{'='*60}")
        logger.info(f"  成功率: {report['results']['success_rate']:.1%}")
        logger.info(f"  吞吐量: {report['results']['throughput_rps']:.1f} rps")
        logger.info(f"  P95延迟: {report['results']['latency']['p95_ms']}ms")
        logger.info(f"  峰值GPU: {report['resource_peak']['gpu_gb']}GB")
        logger.info(f"  {report['status']}")
        
        # 保存
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        bench_path = logs_dir / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(bench_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n📄 压测报告已保存: {bench_path}")
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="并发压测")
    parser.add_argument("--users", type=int, default=3, help="并发用户数")
    parser.add_argument("--requests", type=int, default=30, help="总请求数")
    parser.add_argument("--ramp-up", type=float, default=5.0, help="预热时间(秒)")
    args = parser.parse_args()
    
    benchmark = Benchmark()
    benchmark.run(
        concurrent_users=args.users,
        total_requests=args.requests,
        ramp_up_sec=args.ramp_up
    )


if __name__ == "__main__":
    main()