#!/usr/bin/env python3
"""
资源监控守护脚本
- 实时监控：CPU/内存/磁盘/GPU
- 告警阈值：内存>80%、显存>90%、磁盘>85%
- 日志输出到 logs/monitor.log
- 支持后台运行（nohup）
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置日志
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("monitor")


class SystemMonitor:
    """系统资源监控器"""
    
    # 告警阈值
    THRESHOLDS = {
        "cpu_percent": 90,
        "memory_percent": 80,
        "disk_percent": 85,
        "gpu_percent": 90,
        "gpu_growth_gb": 2.0  # 显存增长超过2G告警
    }
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self.alert_history = []
        self.initial_gpu = None
        
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            logger.error("❌ psutil未安装，运行: pip install psutil")
            sys.exit(1)
    
    def get_metrics(self) -> dict:
        """获取当前指标"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": self.psutil.cpu_percent(),
            "memory": self.psutil.virtual_memory()._asdict(),
            "disk": self.psutil.disk_usage('/')._asdict(),
        }
        
        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                
                metrics["gpu"] = {
                    "allocated_gb": round(alloc, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 2),
                    "percent": round(alloc / total * 100, 1)
                }
                
                if self.initial_gpu is None:
                    self.initial_gpu = alloc
                metrics["gpu"]["growth_gb"] = round(alloc - self.initial_gpu, 2)
        except ImportError:
            pass
        
        return metrics
    
    def check_alerts(self, metrics: dict) -> list:
        """检查告警"""
        alerts = []
        
        # CPU
        cpu = metrics.get("cpu_percent", 0)
        if cpu > self.THRESHOLDS["cpu_percent"]:
            alerts.append(f"🔴 CPU过高: {cpu}%")
        
        # 内存
        mem = metrics.get("memory", {})
        mem_pct = mem.get("percent", 0)
        if mem_pct > self.THRESHOLDS["memory_percent"]:
            alerts.append(f"🔴 内存过高: {mem_pct}%")
        
        # 磁盘
        disk = metrics.get("disk", {})
        disk_pct = disk.get("used", 0) / disk.get("total", 1) * 100
        if disk_pct > self.THRESHOLDS["disk_percent"]:
            alerts.append(f"🔴 磁盘过高: {disk_pct:.1f}%")
        
        # GPU
        gpu = metrics.get("gpu", {})
        if gpu:
            if gpu.get("percent", 0) > self.THRESHOLDS["gpu_percent"]:
                alerts.append(f"🔴 GPU显存过高: {gpu['percent']}%")
            if gpu.get("growth_gb", 0) > self.THRESHOLDS["gpu_growth_gb"]:
                alerts.append(f"🔴 GPU显存泄漏: 增长{gpu['growth_gb']:.2f}GB")
        
        return alerts
    
    def run(self, duration: int = None):
        """
        运行监控
        
        Args:
            duration: 运行时长（秒），None=无限
        """
        logger.info(f"{'='*60}")
        logger.info("🔍 系统资源监控启动")
        logger.info(f"  采样间隔: {self.interval}s")
        logger.info(f"  告警阈值: CPU>{self.THRESHOLDS['cpu_percent']}% | "
                   f"内存>{self.THRESHOLDS['memory_percent']}% | "
                   f"磁盘>{self.THRESHOLDS['disk_percent']}% | "
                   f"GPU>{self.THRESHOLDS['gpu_percent']}%")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                metrics = self.get_metrics()
                alerts = self.check_alerts(metrics)
                
                # 格式化输出
                mem = metrics.get("memory", {})
                mem_str = f"{mem.get('used',0)/1024**3:.1f}/{mem.get('total',0)/1024**3:.1f}GB ({mem.get('percent',0)}%)"
                
                disk = metrics.get("disk", {})
                disk_str = f"{disk.get('used',0)/1024**3:.1f}/{disk.get('total',0)/1024**3:.1f}GB"
                
                gpu_str = "N/A"
                gpu = metrics.get("gpu", {})
                if gpu:
                    gpu_str = f"{gpu['allocated_gb']:.1f}/{gpu['total_gb']:.1f}GB ({gpu['percent']}%)"
                
                # 每10次打印一次状态
                if iteration % 10 == 0:
                    logger.info(f"CPU:{metrics['cpu_percent']:5.1f}% | "
                               f"MEM:{mem_str:>20} | "
                               f"DISK:{disk_str:>18} | "
                               f"GPU:{gpu_str}")
                
                # 告警
                for alert in alerts:
                    if alert not in self.alert_history:
                        logger.warning(alert)
                        self.alert_history.append(alert)
                
                # 保存指标
                self._save_metrics(metrics)
                
                iteration += 1
                
                # 检查运行时长
                if duration and (time.time() - start_time) > duration:
                    logger.info("⏹️ 监控时长到达，退出")
                    break
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            logger.info("⏹️ 监控被中断")
    
    def _save_metrics(self, metrics: dict):
        """保存指标到日志文件"""
        metrics_file = LOG_DIR / "metrics.jsonl"
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="系统资源监控")
    parser.add_argument("--interval", type=int, default=5, help="采样间隔(秒)")
    parser.add_argument("--duration", type=int, default=None, help="运行时长(秒)，默认无限")
    args = parser.parse_args()
    
    monitor = SystemMonitor(interval=args.interval)
    monitor.run(duration=args.duration)


if __name__ == "__main__":
    main()