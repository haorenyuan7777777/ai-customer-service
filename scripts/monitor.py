#!/usr/bin/env python3
"""
资源监控脚本 - 实时监控显存/内存/磁盘
用法: python scripts/monitor.py [--interval 5]
"""
import time
import argparse
import json
from datetime import datetime

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

import psutil


class ResourceMonitor:
    def __init__(self):
        self.gpu_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except:
                pass
    
    def get_gpu_info(self):
        if not self.gpu_available:
            return []
        info = []
        for i in range(self.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            info.append({
                'id': i,
                'name': name.decode() if isinstance(name, bytes) else name,
                'total_mb': mem.total // 1024**2,
                'used_mb': mem.used // 1024**2,
                'free_mb': mem.free // 1024**2,
                'utilization': util.gpu,
                'temperature': pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            })
        return info
    
    def get_memory_info(self):
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            'total_gb': mem.total / 1024**3,
            'available_gb': mem.available / 1024**3,
            'used_gb': mem.used / 1024**3,
            'percent': mem.percent,
            'swap_used_gb': swap.used / 1024**3,
            'swap_total_gb': swap.total / 1024**3
        }
    
    def get_disk_info(self):
        disk = psutil.disk_usage('/')
        return {
            'total_gb': disk.total / 1024**3,
            'used_gb': disk.used / 1024**3,
            'free_gb': disk.free / 1024**3,
            'percent': disk.percent
        }
    
    def check_alerts(self, gpu_info, mem_info, disk_info):
        alerts = []
        # 显存告警 (>14G / 16G = 87.5%)
        for gpu in gpu_info:
            if gpu['used_mb'] > 14336:  # 14GB
                alerts.append(f"🚨 GPU{gpu['id']}显存告警: {gpu['used_mb']/1024:.1f}GB / {gpu['total_mb']/1024:.1f}GB")
        # 内存告警 (>12G / 14G = 85%)
        if mem_info['used_gb'] > 12:
            alerts.append(f"🚨 内存告警: {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB")
        # 磁盘告警 (>35G / 40G = 87.5%)
        if disk_info['free_gb'] < 5:
            alerts.append(f"🚨 磁盘告警: 仅剩 {disk_info['free_gb']:.1f}GB")
        return alerts
    
    def print_status(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gpu_info = self.get_gpu_info()
        mem_info = self.get_memory_info()
        disk_info = self.get_disk_info()
        
        print(f"\n{'='*60}")
        print(f"📊 资源监控 | {now}")
        print(f"{'='*60}")
        
        # GPU
        if gpu_info:
            for gpu in gpu_info:
                used = gpu['used_mb'] / 1024
                total = gpu['total_mb'] / 1024
                bar = '█' * int(used/total*20) + '░' * (20 - int(used/total*20))
                print(f"🎮 GPU{gpu['id']} {gpu['name']}")
                print(f"   显存: [{bar}] {used:.1f}GB / {total:.1f}GB ({used/total*100:.1f}%)")
                print(f"   利用率: {gpu['utilization']}% | 温度: {gpu['temperature']}°C")
        else:
            print("🎮 GPU: 未检测到")
        
        # 内存
        mem_bar = '█' * int(mem_info['percent']/5) + '░' * (20 - int(mem_info['percent']/5))
        print(f"💾 内存: [{mem_bar}] {mem_info['used_gb']:.1f}GB / {mem_info['total_gb']:.1f}GB ({mem_info['percent']}%)")
        print(f"   可用: {mem_info['available_gb']:.1f}GB | Swap: {mem_info['swap_used_gb']:.1f}GB")
        
        # 磁盘
        disk_bar = '█' * int(disk_info['percent']/5) + '░' * (20 - int(disk_info['percent']/5))
        print(f"💿 磁盘: [{disk_bar}] {disk_info['used_gb']:.1f}GB / {disk_info['total_gb']:.1f}GB ({disk_info['percent']}%)")
        print(f"   可用: {disk_info['free_gb']:.1f}GB")
        
        # 告警
        alerts = self.check_alerts(gpu_info, mem_info, disk_info)
        if alerts:
            print(f"\n⚠️ 告警:")
            for alert in alerts:
                print(f"   {alert}")
        else:
            print(f"\n✅ 所有资源正常")
        
        # JSON日志
        log = {
            'timestamp': now,
            'gpu': gpu_info,
            'memory': mem_info,
            'disk': disk_info,
            'alerts': alerts
        }
        with open('logs/resource_monitor.jsonl', 'a') as f:
            f.write(json.dumps(log) + '\n')
        
        return len(alerts) == 0


def main():
    parser = argparse.ArgumentParser(description='资源监控脚本')
    parser.add_argument('--interval', type=int, default=5, help='监控间隔(秒)')
    parser.add_argument('--once', action='store_true', help='只运行一次')
    args = parser.parse_args()
    
    import os
    os.makedirs('logs', exist_ok=True)
    
    monitor = ResourceMonitor()
    
    if args.once:
        monitor.print_status()
    else:
        print(f"开始监控，间隔 {args.interval} 秒 (Ctrl+C停止)")
        try:
            while True:
                monitor.print_status()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n监控已停止")


if __name__ == '__main__':
    main()