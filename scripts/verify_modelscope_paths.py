#!/usr/bin/env python3
"""
验证ModelScope本地模型路径
"""
import sys
from pathlib import Path
from src.config import MODEL_PATHS, MODELSCOPE_CACHE


def verify_paths():
    print("=" * 60)
    print("ModelScope本地模型路径验证")
    print("=" * 60)
    print(f"ModelScope缓存根目录: {MODELSCOPE_CACHE}")
    print(f"目录存在: {MODELSCOPE_CACHE.exists()}")
    print()
    
    all_ok = True
    for name, path in MODEL_PATHS.items():
        p = Path(path)
        exists = p.exists()
        status = "✅ 存在" if exists else "❌ 不存在"
        
        print(f"模型: {name}")
        print(f"  路径: {path}")
        print(f"  状态: {status}")
        
        if exists:
            # 检查关键文件
            config_file = p / "config.json"
            model_file = list(p.glob("pytorch_model.bin")) or list(p.glob("model.safetensors"))
            
            print(f"  config.json: {'✅' if config_file.exists() else '❌'}")
            print(f"  模型权重: {'✅' if model_file else '❌'}")
            
            if config_file.exists():
                import json
                with open(config_file) as f:
                    cfg = json.load(f)
                print(f"  模型类型: {cfg.get('model_type', 'unknown')}")
        else:
            all_ok = False
            print(f"  ⚠️ 请通过以下命令下载:")
            if "bge" in name:
                print(f"     modelscope download --model BAAI/bge-small-zh-v1.5")
            elif "bert" in name:
                print(f"     modelscope download --model google-bert/bert-base-chinese")
        print()
    
    print("=" * 60)
    if all_ok:
        print("✅ 所有模型路径验证通过，可以开始加载")
        return 0
    else:
        print("❌ 部分模型路径不存在，请先下载模型")
        print(f"\n如需指定自定义路径，请设置环境变量:")
        print(f"  export MODELSCOPE_CACHE=/your/custom/path")
        return 1


if __name__ == "__main__":
    sys.exit(verify_paths())