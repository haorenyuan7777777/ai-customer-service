#!/usr/bin/env python3
"""
FastAPI启动诊断脚本
- 检查依赖
- 测试导入
- 尝试启动并捕获错误
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("🔍 FastAPI启动诊断")
print("=" * 50)

# 1. 检查依赖
print("\n【1/4】检查Python依赖...")
deps = ["fastapi", "uvicorn", "pydantic", "torch", "pymilvus", "llama_index", "streamlit"]
missing = []
for dep in deps:
    try:
        __import__(dep.replace("-", "_").split(".")[0])
        print(f"  ✅ {dep}")
    except ImportError:
        print(f"  ❌ {dep} (未安装)")
        missing.append(dep)

if missing:
    print(f"\n⚠️ 缺少依赖: {', '.join(missing)}")
    print("   运行: pip install -r requirements.txt")

# 2. 检查项目导入
print("\n【2/4】检查项目模块导入...")
modules = [
    "src.config",
    "src.models.embedding_model",
    "src.models.intent_model",
    "src.models.llm_client",
    "src.rag.milvus_store",
    "src.rag.llama_index_rag",
    "src.agent.memory",
    "src.agent.tools",
    "src.agent.intent_classifier",
    "src.agent.agent_core",
    "src.promptflow.flow_engine",
]

failed = []
for mod in modules:
    try:
        __import__(mod)
        print(f"  ✅ {mod}")
    except Exception as e:
        print(f"  ❌ {mod}: {str(e)[:60]}")
        failed.append((mod, str(e)))

if failed:
    print(f"\n⚠️ {len(failed)}个模块导入失败，请检查代码")

# 3. 检查端口
print("\n【3/4】检查端口占用...")
import socket

ports = [8080, 8501, 8000, 19530]
for port in ports:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("localhost", port))
        print(f"  ✅ 端口 {port} 空闲")
    except socket.error:
        print(f"  ⚠️ 端口 {port} 已被占用")
    finally:
        s.close()

# 4. 尝试启动FastAPI（仅测试导入）
print("\n【4/4】测试FastAPI应用创建...")
try:
    from src.api.main import app
    print(f"  ✅ FastAPI应用创建成功")
    print(f"  📋 路由: {[r.path for r in app.routes if hasattr(r, 'path')]}")
except Exception as e:
    print(f"  ❌ FastAPI应用创建失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("诊断完成")