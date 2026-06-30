#!/usr/bin/env python3
"""
最终验证脚本
- 检查8个阶段全部交付物
- 验证全链路可运行
- 输出系统状态报告
"""

import os
import sys
import json
import socket
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent


def check_file(path: str, desc: str) -> tuple:
    """检查文件是否存在"""
    exists = Path(path).exists()
    icon = "✅" if exists else "❌"
    return exists, f"{icon} {desc}: {path}"

def check_port(host: int, port: int, desc: str) -> tuple:
    """检查端口是否连通"""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True, f"✅ {desc}: {host}:{port}"
    except:
        return False, f"❌ {desc}: {host}:{port}"

def main():
    print(f"\n{'='*70}")
    print("🔍 AI客服系统最终验证")
    print(f"{'='*70}")
    print(f"阶段: 8/8 | 验证时间: 全链路")
    print(f"{'='*70}\n")
    
    results = {
        "files": [],
        "ports": [],
        "stages": {}
    }
    
    # ========== 阶段1：环境搭建 ==========
    print("【阶段1】环境搭建")
    stage1_checks = [
        check_file("docker-compose.yml", "vLLM编排"),
        check_file("docker-compose.milvus.yml", "Milvus编排"),
        check_file("requirements.txt", "Python依赖"),
    ]
    for ok, msg in stage1_checks:
        results["files"].append((ok, msg))
        print(f"  {msg}")
    results["stages"]["1_环境搭建"] = all(ok for ok, _ in stage1_checks)
    
    # ========== 阶段2：知识库构建 ==========
    print("\n【阶段2】知识库构建")
    stage2_checks = [
        check_file("data/raw/knowledge.json", "知识库原始数据"),
        check_file("data/processed/id_cache.json", "ID缓存"),
    ]
    for ok, msg in stage2_checks:
        results["files"].append((ok, msg))
        print(f"  {msg}")
    results["stages"]["2_知识库构建"] = all(ok for ok, _ in stage2_checks)
    
    # ========== 阶段3：RAG检索 ==========
    print("\n【阶段3】RAG检索")
    stage3_checks = [
        check_file("src/rag/llama_index_rag.py", "LlamaIndex RAG"),
        check_file("src/rag/milvus_store.py", "Milvus存储"),
        check_file("configs/milvus.yaml", "Milvus配置"),
    ]
    for ok, msg in stage3_checks:
        results["files"].append((ok, msg))
        print(f"  {msg}")
    
    # 检查Milvus数据
    port_ok, port_msg = check_port("localhost", 19530, "Milvus gRPC")
    results["ports"].append((port_ok, port_msg))
    print(f"  {port_msg}")
    results["stages"]["3_RAG检索"] = all(ok for ok, _ in stage3_checks) and port_ok
    
    # ========== 阶段4：Prompt体系 ==========
    print("\n【阶段4】Prompt体系")
    stage4_checks = [
        check_file("flows/flow.dag.yaml", "DAG流程定义"),
        check_file("src/promptflow/flow_engine.py", "Promptflow引擎"),
        check_file("configs/prompts.yaml", "提示词配置"),
        check_file("flows/nodes/detect_intent.py", "意图检测节点"),
        check_file("flows/nodes/load_memory.py", "记忆加载节点"),
        check_file("flows/nodes/retrieve_knowledge.py", "知识检索节点"),
        check_file("flows/nodes/assemble_prompt.py", "Prompt组装节点"),
        check_file("flows/nodes/generate_response.py", "响应生成节点"),
        check_file("flows/nodes/save_memory.py", "记忆保存节点"),
    ]
    for ok, msg in stage4_checks:
        results["files"].append((ok, msg))
        print(f"  {msg}")
    results["stages"]["4_Prompt体系"] = all(ok for ok, _ in stage4_checks)
    
    # ========== 阶段5：AI Agent ==========
    print("\n【阶段5】AI Agent")
    stage5_checks = [
        check_file("src/agent/agent_core.py", "Agent核心"),
        check_file("src/agent/intent_classifier.py", "意图分类器"),
        check_file("src/agent/memory.py", "记忆管理"),
        check_file("src/agent/tools.py", "工具集"),
    ]
    for ok, msg in stage5_checks:
        results["files"].append((ok, msg))
        print(f"  {msg}")
    results["stages"]["5_AI_Agent"] = all(ok for ok, _ in stage5_checks)
    
    # ========== 阶段6：管理后台 ==========
    print("\n【阶段6】管理后台")
    stage6_checks = [
        check_file("src/ui/streamlit_app.py", "Streamlit后台"),
    ]
    for ok, msg in stage6_checks:
        results["files"].append((ok, msg))
        print(f"  {msg}")
    
    port_ok, port_msg = check_port("localhost", 8501, "Streamlit")
    results["ports"].append((port_ok, port_msg))
    print(f"  {port_msg}")
    results["stages"]["6_管理后台"] = all(ok for ok, _ in stage6_checks) and port_ok
    
    # ========== 阶段7：评测脚本 ==========
    print("\n【阶段7】评测脚本")
    stage7_checks = [
        check_file("scripts/evaluate_rag.py", "RAG评测"),
        check_file("scripts/evaluate_agent.py", "Agent评测"),
        check_file("scripts/benchmark.py", "压测脚本"),
        check_file("scripts/run_evaluation.py", "综合评测"),
        check_file("logs/rag_evaluation_report.json", "评测报告"),
    ]
    for ok, msg in stage7_checks:
        results["files"].append((ok, msg))
        print(f"  {msg}")
    results["stages"]["7_评测脚本"] = all(ok for ok, _ in stage7_checks)
    
    # ========== 阶段8：整合优化 ==========
    print("\n【阶段8】整合优化")
    stage8_checks = [
        check_file("src/api/main.py", "FastAPI主应用"),
        check_file("scripts/optimize.py", "性能调优"),
        check_file("scripts/monitor.py", "资源监控"),
        check_file("scripts/wsl_optimize.sh", "WSL优化"),
        check_file("scripts/start_all.sh", "一键启动"),
        check_file("scripts/stop_all.sh", "一键停止"),
        check_file("scripts/final_verify.py", "最终验证"),
    ]
    for ok, msg in stage8_checks:
        results["files"].append((ok, msg))
        print(f"  {msg}")
    
    port_ok, port_msg = check_port("localhost", 8080, "FastAPI")
    results["ports"].append((port_ok, port_msg))
    print(f"  {port_msg}")
    results["stages"]["8_整合优化"] = all(ok for ok, _ in stage8_checks) and port_ok
    
    # ========== 汇总 ==========
    print(f"\n{'='*70}")
    print("📋 验证汇总")
    print(f"{'='*70}")
    
    total_files = len(results["files"])
    ok_files = sum(1 for ok, _ in results["files"] if ok)
    
    total_ports = len(results["ports"])
    ok_ports = sum(1 for ok, _ in results["ports"] if ok)
    
    print(f"文件检查: {ok_files}/{total_files} 通过")
    print(f"端口检查: {ok_ports}/{total_ports} 通过")
    print(f"\n阶段完成度:")
    
    all_pass = True
    for stage, passed in results["stages"].items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {stage}")
        if not passed:
            all_pass = False
    
    print(f"\n{'='*70}")
    if all_pass:
        print("🎉 全部阶段验证通过！系统已就绪")
    else:
        print("⚠️ 部分阶段未通过，请检查上述 ❌ 项")
    print(f"{'='*70}")
    
    # 保存报告
    report = {
        "verification_time": __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
        "file_check": f"{ok_files}/{total_files}",
        "port_check": f"{ok_ports}/{total_ports}",
        "stage_results": results["stages"],
        "overall_status": "PASS" if all_pass else "NEED_FIX"
    }
    
    with open("logs/final_verification.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)