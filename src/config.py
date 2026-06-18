"""
全局配置文件
所有路径、参数、阈值集中管理
"""
import os
from pathlib import Path

MODELSCOPE_CACHE = Path(os.getenv("MODELSCOPE_CACHE", "/mnt/e/modelscope")).expanduser()

MODEL_PATHS = {
    "bge-small-zh": str(MODELSCOPE_CACHE / "BAAI" / "bge-small-zh-v1.5"),
    "bert-base-chinese": str(MODELSCOPE_CACHE / "google-bert" / "bert-base-chinese"),
}


# 验证路径存在
for name, path in MODEL_PATHS.items():
    if not Path(path).exists():
        print(f"⚠️ 警告: {name} 模型路径不存在: {path}")
        print(f"   请确认已通过ModelScope下载，或设置 MODELSCOPE_CACHE 环境变量")
        # 回退到HuggingFace在线加载（需网络）
        MODEL_PATHS[name] = name.replace("bge-small-zh", "BAAI/bge-small-zh-v1.5").replace("bert-base-chinese", "bert-base-chinese")


# ========== 硬件约束（不可变更） ==========
HARDWARE = {
    "gpu_total_gb": 16,
    "gpu_available_gb": 10,
    "memory_total_gb": 14,
    "context_limit": 2048,
}

# ========== 模型配置 ==========
MODELS = {
    "llm": {
        "name": "Qwen2.5-1.5B",
        "url": "http://localhost:8000/v1",
        "api_key": "not-needed",
        "max_tokens": 1024,
        "temperature": 0.7,
    },
    "embedding": {
        "name": "BAAI/bge-small-zh-v1.5",
        "local_path": MODEL_PATHS["bge-small-zh"],  # ModelScope本地路径
        "dim": 512,
        "device": "cuda",
        "batch_size": 500,
    },
    "intent": {
        "name": "bert-base-chinese",
        "local_path": MODEL_PATHS["bert-base-chinese"],  # ModelScope本地路径
        "num_labels": 5,
        "device": "cuda",
        "max_length": 128,
    }
}

# ========== Milvus配置（不变） ==========
MILVUS = {
    "host": "localhost",
    "port": "19530",
    "collection_name": "knowledge_base",
    "dim": 512,
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "index_params": {"M": 16, "efConstruction": 128},
    "search_params": {"ef": 64},
}

# ========== 意图标签定义（不变） ==========
INTENT_LABELS = {
    0: "general_query",
    1: "price_inquiry",
    2: "purchase_intent",
    3: "technical_issue",
    4: "complaint",
}

# ========== 数据库配置（不变） ==========
DATABASE = {
    "path": str(Path(__file__).parent.parent / "data" / "chat_memory.db"),
    "max_history_rounds": 3,
}

# ========== Promptflow配置（不变） ==========
PROMPTFLOW = {
    "flows_dir": str(Path(__file__).parent.parent / "flows"),
    "templates_dir": str(Path(__file__).parent.parent / "src" / "promptflow" / "templates"),
}

# ========== API配置（不变） ==========
API = {
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 1,
}

# ========== 数据划分（不变） ==========
DATA_SPLIT = {
    "train_ratio": 0.8,
    "test_ratio": 0.2,
    "random_seed": 42,
}

# ========== 数据路径配置 ==========
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATHS = {
    "raw": str(PROJECT_ROOT / "data" / "raw" / "knowledge.json"),
    "train": str(PROJECT_ROOT / "data" / "train_test_split" / "train.json"),
    "test": str(PROJECT_ROOT / "data" / "train_test_split" / "test.json"),
    "processed": str(PROJECT_ROOT / "data" / "processed"),
}