import sys
import torch
import os

# ----------------- 基础环境检查 -----------------
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA不可用，请检查驱动和CUDA安装。")
    sys.exit(1)

print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"显存总量: {total_mem:.1f} GB")

print("\n=== 顺序加载GPU模型 ===")

# 1. vLLM已在Docker中运行（跳过）
print("1. vLLM Qwen1.5B: 已在Docker运行，端口8000")

# ----------------- 加载BGE -----------------
print("\n2. 加载 BGE-small-zh...")
base_path = "/mnt/e/modelscope"
bge_model_path = os.path.join(base_path, "BAAI", "bge-small-zh-v1.5")
if not os.path.isdir(bge_model_path):
    print(f"❌ BGE模型路径不存在: {bge_model_path}")
    sys.exit(1)

from sentence_transformers import SentenceTransformer
try:
    bge = SentenceTransformer(bge_model_path, device='cuda')
except Exception as e:
    print(f"❌ BGE加载失败: {e}")
    sys.exit(1)

bge_dim = bge.get_sentence_embedding_dimension()
print(f"   ✓ BGE加载完成 | 维度: {bge_dim} | 显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# ----------------- 加载BERT -----------------
print("\n3. 加载 bert-base-chinese...")
bert_model_path = os.path.join(base_path, "google-bert", "bert-base-chinese")
if not os.path.isdir(bert_model_path):
    print(f"❌ BERT模型路径不存在: {bert_model_path}")
    sys.exit(1)

from transformers import AutoTokenizer, AutoModel
try:
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    # 若用于特征提取，使用 AutoModel；若需要分类头，可改为 AutoModelForSequenceClassification，但需注意随机初始化
    bert = AutoModel.from_pretrained(
        bert_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True  # 部分模型可能需要
    ).cuda()
except Exception as e:
    print(f"❌ BERT加载失败: {e}")
    sys.exit(1)

print(f"   ✓ BERT加载完成 | 显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# 清理缓存
torch.cuda.empty_cache()

print(f"\n=== 加载完成 ===")
print(f"总显存占用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
print(f"显存预留: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
print(f"剩余显存: {(total_mem - torch.cuda.memory_allocated()/1024**3):.1f}GB")


# ----------------- Milvus 连接验证 -----------------
print("=== 验证 Milvus 连接 ===")

import sys
from pymilvus import connections, utility

try:
    connections.connect(host='localhost', port='19530', timeout=5)
    version = utility.get_server_version()
    print(f"✓ Milvus版本: {version}")
except Exception as e:
    print(f"❌ Milvus连接失败: {e}")
    sys.exit(1)

print("=== 阶段1环境搭建完成 ===")