# AI 智能客服系统

> 基于 **RAG + LLM + Promptflow** 的企业级智能客服对话系统，集成意图识别、多轮记忆、向量检索与知识库管理。

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [核心模块详解](#核心模块详解)
- [测试与评估](#测试与评估)
- [开发路线图](#开发路线图)
- [许可证](#许可证)

---

## 项目概述

AI 智能客服系统是一个基于 Python 构建的客服对话系统，采用 **RAG（检索增强生成）** 架构，结合本地部署的 LLM（Qwen2.5-1.5B）和向量数据库 Milvus，实现企业级问答型客服机器人。

系统支持：
- **多场景意图识别**（销售转化、技术支持、投诉处理、标准客服）
- **基于 BGE 的语义知识检索**
- **SQLite 多轮对话记忆**
- **Promptflow 工作流引擎**
- **Streamlit 管理后台** + **FastAPI 服务接口**

适合用于搭建企业/产品级问答型客服机器人，或作为开发 RAG + 对话系统的参考起点。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 🎯 意图识别 | 关键词规则 + BERT-base-chinese 模型双级识别，覆盖 4 大类业务意图 |
| 🔍 RAG 检索 | BGE-small-zh 向量编码 + Milvus HNSW 索引，支持相似度阈值过滤与重排序 |
| 🧠 多轮记忆 | SQLite 持久化，支持用户级/会话级隔离，限制 3 轮（受 2048 token 上下文约束） |
| ⚡ 流式响应 | 支持 SSE 流式输出（FastAPI） |
| 🔧 工具调用 | 内置订单查询、库存查询、工单创建、转人工、发送邮件等工具 |
| 📊 管理后台 | Streamlit 仪表盘：知识库 CRUD、对话记录、系统监控、评测报告 |
| 🐳 容器化部署 | Docker Compose 一键启动 vLLM + Milvus + MinIO + Etcd + Attu |
| 🔄 增量更新 | 支持知识库全量重建与增量追加 |
| 📈 评测体系 | RAG 评估、Agent 评估、Benchmark 测试 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户层 (User Layer)                        │
├──────────────────────────┬──────────────────────────────────────┤
│   Streamlit 管理后台      │         FastAPI 服务接口              │
│   (端口 8501)             │         (端口 8080)                   │
│  · 仪表盘                 │   /chat         → 单轮对话            │
│  · 知识库管理             │   /chat/stream  → SSE 流式对话        │
│  · 对话记录               │   /health       → 健康检查            │
│  · 系统监控               │   /metrics      → 系统指标            │
│  · 评测报告               │   /api/v1/*     → 子路由 (chat/admin) │
└──────────────────────────┴──────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent 核心层 (Agent Core)                    │
├─────────────────────────────────────────────────────────────────┤
│  CustomerServiceAgent                                           │
│  ├── 意图识别 (IntentClassifier)                                │
│  │    ├── 关键词规则 (快速路径)                                  │
│  │    └── BERT-base-chinese (GPU 模型)                          │
│  ├── 记忆管理 (MemoryStore)                                     │
│  │    └── SQLite 多轮对话存储                                    │
│  ├── RAG 检索 (LlamaIndexRAG)                                   │
│  │    ├── BGE-small-zh 向量化                                  │
│  │    └── Milvus HNSW 向量检索                                  │
│  ├── LLM 生成 (LLMClient)                                       │
│  │    └── vLLM OpenAI-兼容 API (Qwen2.5-1.5B)                   │
│  └── 工具调用 (ToolRegistry)                                    │
│       └── 订单/库存/工单/转人工/邮件                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Promptflow 引擎层 (Flow Engine)                  │
├─────────────────────────────────────────────────────────────────┤
│  flow.dag.yaml                                                  │
│  ├── detect_intent → load_memory → retrieve_knowledge           │
│  ├── assemble_prompt → generate_response → save_memory           │
│  └── 支持拓扑排序、节点缓存、错误回退、Jinja2 模板               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    基础设施层 (Infrastructure)                     │
├─────────────────────────────────────────────────────────────────┤
│  vLLM (端口 8000)    │  Milvus (端口 19530)   │  SQLite          │
│  Qwen2.5-1.5B        │  HNSW 向量索引          │  chat_memory.db  │
│  GPU 推理 (FP8)      │  Etcd + MinIO + Attu   │  多轮对话历史    │
└─────────────────────────────────────────────────────────────────┘
```

### 对话处理流程

```
用户输入
    │
    ▼
┌───────────────┐     ┌───────────────┐
│ 意图识别      │────→│ 多轮记忆加载  │
│ (关键词/BERT) │     │ (SQLite)      │
└───────────────┘     └───────────────┘
    │                          │
    ▼                          ▼
┌───────────────┐     ┌───────────────┐
│ 知识检索      │     │ 组装 Prompt   │
│ (BGE+Milvus)  │────→│ (Jinja2模板)  │
└───────────────┘     └───────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │ LLM 生成回复  │
                    │ (vLLM Qwen)   │
                    └───────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │ 保存对话记忆  │
                    │ 返回用户      │
                    └───────────────┘
```

---

## 技术栈

### 核心框架

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 开发语言 |
| FastAPI | 0.115.6 | API 后端框架 |
| Uvicorn | 0.34.0 | ASGI 服务器 |
| Streamlit | 1.42.0 | 管理后台 UI |

### AI / ML

| 组件 | 版本 | 用途 |
|------|------|------|
| PyTorch | 2.12.1+cu132 | 深度学习框架 |
| Transformers | 4.51.3 | 预训练模型加载 |
| Sentence-Transformers | 4.0.2 | BGE Embedding 编码 |
| scikit-learn | 1.6.1 | 机器学习工具 |
| vLLM | latest | LLM 推理服务 (Docker) |

### RAG & 向量检索

| 组件 | 版本 | 用途 |
|------|------|------|
| LlamaIndex | 0.12.15 | RAG 框架 |
| PyMilvus | 2.5.6 | Milvus Python SDK |
| Milvus | v2.4.1 | 向量数据库 (Docker) |

### 数据与存储

| 组件 | 版本 | 用途 |
|------|------|------|
| SQLAlchemy | 2.0.38 | ORM (兼容层) |
| sqlite-vec | 0.1.0 | 备用向量存储 (降级方案) |
| Pandas | 2.2.2 | 数据处理 |
| OpenPyXL | 3.1.5 | Excel 导出 |

### 工作流与模板

| 组件 | 版本 | 用途 |
|------|------|------|
| Promptflow | 1.18.5 | 工作流引擎 |
| Jinja2 | 3.1.4 | 提示模板渲染 |
| PyYAML | 6.0.2 | YAML 配置解析 |

### 模型清单

| 模型 | 用途 | 大小 | 设备 |
|------|------|------|------|
| Qwen/Qwen2.5-1.5B-Instruct | 大语言模型 (LLM) | ~1.5B | GPU (vLLM) |
| BAAI/bge-small-zh-v1.5 | 文本向量化 (Embedding) | ~512维 | GPU |
| google-bert/bert-base-chinese | 意图分类 | ~110M | GPU |

---

## 项目结构

```
ai-customer-service/
│
├── 📄 .env                              # 环境变量配置（敏感信息）
├── 📁 .vscode/                          # VSCode 编辑器配置
│   └── 📄 settings.json
│
├── 📄 README.md                         # 项目说明文档
├── 📄 requirements.txt                  # Python 依赖清单
├── 📄 requirements_backup.txt           # 依赖备份
├── 📄 docker-compose.yml                # Docker Compose 主配置（vLLM + Milvus全家桶）
├── 📄 docker-compose.milvus.yml         # Milvus 专用 Docker 配置
│
├── 📁 configs/                          # 配置中心
│   ├── 📄 milvus.yaml                   # Milvus 数据库配置
│   ├── 📄 models.yaml                   # 模型参数配置（LLM / Embedding / Intent）
│   ├── 📄 optimized.yaml                # 优化参数配置
│   └── 📄 prompts.yaml                  # 提示词体系 + 意图路由 + 工具路由配置
│
├── 📁 data/                             # 数据与持久化
│   ├── 📄 chat_memory.db                # SQLite 对话记忆数据库
│   ├── 📁 processed/                    # 处理后数据
│   │   ├── 📄 id_cache.json             # 知识库 ID 缓存
│   │   ├── 📄 test.json                 # 测试集
│   │   └── 📄 train.json                # 训练集
│   └── 📁 raw/                          # 原始数据
│       ├── 📄 knowledge.json            # 知识库原始数据 (Alpaca 格式)
│       └── 📄 knowledge_new.json        # 新增知识库数据
│
├── 📁 flows/                            # Promptflow 工作流定义
│   ├── 📄 flow.dag.yaml                 # DAG 流程定义（6 节点拓扑执行）
│   ├── 📁 nodes/                          # 流程节点实现
│   │   ├── 📄 __init__.py
│   │   ├── 📄 assemble_prompt.py        # 提示组装节点
│   │   ├── 📄 detect_intent.py          # 意图检测节点
│   │   ├── 📄 generate_response.py      # LLM 响应生成节点
│   │   ├── 📄 load_memory.py            # 记忆加载节点
│   │   ├── 📄 retrieve_knowledge.py     # 知识检索节点
│   │   └── 📄 save_memory.py            # 记忆保存节点
│   └── 📁 templates/                      # Jinja2 提示模板
│       └── 📄 customer_service.jinja2   # 客服主模板
│
├── 📁 logs/                             # 运行与评估日志
│   ├── 📄 api.log                       # API 服务日志
│   ├── 📄 monitor.log                   # 监控日志
│   ├── 📄 streamlit.log                 # 前端服务日志
│   ├── 📄 metrics.jsonl                 # 指标日志
│   ├── 📄 benchmark_*.json              # 基准测试报告
│   ├── 📄 final_verification.json       # 最终验证报告
│   └── 📄 rag_evaluation_report.json   # RAG 评测报告
│
├── 📁 scripts/                          # 运维与工具脚本
│   ├── 📄 benchmark.py                  # 性能基准测试
│   ├── 📄 clean_logs.sh                 # 清理日志
│   ├── 📄 diagnose_api.py              # API 诊断工具
│   ├── 📄 evaluate.py                   # 综合评估
│   ├── 📄 evaluate_agent.py             # Agent 评估
│   ├── 📄 evaluate_rag.py               # RAG 评估
│   ├── 📄 final_verify.py              # 最终验证
│   ├── 📄 monitor.py                    # 系统监控
│   ├── 📄 optimize.py                   # 性能优化
│   ├── 📄 run_dashboard.py              # 启动仪表盘
│   ├── 📄 run_evaluation.py             # 运行评测
│   ├── 📄 setup.sh                      # 环境安装脚本
│   ├── 📄 start_all.sh                  # 一键启动全部服务
│   ├── 📄 stop_all.sh                   # 一键停止全部服务
│   ├── 📄 verify_modelscope_paths.py    # 验证模型路径
│   └── 📄 wsl_optimize.sh               # WSL 优化脚本
│
├── 📁 src/                              # 主源码
│   ├── 📄 __init__.py
│   ├── 📄 config.py                     # 全局配置中心（路径/参数/阈值集中管理）
│   │
│   ├── 📁 api/                            # 后端 API 服务
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main.py                     # FastAPI 应用入口（含异步预热、生命周期管理）
│   │   └── 📁 routers/
│   │       ├── 📄 admin.py                # 管理后台路由（统计/对话查询/导出）
│   │       ├── 📄 chat.py                 # 聊天路由（单轮/流式/健康检查）
│   │       └── 📄 knowledge.py            # 知识库路由
│   │
│   ├── 📁 agent/                            # Agent 核心逻辑
│   │   ├── 📄 __init__.py
│   │   ├── 📄 agent_core.py               # Agent 主类（意图→记忆→检索→生成→保存）
│   │   ├── 📄 intent_classifier.py      # 意图分类器（关键词 + BERT 双级）
│   │   ├── 📄 memory.py                   # 多轮记忆管理（SQLite 持久化）
│   │   └── 📄 tools.py                    # 工具注册表与实现（订单/库存/工单/转人工）
│   │
│   ├── 📁 models/                           # 模型加载与封装
│   │   ├── 📄 __init__.py
│   │   ├── 📄 embedding_model.py          # BGE-small-zh 封装（LlamaIndex 兼容）
│   │   ├── 📄 intent_model.py             # 意图模型封装
│   │   └── 📄 llm_client.py               # vLLM OpenAI-兼容 API 客户端
│   │
│   ├── 📁 promptflow/                       # Promptflow 引擎
│   │   ├── 📄 __init__.py
│   │   ├── 📄 flow_engine.py              # DAG 执行引擎（拓扑排序/节点缓存/错误回退）
│   │   └── 📁 templates/
│   │       ├── 📄 customer_service.yaml   # 客服模板配置
│   │       ├── 📄 sales.yaml              # 销售模板配置
│   │       └── 📄 technical_support.yaml  # 技术支持模板配置
│   │
│   ├── 📁 rag/                              # RAG 检索增强生成
│   │   ├── 📄 __init__.py
│   │   ├── 📄 data_loader.py              # Alpaca 数据加载/意图标注/向量化/入库
│   │   ├── 📄 knowledge_updater.py      # 知识库增量更新（全量重建/增量追加）
│   │   ├── 📄 llama_index_rag.py        # LlamaIndex RAG 引擎（检索+重排序）
│   │   └── 📄 milvus_store.py             # Milvus 向量存储封装（Collection/索引/检索）
│   │
│   └── 📁 ui/                               # 前端界面
│       └── 📄 streamlit_app.py            # Streamlit 管理后台（仪表盘/知识库/对话/监控/评测）
│
├── 📁 tests/                              # 测试用例
│   ├── 📄 check_id.py                     # ID 检查工具
│   ├── 📄 test_agent.py                   # Agent 单元测试
│   ├── 📄 test_api.py                     # FastAPI 端点测试
│   ├── 📄 test_milvus.py                  # Milvus 连接测试
│   ├── 📄 test_promptflow.py              # Promptflow 引擎测试
│   └── 📄 test_rag.py                     # RAG 检索测试
│
├── 📁 volumes/                            # Docker 数据卷（持久化）
│   ├── 📁 etcd/                           # Etcd 元数据存储
│   ├── 📁 milvus/                         # Milvus 向量数据
│   └── 📁 minio/                          # MinIO 对象存储
│
└── 📄 env_test.py                         # 环境检查脚本（GPU/Milvus/模型路径验证）
```

---

## 快速开始

### 环境要求

- **操作系统**: Linux / WSL2 (Windows) / macOS
- **Python**: 3.10+
- **GPU**: NVIDIA GPU (推荐 16GB+ 显存，支持 CUDA 13.2)
- **Docker & Docker Compose**: 用于运行 vLLM + Milvus
- **模型文件**: 需预先下载 Qwen2.5-1.5B、BGE-small-zh、bert-base-chinese

### 模型路径配置

模型默认通过 ModelScope 缓存加载，路径配置在 `src/config.py`：

```python
MODELSCOPE_CACHE = "/mnt/e/modelscope"  # 修改为你的实际路径

# 实际加载路径：
# /mnt/e/modelscope/Qwen/Qwen2.5-1.5B-Instruct
# /mnt/e/modelscope/BAAI/bge-small-zh-v1.5
# /mnt/e/modelscope/google-bert/bert-base-chinese
```

或通过环境变量覆盖：

```bash
export MODELSCOPE_CACHE="/path/to/your/modelscope"
```

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境检查

```bash
python env_test.py
```

输出示例：
```
PyTorch版本: 2.12.1+cu132
CUDA可用: True
CUDA版本: 12.4
GPU: NVIDIA GeForce RTX 5060 Ti
显存总量: 16.0 GB

=== 顺序加载GPU模型 ===
1. vLLM Qwen1.5B: 已在Docker运行，端口8000
2. 加载 BGE-small-zh...
   ✓ BGE加载完成 | 维度: 512 | 显存: 2.15GB
3. 加载 bert-base-chinese...
   ✓ BERT加载完成 | 显存: 2.45GB

=== 验证 Milvus 连接 ===
✓ Milvus版本: 2.4.1
=== 阶段1环境搭建完成 ===
```

### 3. 一键启动全部服务

```bash
bash scripts/start_all.sh
```

该脚本将自动执行：
1. 检查 Python3 和 Docker 依赖
2. 启动 Docker 容器（vLLM + Milvus + MinIO + Etcd + Attu）
3. 等待服务就绪（端口检测）
4. 预热 Embedding 和 Intent 模型
5. 启动 Streamlit 管理后台（端口 8501）
6. 启动 FastAPI 服务（端口 8080）

### 4. 访问服务

| 服务 | 地址 | 说明 |
|------|------|------|
| 管理后台 | http://localhost:8501 | Streamlit 仪表盘 |
| API 服务 | http://localhost:8080 | FastAPI 主服务 |
| API 文档 | http://localhost:8080/docs | Swagger UI 自动文档 |
| 健康检查 | http://localhost:8080/health | 服务状态检测 |
| Milvus UI | http://localhost:8001 | Attu 可视化工具 |

### 5. 快速测试

```bash
# 健康检查
curl http://localhost:8080/health

# 单轮对话
curl -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好，请问铅酸电池如何充电？"}'

# 带用户ID和会话ID的对话
curl -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message":"多少钱",
    "user_id":"user_001",
    "session_id":"session_001"
  }'
```

### 6. 停止服务

```bash
bash scripts/stop_all.sh
```

---

## API 文档

### 主服务接口 (端口 8080)

#### `POST /chat` — 单轮对话

**请求体：**

```json
{
  "message": "用户输入消息",
  "user_id": "anonymous",
  "session_id": "default",
  "use_flow": true
}
```

**响应体：**

```json
{
  "response": "助手回复内容",
  "intent": "销售转化",
  "category": "sales",
  "context_used": "检索到的知识上下文（截断200字符）",
  "execution_time_ms": 1234.56
}
```

#### `GET /health` — 健康检查

```json
{
  "status": "ready",
  "services": {
    "vllm": true,
    "milvus_grpc": true,
    "milvus_http": true
  }
}
```

#### `GET /metrics` — 系统指标

```json
{
  "cpu_percent": 12.5,
  "memory_percent": 45.2,
  "gpu_allocated_gb": 2.15,
  "gpu_total_gb": 16.0,
  "models_ready": true
}
```

### 子路由接口 (前缀 `/api/v1`)

#### 聊天路由 (`/api/v1/chat`)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/send` | 单轮对话 |
| POST | `/stream` | SSE 流式对话 |
| GET | `/health` | 聊天服务健康检查 |

#### 管理路由 (`/api/v1/admin`)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/stats` | 系统统计概览 |
| GET | `/conversations` | 查询对话记录 |
| DELETE | `/conversations/clear` | 清空对话记录 |

---

## 核心模块详解

### 1. 意图识别 (`src/agent/intent_classifier.py`)

采用**双级识别策略**：

- **L1 关键词规则**：快速匹配高频场景（覆盖价格、故障、投诉等关键词），置信度 0.85
- **L2 BERT 模型**：基于 `bert-base-chinese` 的 4 分类模型（标准客服/销售转化/技术支持/投诉处理），处理长尾语义

**输出格式：**
```python
{
    "intent": "销售转化",
    "category": "sales",
    "priority": 2,        # 1=最高(投诉), 4=最低(一般)
    "keywords": ["价格"],
    "confidence": 0.92,
    "source": "bert"      # 或 "keyword" / "fallback"
}
```

### 2. RAG 检索 (`src/rag/llama_index_rag.py`)

**检索流程：**
1. 用户查询 → BGE-small-zh 编码为 512 维向量
2. Milvus HNSW 索引检索 Top-K (默认 5)
3. 相似度过滤（COSINE 距离 < 0.7 丢弃）
4. 去重重排序（取 Top-3）
5. 格式化上下文注入 Prompt

**配置参数：**
```python
RAGConfig(
    top_k=5,
    similarity_cutoff=0.7,  # COSINE 距离，越接近 1 越相似
    rerank_top_n=3,
    use_reranker=True
)
```

### 3. 多轮记忆 (`src/agent/memory.py`)

- SQLite 持久化，线程安全连接池
- 支持 `user_id` + `session_id` 两级隔离
- 默认保留最近 3 轮（受 2048 token 上下文约束）
- 自动格式化历史对话为 Prompt 注入格式

### 4. 工具调用 (`src/agent/tools.py`)

内置工具清单：

| 工具 | 功能 | 适用场景 |
|------|------|----------|
| `query_order` | 查询订单信息 | 售后/物流 |
| `check_inventory` | 查询产品库存 | 销售咨询 |
| `create_ticket` | 创建工单 | 技术支持/投诉 |
| `transfer_to_human` | 转接人工客服 | 复杂/紧急问题 |
| `send_email` | 发送邮件 | 通知/跟进 |

### 5. Promptflow 引擎 (`src/promptflow/flow_engine.py`)

模拟 Azure Promptflow 的 DAG 执行引擎：

- **拓扑排序**：自动解析节点依赖，按正确顺序执行
- **变量引用**：支持 `${node_name.output}` 引用语法
- **节点缓存**：同 session 内避免重复执行
- **错误回退**：节点失败时返回默认输出，不中断流程
- **Jinja2 模板**：支持从文件或 YAML 配置动态渲染提示词

**DAG 流程定义** (`flows/flow.dag.yaml`)：
```yaml
输入 → detect_intent → load_memory → retrieve_knowledge
  → assemble_prompt → generate_response → save_memory → 输出
```

### 6. 知识库管理 (`src/rag/data_loader.py`)

支持 Alpaca 格式数据：

- 自动检测 JSON 格式（标准数组 / JSON Lines / 多对象拼接）
- 基于关键词的半监督意图标注
- 80/20 分层抽样（保持意图分布一致）
- GPU 批量向量化（500 条/批次）
- 分批插入 Milvus（1000 条/批次）

---

## 测试与评估

### 单元测试

```bash
# 运行全部测试
pytest tests/ -v

# 单独测试
pytest tests/test_api.py -v
pytest tests/test_rag.py -v
pytest tests/test_agent.py -v
pytest tests/test_milvus.py -v
pytest tests/test_promptflow.py -v
```

### 评估脚本

```bash
# RAG 评估
python scripts/evaluate_rag.py

# Agent 评估
python scripts/evaluate_agent.py

# 综合评估
python scripts/evaluate.py

# 性能基准测试
python scripts/benchmark.py

# 最终验证
python scripts/final_verify.py
```

### 诊断工具

```bash
# API 服务诊断
python scripts/diagnose_api.py

# 模型路径验证
python scripts/verify_modelscope_paths.py
```

---

## 开发路线图

### 当前阶段 (v1.0)

- ✅ FastAPI + Streamlit 双服务架构
- ✅ BGE + Milvus RAG 检索链路
- ✅ 关键词 + BERT 双级意图识别
- ✅ SQLite 多轮记忆管理
- ✅ Promptflow DAG 执行引擎
- ✅ 内置工具调用（订单/库存/工单/转人工）
- ✅ Docker Compose 容器化部署
- ✅ 管理后台（仪表盘/知识库/对话/监控）

### 下一阶段 (v2.0)

- 🔄 对接 vLLM 原生 stream 接口（真正流式输出）
- 🔄 BGE-Reranker-large 重排序模型
- 🔄 BERT 意图模型在 11,157 条数据上微调
- 🔄 多 Agent 协作（销售 Agent + 技术 Agent）
- 🔄 生产级对话日志与埋点分析
- 🔄 对接真实业务系统 API（ERP/CRM/工单系统）
- 🔄 权限管理与多租户支持
- 🔄 迁移至 Azure AI Studio Promptflow

---

## 硬件配置参考

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | 8GB 显存 | 16GB+ 显存 (RTX 4060/5060 Ti+) |
| 内存 | 8GB | 16GB+ |
| 磁盘 | 20GB SSD | 50GB+ SSD |
| CUDA | 11.8+ | 12.4+ |

**当前开发环境:**
- WSL2 + Ubuntu 22.04
- NVIDIA GeForce RTX 5060 Ti (16GB)
- CUDA 13.2
- 总显存占用: ~2.5GB (vLLM 在 Docker 内)

---

## 常见问题 (FAQ)

**Q: vLLM 启动失败？**
> 检查 Docker 日志：`docker logs vllm-qwen`。确保模型路径正确挂载，且模型文件已下载。

**Q: Milvus 连接失败？**
> 确认 Docker 容器已启动：`docker ps | grep milvus`。Milvus 首次启动需要 30-60 秒初始化。

**Q: BGE 模型加载失败？**
> 检查模型路径是否存在，或通过 `MODELSCOPE_CACHE` 环境变量指定正确路径。

**Q: 显存不足？**
> 调整 `docker-compose.yml` 中 vLLM 的 `gpu-memory-utilization` 参数，或降低 `max-model-len`。

**Q: 如何添加新知识？**
> 1. 通过 Streamlit 管理后台「知识库管理」页面直接添加
> 2. 或编辑 `data/raw/knowledge.json` 后运行 `python src/rag/data_loader.py`

---

## 贡献指南

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/xxx`
3. 提交更改：`git commit -m "feat: xxx"`
4. 推送分支：`git push origin feature/xxx`
5. 创建 Pull Request

---

## 许可证

[MIT License](LICENSE)

---

> 本项目为演示/学习用途，生产部署请根据实际需求进行安全加固和性能优化。
