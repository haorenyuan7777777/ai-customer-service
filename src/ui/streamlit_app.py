"""
Streamlit管理后台
- 知识库CRUD（浏览/搜索/添加/删除/Excel导出）
- 对话记录查看（按用户/会话筛选/Excel导出）
- 系统监控仪表盘（CPU/内存/磁盘/GPU实时）
- 评测报告展示

【演示实现】：直接读取本地文件+Milvus+SQLite
【生产目标】：对接独立后端API，支持多用户权限
"""

import os
import sys
import json
import sqlite3
import time
import socket
from pathlib import Path
from datetime import datetime
from collections import Counter
from io import BytesIO

import streamlit as st
import pandas as pd

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config, DATA_PATHS, DATABASE, MODELS, MILVUS, HARDWARE
from src.rag.milvus_store import get_milvus_store
from src.models.embedding_model import get_embedding_model

# ============ 页面配置 ============
st.set_page_config(
    page_title="AI客服系统管理后台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CSS样式 ============
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .stProgress > div > div > div > div {
        background-color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# ============ 数据加载函数 ============

@st.cache_data(ttl=60)
def load_knowledge_data():
    """加载知识库原始数据，支持标准 JSON 数组或每行一个 JSON 对象"""
    raw_path = DATA_PATHS["raw"]
    if not Path(raw_path).exists():
        return []
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return []
        
        # 1. 尝试直接解析为标准 JSON 数组
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            else:
                # 如果是单个对象，包装成列表
                return [data]
        except json.JSONDecodeError:
            # 2. 尝试按行解析（每行一个 JSON 对象）
            lines = content.splitlines()
            data = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        data.append(obj)
                    except json.JSONDecodeError:
                        continue
            return data

@st.cache_data(ttl=30)
def load_conversations():
    """加载对话记录（SQLite）"""
    db_path = DATABASE["path"]
    if not Path(db_path).exists():
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("""
        SELECT user_id, session_id, user_message, assistant_message, intent, created_at
        FROM chat_history
        ORDER BY created_at DESC
        LIMIT 1000
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@st.cache_data(ttl=60)
def get_milvus_stats():
    """获取Milvus统计"""
    try:
        store = get_milvus_store()
        return store.get_stats()
    except Exception as e:
        return {"error": str(e), "total_entities": 0}

def get_system_metrics():
    """获取系统资源指标"""
    import psutil
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "memory": psutil.virtual_memory()._asdict(),
        "disk": psutil.disk_usage('/')._asdict(),
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            metrics["gpu"] = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            }
    except ImportError:
        pass
    
    return metrics

def check_port(host, port, timeout=1):
    """检测端口是否连通"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except:
        return False

# ============ 页面：仪表盘 ============

def render_dashboard():
    st.title("📊 系统仪表盘")
    
    # 加载数据
    knowledge = load_knowledge_data()
    conversations = load_conversations()
    milvus_stats = get_milvus_stats()
    metrics = get_system_metrics()
    
    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(knowledge):,}</div>
            <div class="metric-label">知识库条目</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(conversations):,}</div>
            <div class="metric-label">历史对话</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        entities = milvus_stats.get("total_entities", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{entities:,}</div>
            <div class="metric-label">向量索引</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        gpu = metrics.get("gpu", {})
        gpu_text = f"{gpu.get('allocated_gb', 0):.1f}G" if gpu else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{gpu_text}</div>
            <div class="metric-label">GPU显存占用</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 图表区域
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🎯 意图分布（最近对话）")
        if conversations:
            intents = [c.get("intent") or "标准客服" for c in conversations]
            intent_counts = Counter(intents)
            
            # 使用Streamlit原生chart（避免plotly依赖）
            df_intent = pd.DataFrame({
                "意图": list(intent_counts.keys()),
                "数量": list(intent_counts.values())
            })
            st.bar_chart(df_intent.set_index("意图"))
        else:
            st.info("暂无对话数据")
    
    with col_right:
        st.subheader("💻 系统资源")
        mem = metrics.get("memory", {})
        if mem:
            mem_percent = mem.get("percent", 0)
            st.progress(min(mem_percent / 100, 1.0), text=f"内存: {mem_percent}%")
        
        disk = metrics.get("disk", {})
        if disk:
            used_gb = disk.get("used", 0) / 1024**3
            total_gb = disk.get("total", 0) / 1024**3
            disk_percent = (used_gb / total_gb * 100) if total_gb else 0
            st.progress(min(disk_percent / 100, 1.0), 
                       text=f"磁盘: {used_gb:.1f}/{total_gb:.1f}GB ({disk_percent:.1f}%)")
        
        if "gpu" in metrics:
            gpu = metrics["gpu"]
            alloc_percent = gpu['allocated_gb'] / gpu['total_gb'] * 100
            st.progress(min(alloc_percent / 100, 1.0),
                       text=f"GPU: {gpu['allocated_gb']:.1f}/{gpu['total_gb']:.1f}GB ({alloc_percent:.1f}%)")

# ============ 页面：知识库管理 ============

def render_knowledge_management():
    st.title("📚 知识库管理")
    
    knowledge = load_knowledge_data()
    
    # 搜索
    search_query = st.text_input("🔍 搜索知识库（支持instruction内容模糊搜索）", placeholder="输入关键词...")
    if search_query:
        filtered = [k for k in knowledge if search_query in k.get("instruction", "")]
    else:
        filtered = knowledge
    
    st.caption(f"展示 {len(filtered)} / 总计 {len(knowledge)} 条记录")
    
    # 表格展示
    if filtered:
        df = pd.DataFrame(filtered)
        display_cols = ["id", "instruction", "output", "intent", "category"]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], use_container_width=True, height=400)
    else:
        st.info("未找到匹配记录")
    
    st.divider()
    
    # 添加新知识
    st.subheader("➕ 添加新知识")
    with st.form("add_knowledge_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_instruction = st.text_area("问题（instruction）*", height=100, 
                placeholder="请输入用户可能问的问题...")
            new_intent = st.selectbox("意图标签*", 
                ["general_query", "price_inquiry", "purchase_intent", "technical_issue", "complaint"],
                index=0)
        with col2:
            new_output = st.text_area("答案（output）*", height=100,
                placeholder="请输入标准答案...")
            new_category = st.text_input("分类（category）", value="general",
                help="业务分类，如：售前咨询、售后支持等")
        
        submitted = st.form_submit_button("🚀 添加并GPU向量化", type="primary", use_container_width=True)
        
        if submitted:
            if not new_instruction or not new_output:
                st.error("❌ 问题和答案为必填项")
            else:
                try:
                    # 生成新ID
                    new_id = max([k.get("id", 0) for k in knowledge] + [0]) + 1
                    
                    new_item = {
                        "id": new_id,
                        "instruction": new_instruction,
                        "input": "",
                        "output": new_output,
                        "intent": new_intent,
                        "category": new_category
                    }
                    
                    # 1. 更新JSON
                    knowledge.append(new_item)
                    with open(DATA_PATHS["raw"], 'w', encoding='utf-8') as f:
                        json.dump(knowledge, f, ensure_ascii=False, indent=2)
                    
                    # 2. GPU向量化并插入Milvus
                    with st.spinner("正在GPU向量化并插入Milvus..."):
                        embed_model = get_embedding_model()
                        vector = embed_model.get_text_embedding(new_instruction)
                        
                        store = get_milvus_store()
                        store.load_collection()
                        
                        # 构造插入数据（按字段列表组织）
                        store.collection.insert([
                            [new_id],           # id
                            [vector],           # vector
                            [new_instruction],  # instruction
                            [new_output],       # output
                            [new_category],     # category
                            [new_intent]        # intent
                        ])
                        
                        # 刷新索引
                        store.collection.flush()
                    
                    st.success(f"✅ 添加成功！ID: {new_id} | 已自动向量化并入库")
                    st.cache_data.clear()
                    
                except Exception as e:
                    st.error(f"❌ 操作失败: {str(e)}")
                    st.info("提示：请确认Milvus服务已启动（端口19530）且BGE模型已加载")
    
    # 删除功能
    st.subheader("🗑️ 删除知识")
    del_col1, del_col2 = st.columns([3, 1])
    with del_col1:
        delete_id = st.number_input("输入要删除的ID", min_value=1, step=1, value=1)
    with del_col2:
        st.write("")
        st.write("")
        if st.button("删除", type="secondary", use_container_width=True):
            # 从JSON删除
            new_knowledge = [k for k in knowledge if k.get("id") != delete_id]
            if len(new_knowledge) == len(knowledge):
                st.warning(f"ID {delete_id} 不存在")
            else:
                with open(DATA_PATHS["raw"], 'w', encoding='utf-8') as f:
                    json.dump(new_knowledge, f, ensure_ascii=False, indent=2)
                
                # 【演示实现】Milvus删除需通过ID精确删除
                try:
                    store = get_milvus_store()
                    store.load_collection()
                    store.collection.delete(f"id in [{delete_id}]")
                    st.success(f"✅ 已删除ID {delete_id}（JSON + Milvus）")
                except Exception as e:
                    st.warning(f"JSON已删除，Milvus删除失败（可忽略）: {e}")
                
                st.cache_data.clear()
    
    # 批量导入/导出
    st.divider()
    st.subheader("📤 批量导入 / 导出")
    imp_col, exp_col = st.columns(2)
    
    with imp_col:
        uploaded = st.file_uploader("上传JSON文件（Alpaca格式）", type=["json"])
        if uploaded is not None:
            try:
                new_data = json.load(uploaded)
                if isinstance(new_data, list):
                    # 补充ID
                    max_id = max([k.get("id", 0) for k in knowledge] + [0])
                    for i, item in enumerate(new_data):
                        if "id" not in item:
                            item["id"] = max_id + i + 1
                    
                    knowledge.extend(new_data)
                    with open(DATA_PATHS["raw"], 'w', encoding='utf-8') as f:
                        json.dump(knowledge, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"✅ 成功导入 {len(new_data)} 条记录")
                    st.info("⚠️ 请前往【系统监控】页面点击「重建Milvus索引」完成向量化")
                    st.cache_data.clear()
                else:
                    st.error("JSON格式错误：应为列表")
            except Exception as e:
                st.error(f"导入失败: {e}")
    
    with exp_col:
        if knowledge:
            df_exp = pd.DataFrame(knowledge)
            csv = df_exp.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 导出CSV",
                data=csv,
                file_name=f"knowledge_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Excel导出
            excel_buffer = BytesIO()
            df_exp.to_excel(excel_buffer, index=False, engine='openpyxl')
            st.download_button(
                label="📥 导出Excel",
                data=excel_buffer.getvalue(),
                file_name=f"knowledge_export_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# ============ 页面：对话记录 ============

def render_conversations():
    st.title("💬 对话记录")
    
    conversations = load_conversations()
    
    if not conversations:
        st.info("暂无对话记录，请先与AI客服对话")
        return
    
    # 筛选器
    col1, col2, col3 = st.columns(3)
    with col1:
        users = ["全部"] + sorted(list(set(c["user_id"] for c in conversations)))
        selected_user = st.selectbox("按用户筛选", users)
    with col2:
        sessions = ["全部"] + sorted(list(set(c["session_id"] for c in conversations)))
        selected_session = st.selectbox("按会话筛选", sessions)
    with col3:
        intents = ["全部"] + sorted(list(set(c.get("intent") or "标准客服" for c in conversations)))
        selected_intent = st.selectbox("按意图筛选", intents)
    
    filtered = conversations
    if selected_user != "全部":
        filtered = [c for c in filtered if c["user_id"] == selected_user]
    if selected_session != "全部":
        filtered = [c for c in filtered if c["session_id"] == selected_session]
    if selected_intent != "全部":
        filtered = [c for c in filtered if (c.get("intent") or "标准客服") == selected_intent]
    
    st.caption(f"共 {len(filtered)} 条记录（展示最近50条）")
    
    # 展示对话
    for conv in filtered[:50]:
        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                badge_color = {
                    "投诉处理": "🔴",
                    "销售转化": "🟢",
                    "技术支持": "🔵",
                    "标准客服": "⚪"
                }.get(conv.get("intent") or "标准客服", "⚪")
                st.markdown(f"{badge_color} **{conv.get('intent') or '标准客服'}**")
                st.caption(f"🕐 {conv['created_at'][:16]}")
            with cols[1]:
                with st.expander(f"👤 {conv['user_message'][:60]}{'...' if len(conv['user_message']) > 60 else ''}"):
                    st.markdown(f"**用户**: {conv['user_message']}")
                    st.markdown(f"**助手**: {conv['assistant_message']}")
                    st.divider()
                    st.caption(f"user_id: `{conv['user_id']}` | session_id: `{conv['session_id']}`")
    
    # 导出
    if filtered:
        st.divider()
        df_conv = pd.DataFrame(filtered)
        csv = df_conv.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 导出对话记录为CSV",
            data=csv,
            file_name=f"conversations_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============ 页面：系统监控 ============

def render_system_monitor():
    st.title("🔧 系统监控")
    
    # 自动刷新控制
    auto_refresh = st.checkbox("🔁 自动刷新（每5秒）", value=False)
    if auto_refresh:
        st.empty()
        time.sleep(0.5)  # 给UI响应时间
    
    metrics = get_system_metrics()
    
    # 资源卡片
    st.subheader("硬件资源")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu = metrics.get("cpu_percent", 0)
        st.metric("CPU使用率", f"{cpu}%")
        st.progress(min(cpu / 100, 1.0), text=f"{cpu}%")
    
    with col2:
        mem = metrics.get("memory", {})
        if mem:
            used_gb = mem.get("used", 0) / 1024**3
            total_gb = mem.get("total", 0) / 1024**3
            percent = mem.get("percent", 0)
            st.metric("内存使用", f"{used_gb:.1f}/{total_gb:.1f} GB")
            st.progress(min(percent / 100, 1.0), text=f"{percent}%")
    
    with col3:
        disk = metrics.get("disk", {})
        if disk:
            used_gb = disk.get("used", 0) / 1024**3
            total_gb = disk.get("total", 0) / 1024**3
            percent = (used_gb / total_gb * 100) if total_gb else 0
            st.metric("磁盘使用", f"{used_gb:.1f}/{total_gb:.1f} GB")
            st.progress(min(percent / 100, 1.0), text=f"{percent:.1f}%")
    
    # GPU监控
    st.divider()
    st.subheader("🎮 GPU显存监控")
    gpu = metrics.get("gpu", {})
    if gpu:
        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        with col_g1:
            st.metric("已分配显存", f"{gpu['allocated_gb']:.2f} GB")
        with col_g2:
            st.metric("预留显存", f"{gpu['reserved_gb']:.2f} GB")
        with col_g3:
            st.metric("显存总量", f"{gpu['total_gb']:.2f} GB")
        with col_g4:
            free = gpu['total_gb'] - gpu['allocated_gb']
            st.metric("剩余显存", f"{free:.2f} GB")
        
        alloc_percent = gpu['allocated_gb'] / gpu['total_gb'] * 100
        color = "#ff4b4b" if alloc_percent > 80 else "#0066cc"
        st.progress(min(alloc_percent / 100, 1.0), 
                   text=f"显存占用: {gpu['allocated_gb']:.2f}/{gpu['total_gb']:.2f}GB ({alloc_percent:.1f}%)")
    else:
        st.info("未检测到NVIDIA GPU")
    
    # 服务状态检测
    st.divider()
    st.subheader("🔌 服务状态检测")
    
    services = [
        ("vLLM (Qwen2.5-1.5B)", "localhost", 8000, "大模型推理服务"),
        ("Milvus gRPC", "localhost", 19530, "向量数据库"),
        ("Milvus HTTP", "localhost", 9091, "向量数据库HTTP"),
        ("Milvus MinIO", "localhost", 9001, "对象存储"),
        ("Attu (Milvus UI)", "localhost", 8001, "Milvus可视化"),
    ]
    
    svc_cols = st.columns(len(services))
    for idx, (name, host, port, desc) in enumerate(services):
        is_up = check_port(host, port)
        with svc_cols[idx]:
            if is_up:
                st.success(f"🟢 {name}\n\n`{host}:{port}`")
            else:
                st.error(f"🔴 {name}\n\n`{host}:{port}`")
            st.caption(desc)
    
    # 运维操作
    st.divider()
    st.subheader("🛠️ 运维操作")
    
    op_col1, op_col2 = st.columns(2)
    with op_col1:
        if st.button("🔄 重建Milvus索引", type="primary", use_container_width=True):
            with st.spinner("正在重建索引（可能需要几分钟）..."):
                try:
                    # 【演示实现】调用知识库更新器
                    from src.rag.knowledge_updater import rebuild_index
                    rebuild_index()
                    st.success("✅ Milvus索引重建完成")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"❌ 重建失败: {e}")
                    st.info("提示：请确认Milvus和BGE模型服务正常")
    
    with op_col2:
        if st.button("🧹 清空对话记录", type="secondary", use_container_width=True):
            try:
                db_path = DATABASE["path"]
                if Path(db_path).exists():
                    conn = sqlite3.connect(db_path)
                    conn.execute("DELETE FROM chat_history")
                    conn.commit()
                    conn.close()
                    st.success("✅ 对话记录已清空")
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"❌ 清空失败: {e}")

# ============ 页面：评测报告 ============

def render_evaluation():
    st.title("📈 评测报告")
    
    report_path = PROJECT_ROOT / "logs" / "rag_evaluation_report.json"
    
    if not report_path.exists():
        st.info("暂无评测报告，请先运行评测脚本生成报告。")
        st.code("python scripts/evaluate_rag.py", language="bash")
        st.caption("运行上述命令生成评测报告后，刷新本页面查看")
        return
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"❌ 报告文件格式错误（JSON解析失败）: {e}")
        return
    except Exception as e:
        st.error(f"❌ 读取报告文件失败: {e}")
        return
    
    # 确保报告是字典
    if not isinstance(report, dict):
        st.error(f"报告根格式错误，期望字典，实际为: {type(report).__name__}")
        st.json(report)  # 展示原始内容便于调试
        return
    
    # 获取 tests 部分（如果存在），否则使用整个报告
    tests = report.get("tests")
    if tests is None:
        tests = report
    
    # 如果 tests 不是字典，直接展示整个报告
    if not isinstance(tests, dict):
        st.warning("报告中的 'tests' 字段不是字典，显示整个报告内容：")
        st.json(report)
        return
    
    # 遍历 tests 字典
    for name, result in tests.items():
        with st.expander(f"📋 {name}", expanded=True):
            # 如果 result 不是字典，直接显示其值
            if not isinstance(result, dict):
                st.write(result)
                continue
            
            col1, col2 = st.columns([1, 2])
            with col1:
                status = result.get("status", "未知")
                if "PASS" in str(status):
                    st.success(status)
                elif "SKIP" in str(status) or "⏭️" in str(status):
                    st.info(status)
                else:
                    st.warning(status)
            with col2:
                st.json(result)
    
    # 显示总结（如果存在）
    if "summary" in report and isinstance(report["summary"], dict):
        st.divider()
        st.subheader("📊 总评")
        summary = report["summary"]
        st.json(summary)


# ============ 侧边栏 & 主入口 ============

def main():
    with st.sidebar:
        st.title("🤖 AI客服系统")
        st.markdown("---")
        
        page = st.radio(
            "📍 导航菜单",
            ["仪表盘", "知识库管理", "对话记录", "系统监控", "评测报告"],
            index=0
        )
        
        st.markdown("---")
        st.caption(f"**环境**: WSL2 + RTX 5060Ti")
        st.caption(f"**版本**: v1.0 | 阶段6/8")
        
        # 快捷操作
        st.markdown("---")
        st.subheader("⚡ 快捷操作")
        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # 路由
    if page == "仪表盘":
        render_dashboard()
    elif page == "知识库管理":
        render_knowledge_management()
    elif page == "对话记录":
        render_conversations()
    elif page == "系统监控":
        render_system_monitor()
    elif page == "评测报告":
        render_evaluation()

if __name__ == "__main__":
    main()