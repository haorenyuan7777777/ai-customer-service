"""
Streamlit管理后台
知识库CRUD、对话记录查看、Excel导出
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

API_BASE = "http://localhost:8080/api/v1"

st.set_page_config(
    page_title="AI客服管理后台",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI客服系统管理后台")

# 侧边栏导航
page = st.sidebar.radio("功能菜单", [
    "对话测试", "知识库管理", "会话记录", "系统状态"
])

if page == "对话测试":
    st.header("💬 对话测试")
    
    session_id = st.text_input("会话ID", value="test_session_001")
    query = st.text_area("输入问题", value="铅酸蓄电池正确使用的注意事项有哪些？")
    
    if st.button("发送", type="primary"):
        with st.spinner("处理中..."):
            try:
                resp = requests.post(f"{API_BASE}/chat", json={
                    "query": query,
                    "session_id": session_id
                })
                result = resp.json()
                
                st.success("回复生成成功")
                st.markdown(f"**意图识别**: {result['intent']} (置信度: {result['intent_confidence']:.3f})")
                st.markdown(f"**回复**:")
                st.info(result['response'])
                
                if result['sources']:
                    with st.expander("查看知识来源"):
                        for src in result['sources']:
                            st.markdown(f"- [{src['score']:.3f}] {src['instruction']}")
                
            except Exception as e:
                st.error(f"请求失败: {e}")

elif page == "知识库管理":
    st.header("📚 知识库管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("添加知识")
        instruction = st.text_area("问题")
        output = st.text_area("答案")
        category = st.selectbox("分类", ["general", "price", "technical", "sales"])
        
        if st.button("添加", type="primary"):
            try:
                resp = requests.post(f"{API_BASE}/knowledge", json={
                    "instruction": instruction,
                    "output": output,
                    "category": category
                })
                if resp.status_code == 200:
                    st.success("添加成功!")
                else:
                    st.error(f"添加失败: {resp.text}")
            except Exception as e:
                st.error(f"请求失败: {e}")
    
    with col2:
        st.subheader("检索知识")
        search_query = st.text_input("搜索关键词")
        if st.button("搜索"):
            try:
                resp = requests.get(f"{API_BASE}/knowledge/search", 
                                    params={"query": search_query, "top_k": 10})
                results = resp.json().get("results", [])
                
                for r in results:
                    with st.container():
                        st.markdown(f"**[{r['score']:.3f}]** {r['instruction']}")
                        st.text(r['output'][:100] + "...")
                        st.divider()
            except Exception as e:
                st.error(f"搜索失败: {e}")

elif page == "会话记录":
    st.header("📝 会话记录")
    
    try:
        resp = requests.get(f"{API_BASE}/conversations")
        sessions = resp.json().get("sessions", [])
        
        selected_session = st.selectbox("选择会话", sessions)
        
        if selected_session:
            resp = requests.get(f"{API_BASE}/conversations/{selected_session}")
            history = resp.json().get("history", [])
            
            for entry in history:
                role = entry["role"]
                content = entry["content"]
                if role == "user":
                    st.chat_message("user").write(content)
                else:
                    st.chat_message("assistant").write(content)
        
        # 导出按钮
        if st.button("导出全部对话(Excel)"):
            resp = requests.get(f"{API_BASE}/conversations/export")
            if resp.status_code == 200:
                st.download_button(
                    "下载Excel",
                    resp.content,
                    file_name=f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    except Exception as e:
        st.error(f"加载失败: {e}")

elif page == "系统状态":
    st.header("📊 系统状态")
    
    try:
        resp = requests.get(f"{API_BASE}/stats")
        stats = resp.json()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("知识库数量", stats["milvus"]["num_entities"])
        
        with col2:
            st.metric("总对话数", stats["memory"]["total_messages"])
        
        with col3:
            gpu = stats.get("gpu", {})
            if gpu:
                st.metric("GPU显存占用", f"{gpu.get('allocated_gb', 0):.1f}GB")
        
        # GPU显存图表
        if gpu:
            st.subheader("GPU显存使用")
            gpu_data = {
                "已分配": gpu.get("allocated_gb", 0),
                "空闲": gpu.get("total_gb", 16) - gpu.get("allocated_gb", 0)
            }
            st.bar_chart(gpu_data)
        
    except Exception as e:
        st.error(f"获取状态失败: {e}")