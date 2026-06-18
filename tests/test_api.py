"""
API接口测试 - FastAPI端点测试
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


class TestChatAPI:
    """测试对话接口"""
    
    def test_chat_endpoint(self):
        """测试基础对话接口"""
        response = client.post("/api/v1/chat", json={
            "query": "铅酸蓄电池正确使用的注意事项有哪些？",
            "session_id": "test_api_001"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "intent" in data
        print(f"✅ 对话接口测试通过")
        print(f"   意图: {data.get('intent')}")
        print(f"   回复: {data.get('response', '')[:50]}...")
    
    def test_chat_with_history(self):
        """测试带历史记录的对话"""
        # 第一轮
        client.post("/api/v1/chat", json={
            "query": "你好",
            "session_id": "test_history_001"
        })
        
        # 第二轮
        response = client.post("/api/v1/chat", json={
            "query": "电池怎么充电",
            "session_id": "test_history_001"
        })
        
        assert response.status_code == 200
        print(f"✅ 多轮对话接口测试通过")
    
    def test_chat_invalid_request(self):
        """测试无效请求处理"""
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422  # FastAPI验证错误
        print(f"✅ 无效请求处理测试通过")


class TestKnowledgeAPI:
    """测试知识库接口"""
    
    def test_add_knowledge(self):
        """测试添加知识"""
        response = client.post("/api/v1/knowledge", json={
            "instruction": "测试问题",
            "output": "测试答案",
            "category": "test"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        print(f"✅ 添加知识测试通过")
    
    def test_search_knowledge(self):
        """测试知识检索"""
        response = client.get("/api/v1/knowledge/search", params={
            "query": "铅酸电池",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        print(f"✅ 知识检索测试通过，返回{len(data.get('results', []))}条结果")
    
    def test_delete_knowledge(self):
        """测试删除知识"""
        # 先添加
        add_resp = client.post("/api/v1/knowledge", json={
            "instruction": "待删除问题",
            "output": "待删除答案"
        })
        doc_id = add_resp.json().get("id", "test_delete_001")
        
        # 再删除
        response = client.delete(f"/api/v1/knowledge/{doc_id}")
        assert response.status_code in [200, 404]  # 成功或已不存在
        print(f"✅ 删除知识测试通过")


class TestAdminAPI:
    """测试管理接口"""
    
    def test_health_check(self):
        """测试健康检查"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print(f"✅ 健康检查测试通过: {data}")
    
    def test_stats(self):
        """测试统计信息"""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "milvus" in data
        assert "gpu" in data
        print(f"✅ 统计信息测试通过")
    
    def test_export_conversations(self):
        """测试导出对话记录"""
        response = client.get("/api/v1/conversations/export")
        assert response.status_code == 200
        assert response.headers["content-type"] in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/json"
        ]
        print(f"✅ 导出对话测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])