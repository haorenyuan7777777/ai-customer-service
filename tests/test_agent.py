"""
Agent功能测试 - 意图识别 + 记忆 + 工具调用
"""
import pytest
import os
import tempfile
from src.models.intent_model import IntentClassifier
from src.agent.memory import ChatMemory
from src.agent.tools import ToolRegistry


class TestIntentClassification:
    """测试意图识别"""
    
    @pytest.fixture(scope="class")
    def classifier(self):
        return IntentClassifier()
    
    def test_rule_based_intent(self, classifier):
        """测试规则匹配意图"""
        test_cases = [
            ("这个多少钱？", "price_inquiry"),
            ("我想购买这个", "purchase_intent"),
            ("电池坏了怎么修？", "technical_issue"),
            ("我要投诉你们", "complaint"),
            ("你好，请问在吗？", "general_query"),
        ]
        
        for text, expected in test_cases:
            result = classifier.predict(text, use_rule=True)
            assert result["intent"] == expected, f"'{text}' 应识别为 {expected}，实际是 {result['intent']}"
            assert result["method"] == "rule"
            print(f"✅ '{text}' -> {result['intent']} (规则匹配)")
    
    def test_model_based_intent(self, classifier):
        """测试模型推理意图"""
        # 规则未命中的情况，应回退到模型
        text = "这款产品性价比如何"
        result = classifier.predict(text, use_rule=False)
        
        assert "intent" in result
        assert "confidence" in result
        assert result["method"] == "model"
        assert sum(result["probabilities"].values()) > 0.99
        
        print(f"✅ 模型推理: '{text}' -> {result['intent']} (置信度: {result['confidence']:.3f})")
    
    def test_intent_confidence(self, classifier):
        """测试意图置信度合理性"""
        result = classifier.predict("价格是多少")
        assert 0 <= result["confidence"] <= 1
        assert result["probabilities"]["price_inquiry"] > 0.5
        print(f"✅ 置信度测试通过: {result['confidence']:.3f}")


class TestChatMemory:
    """测试多轮记忆"""
    
    @pytest.fixture
    def memory(self):
        # 使用临时数据库
        db_path = tempfile.mktemp(suffix=".db")
        mem = ChatMemory(db_path=db_path)
        yield mem
        # 清理
        os.remove(db_path)
    
    def test_save_and_retrieve(self, memory):
        """测试保存和读取对话历史"""
        session_id = "test_session_001"
        
        # 保存对话
        memory.save(session_id, "user", "你好")
        memory.save(session_id, "assistant", "您好，有什么可以帮您？")
        memory.save(session_id, "user", "电池怎么充电")
        
        # 读取历史
        history = memory.get_history(session_id, limit=3)
        
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
        
        print(f"✅ 记忆读写测试通过，共{len(history)}轮对话")
    
    def test_history_limit(self, memory):
        """测试历史记录限制"""
        session_id = "test_limit"
        
        # 保存5轮对话
        for i in range(5):
            memory.save(session_id, "user", f"问题{i}")
            memory.save(session_id, "assistant", f"回答{i}")
        
        # 限制读取3轮
        history = memory.get_history(session_id, limit=3)
        assert len(history) == 3
        # 应返回最近的3条
        assert history[-1]["content"] == "回答4"
        
        print(f"✅ 历史限制测试通过，返回最近{len(history)}轮")
    
    def test_multiple_sessions(self, memory):
        """测试多会话隔离"""
        memory.save("session_a", "user", "会话A的问题")
        memory.save("session_b", "user", "会话B的问题")
        
        history_a = memory.get_history("session_a")
        history_b = memory.get_history("session_b")
        
        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0]["content"] == "会话A的问题"
        assert history_b[0]["content"] == "会话B的问题"
        
        print(f"✅ 多会话隔离测试通过")


class TestToolRegistry:
    """测试工具调用"""
    
    @pytest.fixture
    def registry(self):
        reg = ToolRegistry()
        
        # 注册测试工具
        @reg.register(name="get_price", description="查询价格")
        def get_price(product: str) -> dict:
            return {"product": product, "price": 199.99}
        
        @reg.register(name="check_stock", description="检查库存")
        def check_stock(product: str) -> dict:
            return {"product": product, "stock": 100}
        
        return reg
    
    def test_tool_list(self, registry):
        """测试工具列表"""
        tools = registry.list_tools()
        assert len(tools) == 2
        assert any(t["name"] == "get_price" for t in tools)
        print(f"✅ 工具列表测试通过，共{len(tools)}个工具")
    
    def test_tool_execution(self, registry):
        """测试工具执行"""
        result = registry.execute("get_price", {"product": "铅酸电池"})
        assert result["product"] == "铅酸电池"
        assert result["price"] == 199.99
        print(f"✅ 工具执行测试通过: {result}")
    
    def test_tool_not_found(self, registry):
        """测试工具不存在"""
        with pytest.raises(ValueError):
            registry.execute("nonexistent_tool", {})
        print(f"✅ 工具不存在异常测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])