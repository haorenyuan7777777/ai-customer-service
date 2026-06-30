"""
Promptflow流程测试
- 测试DAG执行
- 测试模板渲染
- 测试节点回退
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.promptflow.flow_engine import PromptFlowEngine, get_flow_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_flow():
    """测试基础流程执行"""
    logger.info("=" * 50)
    logger.info("【测试1】基础流程执行")
    
    engine = PromptFlowEngine()
    
    result = engine.run({
        "user_message": "铅酸蓄电池正确使用的注意事项有哪些？",
        "user_id": "test_user_001",
        "session_id": "session_001"
    })
    
    logger.info(f"输出keys: {list(result.keys())}")
    logger.info(f"意图: {result.get('intent')}")
    logger.info(f"分类: {result.get('category')}")
    logger.info(f"响应: {result.get('response', '')[:100]}...")
    logger.info(f"执行元信息: {result.get('_execution_meta')}")
    
    assert 'response' in result
    assert 'intent' in result
    logger.info("✅ 基础流程测试通过")
    return result


def test_sales_intent():
    """测试销售意图路由"""
    logger.info("\n" + "=" * 50)
    logger.info("【测试2】销售意图路由")
    
    engine = get_flow_engine()
    
    result = engine.run({
        "user_message": "这个电池多少钱？有优惠吗？",
        "user_id": "test_user_002"
    })
    
    logger.info(f"检测意图: {result.get('intent')}")
    logger.info(f"触发关键词: {result.get('context_used', '')[:100]}")
    
    assert result.get('intent') == "销售转化"
    logger.info("✅ 销售意图路由测试通过")


def test_complaint_intent():
    """测试投诉意图路由（高优先级）"""
    logger.info("\n" + "=" * 50)
    logger.info("【测试3】投诉意图路由")
    
    engine = get_flow_engine()
    
    result = engine.run({
        "user_message": "你们的产品太垃圾了，我要投诉！",
        "user_id": "test_user_003"
    })
    
    logger.info(f"检测意图: {result.get('intent')}")
    logger.info(f"优先级: {result.get('category')}")
    
    assert result.get('intent') == "投诉处理"
    logger.info("✅ 投诉意图路由测试通过")


def test_template_rendering():
    """测试模板渲染"""
    logger.info("\n" + "=" * 50)
    logger.info("【测试4】模板渲染")
    
    engine = get_flow_engine()
    
    # 测试标准客服模板
    prompt = engine.render_template(
        "customer_service",
        user_message="测试问题",
        memory="用户：之前问过价格\n助手：价格是100元",
        context="[知识1] 问题：电池价格？\n答案：100元"
    )
    
    logger.info(f"渲染Prompt长度: {len(prompt)}字符")
    logger.info(f"Prompt前200字: {prompt[:200]}...")
    
    assert "测试问题" in prompt
    assert "100元" in prompt
    logger.info("✅ 模板渲染测试通过")


def test_error_fallback():
    """测试节点错误回退"""
    logger.info("\n" + "=" * 50)
    logger.info("【测试5】错误回退")
    
    engine = PromptFlowEngine()
    
    # 模拟一个会导致错误的输入
    result = engine.run({
        "user_message": "",  # 空输入
        "user_id": "test_user_004"
    })
    
    # 即使出错，也应该有响应
    assert 'response' in result
    assert result['response'] != ""
    logger.info(f"错误回退响应: {result['response'][:50]}...")
    logger.info("✅ 错误回退测试通过")


def run_all_tests():
    """运行全部测试"""
    tests = [
        test_basic_flow,
        test_sales_intent,
        test_complaint_intent,
        test_template_rendering,
        test_error_fallback
    ]
    
    results = []
    for test in tests:
        try:
            test()
            results.append((test.__name__, "PASS"))
        except Exception as e:
            logger.error(f"❌ {test.__name__} 失败: {e}")
            results.append((test.__name__, f"FAIL: {e}"))
    
    logger.info("\n" + "=" * 50)
    logger.info("📋 Promptflow测试报告")
    logger.info("=" * 50)
    for name, status in results:
        icon = "✅" if status == "PASS" else "❌"
        logger.info(f"{icon} {name}: {status}")
    
    all_pass = all(s == "PASS" for _, s in results)
    logger.info(f"\n总评: {'✅ 全部通过' if all_pass else '⚠️ 存在失败项'}")
    return all_pass


if __name__ == "__main__":
    run_all_tests()


# """
# Promptflow工作流测试
# 验证DAG解析、节点执行、变量引用
# """
# import pytest
# import tempfile
# import os
# from pathlib import Path

# from src.promptflow.flow_engine import PromptFlowEngine


# class TestPromptFlowEngine:
#     """测试Promptflow引擎核心功能"""
    
#     @pytest.fixture
#     def temp_flow_dir(self):
#         """创建临时工作流目录"""
#         with tempfile.TemporaryDirectory() as tmpdir:
#             flow_dir = Path(tmpdir) / "test_flow"
#             flow_dir.mkdir()
            
#             # 创建 flow.dag.yaml
#             dag_yaml = flow_dir / "flow.dag.yaml"
#             dag_yaml.write_text("""
# $schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json

# inputs:
#   text:
#     type: string
#     default: "hello"

# outputs:
#   result:
#     type: string
#     reference: ${uppercase.output}

# nodes:
# - name: uppercase
#   type: python
#   source:
#     type: code
#     path: uppercase.py
#   inputs:
#     text: ${inputs.text}
# """, encoding='utf-8')
            
#             # 创建 uppercase.py 节点文件
#             node_file = flow_dir / "uppercase.py"
#             node_file.write_text("""
# def uppercase(text: str) -> str:
#     return text.upper()

# try:
#     from promptflow.core import tool
#     @tool
#     def main(text: str) -> str:
#         return uppercase(text)
# except ImportError:
#     pass
# """, encoding='utf-8')
            
#             yield str(flow_dir)
    
#     def test_load_flow(self, temp_flow_dir):
#         """测试加载工作流定义"""
#         engine = PromptFlowEngine(temp_flow_dir)
#         assert engine.flow_def is not None
#         assert "nodes" in engine.flow_def
#         assert len(engine.flow_def["nodes"]) == 1
#         print("✅ 工作流加载测试通过")
    
#     def test_resolve_reference(self, temp_flow_dir):
#         """测试变量引用解析"""
#         engine = PromptFlowEngine(temp_flow_dir)
#         engine.context = {
#             "inputs": {"text": "hello"},
#             "uppercase": {"output": "HELLO"}
#         }
        
#         # 测试 inputs.text 解析
#         result = engine._resolve_reference("${inputs.text}")
#         assert result == "hello"
        
#         # 测试节点输出解析
#         result = engine._resolve_reference("${uppercase.output}")
#         assert result == "HELLO"
        
#         # 测试非引用字符串原样返回
#         result = engine._resolve_reference("plain_text")
#         assert result == "plain_text"
        
#         print("✅ 变量引用解析测试通过")
    
#     def test_execute_python_node(self, temp_flow_dir):
#         """测试Python节点执行"""
#         engine = PromptFlowEngine(temp_flow_dir)
        
#         node = {
#             "name": "uppercase",
#             "type": "python",
#             "source": {"type": "code", "path": "uppercase.py"},
#             "inputs": {"text": "${inputs.text}"}
#         }
        
#         engine.context = {"inputs": {"text": "hello"}}
#         resolved = engine._resolve_inputs(node["inputs"])
#         result = engine._execute_python_node(node, resolved)
        
#         assert result == "HELLO"
#         print("✅ Python节点执行测试通过")
    
#     def test_run_complete_flow(self, temp_flow_dir):
#         """测试完整工作流执行"""
#         engine = PromptFlowEngine(temp_flow_dir)
#         result = engine.run({"text": "hello"})
        
#         assert result["result"] == "HELLO"
#         print("✅ 完整工作流执行测试通过")
    
#     def test_node_file_not_found(self, temp_flow_dir):
#         """测试节点文件不存在时的错误提示"""
#         # 创建指向不存在文件的工作流
#         bad_flow_dir = Path(temp_flow_dir).parent / "bad_flow"
#         bad_flow_dir.mkdir()
        
#         dag_yaml = bad_flow_dir / "flow.dag.yaml"
#         dag_yaml.write_text("""
# $schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json
# inputs:
#   text: {type: string, default: "test"}
# outputs:
#   result: {type: string, reference: ${bad_node.output}}
# nodes:
# - name: bad_node
#   type: python
#   source:
#     type: code
#     path: nonexistent.py
#   inputs:
#     text: ${inputs.text}
# """, encoding='utf-8')
        
#         engine = PromptFlowEngine(str(bad_flow_dir))
        
#         with pytest.raises(FileNotFoundError) as exc_info:
#             engine.run({"text": "test"})
        
#         error_msg = str(exc_info.value)
#         assert "节点文件不存在" in error_msg
#         assert "nonexistent.py" in error_msg
#         assert "提示: 在 flows/ 目录下创建 nodes/ 子目录" in error_msg
#         print("✅ 节点文件不存在错误提示测试通过")


# class TestPromptFlowNodes:
#     """测试实际业务节点代理文件"""
    
#     def test_detect_intent_node_import(self):
#         """测试意图识别节点可导入"""
#         import sys
#         from pathlib import Path
        
#         # 模拟 flows/nodes/detect_intent.py 的导入逻辑
#         project_root = Path(__file__).parent.parent
#         sys.path.insert(0, str(project_root))
        
#         # 验证 src.agent.intent_classifier 可导入
#         from src.agent.intent_classifier import classify_intent
#         result = classify_intent("这个多少钱")
        
#         assert "intent" in result
#         assert result["intent"] == "price_inquiry"
#         print(f"✅ 意图识别节点导入测试通过: {result['intent']}")
    
#     def test_retrieve_knowledge_node_import(self):
#         """测试知识检索节点可导入"""
#         import sys
#         from pathlib import Path
        
#         project_root = Path(__file__).parent.parent
#         sys.path.insert(0, str(project_root))
        
#         from src.rag.llama_index_rag import retrieve
#         # retrieve 函数存在即可，实际调用需要Milvus连接
#         assert callable(retrieve)
#         print("✅ 知识检索节点导入测试通过")
    
#     def test_generate_response_node_import(self):
#         """测试LLM生成节点可导入"""
#         import sys
#         from pathlib import Path
        
#         project_root = Path(__file__).parent.parent
#         sys.path.insert(0, str(project_root))
        
#         from src.models.llm_client import LLMClient
#         client = LLMClient()
#         assert client.url == "http://localhost:8000/v1"
#         print("✅ LLM生成节点导入测试通过")


# class TestPromptFlowVariableResolution:
#     """测试复杂变量引用场景"""
    
#     def test_nested_dict_reference(self):
#         """测试嵌套字典引用"""
#         from src.promptflow.flow_engine import PromptFlowEngine
        
#         engine = PromptFlowEngine.__new__(PromptFlowEngine)
#         engine.context = {
#             "detect_intent": {
#                 "intent": "price_inquiry",
#                 "confidence": 0.95,
#                 "method": "rule"
#             }
#         }
        
#         # 测试嵌套属性访问
#         result = engine._resolve_reference("${detect_intent.intent}")
#         assert result == "price_inquiry"
        
#         result = engine._resolve_reference("${detect_intent.confidence}")
#         assert result == 0.95
        
#         print("✅ 嵌套字典引用测试通过")
    
#     def test_list_reference(self):
#         """测试列表类型引用"""
#         from src.promptflow.flow_engine import PromptFlowEngine
        
#         engine = PromptFlowEngine.__new__(PromptFlowEngine)
#         engine.context = {
#             "retrieve_knowledge": [
#                 {"id": "1", "score": 0.9},
#                 {"id": "2", "score": 0.8}
#             ]
#         }
        
#         result = engine._resolve_reference("${retrieve_knowledge}")
#         assert isinstance(result, list)
#         assert len(result) == 2
        
#         print("✅ 列表类型引用测试通过")


# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "-s"])