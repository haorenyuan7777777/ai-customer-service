"""
Agent端到端测试
- 意图识别准确率
- 多轮记忆连贯性
- 工具调用正确性
- 完整对话链路
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.agent.agent_core import get_agent, CustomerServiceAgent
from src.agent.intent_classifier import get_intent_classifier
from src.agent.memory import get_memory_store
from src.agent.tools import get_tool_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_intent_classification():
    """测试意图识别"""
    logger.info("=" * 50)
    logger.info("【测试1】意图识别")
    
    clf = get_intent_classifier()
    
    test_cases = [
        ("这个电池多少钱？", "销售转化"),
        ("电池充不进去电怎么办？", "技术支持"),
        ("你们这是骗人的，我要投诉！", "投诉处理"),
        ("铅酸蓄电池正确使用的注意事项有哪些？", "标准客服"),
        ("有优惠活动吗？想批量购买", "销售转化"),
        ("产品坏了，怎么修？", "技术支持"),
    ]
    
    correct = 0
    for msg, expected in test_cases:
        result = clf.classify(msg)
        actual = result["intent"]
        is_correct = actual == expected
        correct += int(is_correct)
        
        icon = "✅" if is_correct else "❌"
        logger.info(f"{icon} 输入: {msg[:20]}... → 预期: {expected}, 实际: {actual}")
    
    accuracy = correct / len(test_cases)
    logger.info(f"准确率: {accuracy:.1%} ({correct}/{len(test_cases)})")
    assert accuracy >= 0.8, f"意图识别准确率过低: {accuracy}"
    logger.info("✅ 意图识别测试通过")
    return accuracy


def test_multi_turn_memory():
    """测试多轮记忆"""
    logger.info("\n" + "=" * 50)
    logger.info("【测试2】多轮记忆")
    
    agent = get_agent()
    user_id = "test_user_memory"
    session_id = "session_001"
    
    # 第一轮
    r1 = agent.chat("你好，我想买电池", user_id, session_id)
    logger.info(f"轮1: {r1['response'][:50]}...")
    
    # 第二轮（应包含第一轮记忆）
    r2 = agent.chat("刚才说的那个多少钱？", user_id, session_id)
    logger.info(f"轮2: {r2['response'][:50]}...")
    
    # 验证记忆
    memory = agent.memory
    history  = memory.get_history(user_id, session_id, limit=10)
    assert len(history) >= 2, f"记忆轮数不足: {len(history)}"

    logger.info(f"记忆轮数: {len(history)}")
    for entry in history:
        logger.info(f"  [用户] {entry['user_message'][:30]}...")
        logger.info(f"  [助手] {entry['assistant_message'][:30]}...")
        logger.info(f"  [意图] {entry['intent'][:30]}...")
        logger.info(f"  [时间] {entry['timestamp'][:30]}...")
    
    logger.info(f"记忆轮数: {len(history)}")

    memory.clear_history(user_id, session_id)
    logger.info("✅ 多轮记忆测试通过")


def test_tool_execution():
    """测试工具调用"""
    logger.info("\n" + "=" * 50)
    logger.info("【测试3】工具调用")
    
    registry = get_tool_registry()
    
    # 查询订单
    result = registry.call("query_order", order_id="ORD2024001")
    assert result["success"]
    assert result["data"]["order_id"] == "ORD2024001"
    logger.info(f"查询订单: {result['data']}")
    
    # 转人工
    result = registry.call("transfer_to_human", reason="复杂问题")
    assert result["success"]
    assert "ticket_id" in result["data"]
    logger.info(f"转人工: {result['data']['ticket_id']}")
    
    # 查询库存
    result = registry.call("check_inventory", product_name="铅酸蓄电池")
    assert result["success"]
    assert result["data"]["stock"] > 0
    logger.info(f"查询库存: {result['data']['stock']}件")
    
    logger.info("✅ 工具调用测试通过")


def test_full_pipeline():
    """测试完整对话链路"""
    logger.info("\n" + "=" * 50)
    logger.info("【测试4】完整对话链路")
    
    agent = get_agent()
    
    result = agent.chat(
        "这个铅酸蓄电池多少钱？有优惠吗？",
        user_id="test_user_pipeline",
        session_id="pipeline_001"
    )
    
    logger.info(f"意图: {result.get('intent')}")
    logger.info(f"分类: {result.get('category')}")
    response = result.get('response', '') or ""
    logger.info(f"响应: {response[:100]}...")
    
    # 获取耗时（兼容 Promptflow 和直接调用）
    elapsed = result.get('execution_time_ms', 0)
    logger.info(f"耗时: {elapsed}ms")
    
    assert result.get("intent") == "销售转化"
    assert len(response) > 10
    # 放宽时间检查，生产环境可调整
    assert elapsed < 30000, f"响应过慢: {elapsed}ms"  # 30秒兜底
    
    logger.info("✅ 完整链路测试通过")

def test_complaint_priority():
    """测试投诉高优先级"""
    logger.info("\n" + "=" * 50)
    logger.info("【测试5】投诉高优先级")
    
    agent = get_agent()
    
    result = agent.chat(
        "你们产品质量太差了，我要退款！",
        user_id="test_user_complaint",
        session_id="complaint_001"
    )
    
    assert result["intent"] == "投诉处理"
    # 投诉响应应包含安抚性语言
    response = result["response"]
    assert any(word in response for word in ["抱歉", "理解", "歉意", "记录"]), \
        f"投诉响应缺少安抚: {response[:50]}"
    
    logger.info(f"投诉响应: {response[:100]}...")
    logger.info("✅ 投诉优先级测试通过")


def run_all_tests():
    """运行全部测试"""
    tests = [
        test_intent_classification,
        test_multi_turn_memory,
        test_tool_execution,
        test_full_pipeline,
        test_complaint_priority,
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
    logger.info("📋 Agent测试报告")
    logger.info("=" * 50)
    for name, status in results:
        icon = "✅" if status == "PASS" else "❌"
        logger.info(f"{icon} {name}: {status}")
    
    all_pass = all(s == "PASS" for _, s in results)
    logger.info(f"\n总评: {'✅ 全部通过' if all_pass else '⚠️ 存在失败项'}")
    return all_pass


if __name__ == "__main__":
    run_all_tests()