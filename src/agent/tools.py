"""
Agent工具集
- Python函数实现的工具调用
- 支持：查询订单、转人工、发送邮件、查询库存等

【演示实现】：模拟数据返回
【生产目标】：对接真实业务系统API
"""

import logging
import random
from typing import Dict, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        self.register("query_order", query_order)
        self.register("transfer_to_human", transfer_to_human)
        self.register("check_inventory", check_inventory)
        self.register("send_email", send_email)
        self.register("create_ticket", create_ticket)
    
    def register(self, name: str, func):
        """注册工具"""
        self.tools[name] = func
        logger.info(f"🔧 注册工具: {name}")
    
    def call(self, name: str, **kwargs) -> Dict[str, Any]:
        """调用工具"""
        if name not in self.tools:
            return {
                "success": False,
                "error": f"工具不存在: {name}",
                "data": None
            }
        
        try:
            result = self.tools[name](**kwargs)
            return {
                "success": True,
                "data": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    def list_tools(self) -> list:
        """列出所有工具"""
        return list(self.tools.keys())


# ============ 具体工具实现 ============

def query_order(order_id: Optional[str] = None, phone: Optional[str] = None) -> Dict:
    """
    查询订单信息
    
    【演示实现】：返回模拟数据
    """
    logger.info(f"🔍 查询订单: order_id={order_id}, phone={phone}")
    
    # 模拟数据
    mock_orders = {
        "ORD2024001": {
            "order_id": "ORD2024001",
            "status": "已发货",
            "product": "铅酸蓄电池 12V 100Ah",
            "amount": 599.00,
            "create_time": "2024-01-15",
            "tracking": "SF1234567890"
        },
        "ORD2024002": {
            "order_id": "ORD2024002",
            "status": "待付款",
            "product": "锂电池 48V 20Ah",
            "amount": 1299.00,
            "create_time": "2024-01-20",
            "tracking": None
        }
    }
    
    if order_id and order_id in mock_orders:
        return mock_orders[order_id]
    
    # 模糊匹配
    if phone:
        return {
            "orders": [
                {"order_id": "ORD2024001", "status": "已发货", "amount": 599.00},
                {"order_id": "ORD2024003", "status": "已完成", "amount": 299.00}
            ],
            "total": 2
        }
    
    return {"error": "未找到订单，请提供订单号或手机号"}


def transfer_to_human(reason: str = "", priority: str = "normal") -> Dict:
    """
    转接人工客服
    
    【演示实现】：记录转接请求
    """
    logger.info(f"👤 转接人工: reason={reason}, priority={priority}")
    
    ticket_id = f"TKT{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(100,999)}"
    
    return {
        "ticket_id": ticket_id,
        "status": "queued",
        "estimated_wait": "3-5分钟",
        "queue_position": random.randint(1, 10),
        "message": f"已为您创建工单 {ticket_id}，预计等待3-5分钟，请保持在线。"
    }


def check_inventory(product_name: str) -> Dict:
    """
    查询库存
    
    【演示实现】：模拟库存数据
    """
    logger.info(f"📦 查询库存: {product_name}")
    
    mock_inventory = {
        "铅酸蓄电池": {"stock": 156, "warehouse": "上海仓", "restock_date": "2024-02-01"},
        "锂电池": {"stock": 23, "warehouse": "深圳仓", "restock_date": "2024-01-25"},
        "充电器": {"stock": 500, "warehouse": "上海仓", "restock_date": None}
    }
    
    # 模糊匹配
    for key, value in mock_inventory.items():
        if key in product_name or product_name in key:
            return {
                "product": key,
                **value,
                "available": value["stock"] > 0
            }
    
    return {"error": f"未找到产品: {product_name}"}


def send_email(to: str, subject: str, content: str) -> Dict:
    """
    发送邮件
    
    【演示实现】：模拟发送
    """
    logger.info(f"📧 发送邮件: to={to}, subject={subject}")
    
    return {
        "message_id": f"MSG{random.randint(100000, 999999)}",
        "status": "sent",
        "to": to,
        "sent_at": datetime.now().isoformat()
    }


def create_ticket(
    user_id: str,
    category: str,
    description: str,
    priority: str = "normal"
) -> Dict:
    """
    创建工单
    
    【演示实现】：模拟工单系统
    """
    logger.info(f"🎫 创建工单: user={user_id}, category={category}")
    
    ticket_id = f"TKT{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return {
        "ticket_id": ticket_id,
        "status": "open",
        "category": category,
        "priority": priority,
        "created_at": datetime.now().isoformat(),
        "message": f"工单 {ticket_id} 已创建，我们将尽快处理。"
    }


# 全局注册表
_tool_registry: Optional[ToolRegistry] = None

def get_tool_registry() -> ToolRegistry:
    """获取工具注册表单例"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def run_tool(name: str, **kwargs) -> Dict[str, Any]:
    """便捷调用工具"""
    registry = get_tool_registry()
    return registry.call(name, **kwargs)