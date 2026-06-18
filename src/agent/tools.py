"""
工具调用注册中心
支持Python函数注册、参数验证、执行
"""
import inspect
from typing import Dict, List, Callable, Any
from functools import wraps


class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self._tools: Dict[str, Dict] = {}
    
    def register(self, name: str = None, description: str = ""):
        """
        工具注册装饰器
        
        Args:
            name: 工具名称，默认使用函数名
            description: 工具描述
        
        Example:
            @registry.register(name="get_price", description="查询价格")
            def get_price(product: str) -> dict:
                return {"price": 199.99}
        """
        def decorator(func: Callable):
            tool_name = name or func.__name__
            
            # 获取函数签名
            sig = inspect.signature(func)
            params = []
            for param_name, param in sig.parameters.items():
                param_info = {
                    "name": param_name,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "any",
                    "required": param.default == inspect.Parameter.empty
                }
                params.append(param_info)
            
            self._tools[tool_name] = {
                "name": tool_name,
                "description": description or func.__doc__ or "",
                "function": func,
                "parameters": params
            }
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def list_tools(self) -> List[Dict]:
        """列出所有可用工具"""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
            for t in self._tools.values()
        ]
    
    def get_tool(self, name: str) -> Dict:
        """获取工具信息"""
        if name not in self._tools:
            raise ValueError(f"工具 '{name}' 不存在。可用工具: {list(self._tools.keys())}")
        return self._tools[name]
    
    def execute(self, name: str, params: Dict[str, Any]) -> Any:
        """
        执行工具
        
        Args:
            name: 工具名称
            params: 参数字典
        
        Returns:
            工具执行结果
        """
        tool = self.get_tool(name)
        func = tool["function"]
        
        try:
            result = func(**params)
            return {
                "status": "success",
                "tool": name,
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "tool": name,
                "error": str(e)
            }
    
    def execute_by_intent(self, intent: str, context: Dict) -> Dict:
        """
        根据意图自动选择并执行工具
        
        Args:
            intent: 意图标签
            context: 上下文信息
        """
        intent_tool_map = {
            "price_inquiry": "get_price",
            "purchase_intent": "create_order",
            "technical_issue": "troubleshoot",
        }
        
        tool_name = intent_tool_map.get(intent)
        if tool_name and tool_name in self._tools:
            return self.execute(tool_name, context)
        
        return {
            "status": "no_tool",
            "message": f"意图 '{intent}' 没有对应的工具"
        }


# 创建全局注册中心
tool_registry = ToolRegistry()


# ========== 预定义工具 ==========
@tool_registry.register(name="get_price", description="查询产品价格")
def get_price(product: str) -> dict:
    """查询指定产品的价格"""
    # 演示实现：模拟价格查询
    price_map = {
        "铅酸电池": 299.99,
        "锂电池": 599.99,
        "充电器": 89.99,
    }
    return {
        "product": product,
        "price": price_map.get(product, "暂无报价"),
        "currency": "CNY"
    }


@tool_registry.register(name="check_stock", description="检查产品库存")
def check_stock(product: str) -> dict:
    """检查指定产品的库存状态"""
    import random
    stock = random.randint(0, 500)
    return {
        "product": product,
        "stock": stock,
        "available": stock > 0
    }


@tool_registry.register(name="create_order", description="创建订单")
def create_order(product: str, quantity: int = 1) -> dict:
    """创建购买订单"""
    import uuid
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    return {
        "order_id": order_id,
        "product": product,
        "quantity": quantity,
        "status": "created"
    }


@tool_registry.register(name="troubleshoot", description="故障排查指导")
def troubleshoot(product: str, symptom: str) -> dict:
    """提供故障排查建议"""
    guides = {
        "无法充电": "请检查充电器连接和电池接触点",
        "续航短": "建议进行电池校准或更换老化电池",
        "发热": "立即停止使用，检查是否过充",
    }
    return {
        "product": product,
        "symptom": symptom,
        "guide": guides.get(symptom, "请联系技术支持"),
        "urgent": symptom in ["发热", "冒烟"]
    }