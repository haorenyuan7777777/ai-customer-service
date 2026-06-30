"""
Promptflow模拟执行引擎
- 解析YAML格式的DAG定义
- 按拓扑序执行节点
- 支持节点缓存、错误回退、执行日志
- 与LlamaIndex RAG、Agent记忆集成

【演示实现】：Python顺序执行，模拟Azure Promptflow
【生产目标】：迁移至Azure AI Studio
"""

import os
import yaml
import logging
import traceback
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from collections import deque
import importlib.util
import time

from jinja2 import Environment, FileSystemLoader, select_autoescape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NodeResult:
    """节点执行结果"""
    node_name: str
    status: str  # "success", "error", "cached"
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class PromptFlowEngine:
    """
    Promptflow执行引擎
    
    功能：
    1. 加载YAML流程定义
    2. 解析DAG拓扑序
    3. 执行Python/LLM节点
    4. 节点级缓存（避免重复执行）
    5. 错误回退（节点失败→默认输出）
    """
    
    def __init__(
        self,
        dag_path: str = "flows/flow.dag.yaml",
        prompts_path: str = "configs/prompts.yaml",
        templates_dir: str = "flows/templates"
    ):
        self.dag_path = dag_path
        self.prompts_path = prompts_path
        self.templates_dir = templates_dir
        
        # 加载配置
        self.dag = self._load_dag()
        self.prompts = self._load_prompts()
        
        # Jinja2模板引擎
        self.jinja_env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(['html', 'xml', 'jinja2']),
            enable_async=False
        )
        
        # 节点缓存（同session内避免重复执行）
        self.node_cache: Dict[str, Any] = {}
        
        # 节点函数映射（动态加载）
        self.node_functions: Dict[str, Callable] = {}
        self._load_node_functions()
        
        logger.info(f"✅ Promptflow引擎初始化完成: {dag_path}")
    
    def _load_dag(self) -> Dict[str, Any]:
        """加载DAG定义"""
        with open(self.dag_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_prompts(self) -> Dict[str, Any]:
        """加载提示词配置"""
        if not os.path.exists(self.prompts_path):
            logger.warning(f"提示词文件不存在: {self.prompts_path}，使用空配置")
            return {}
        
        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    
    def _load_node_functions(self):
        """动态加载flows/nodes/下的Python节点"""
        nodes_dir = os.path.join(os.path.dirname(self.dag_path), "nodes")
        if not os.path.exists(nodes_dir):
            logger.warning(f"节点目录不存在: {nodes_dir}")
            return
        
        for filename in os.listdir(nodes_dir):
            if not filename.endswith('.py') or filename.startswith('_'):
                continue
            
            node_name = filename[:-3]  # 去掉.py
            filepath = os.path.join(nodes_dir, filename)
            
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(f"flow_node.{node_name}", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找run函数
            if hasattr(module, 'run'):
                self.node_functions[node_name] = module.run
                logger.info(f"  加载节点: {node_name}")
    
    def _resolve_inputs(self, node_inputs: Dict, context: Dict) -> Dict:
        """
        解析节点输入（支持${引用}）
        """
        resolved = {}
        for key, value in node_inputs.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                path = value[2:-1].strip()
                parts = path.split('.')
                
                current = context
                for part in parts:
                    if current is None:
                        logger.warning(f"引用路径中断: {value}，当前值为None")
                        current = ""
                        break
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        logger.warning(f"无法解析引用: {value}，使用空值")
                        current = ""
                        break
                resolved[key] = current
            else:
                resolved[key] = value
        return resolved
    
    def _get_execution_order(self) -> List[str]:
        """
        拓扑排序获取节点执行顺序
        确保依赖节点先执行
        """
        nodes = {n['name']: n for n in self.dag.get('nodes', [])}
        inputs = set(self.dag.get('inputs', {}).keys())
        
        # 构建依赖图
        in_degree = {name: 0 for name in nodes}
        dependents = {name: [] for name in nodes}
        
        for name, node in nodes.items():
            for input_val in node.get('inputs', {}).values():
                if isinstance(input_val, str) and input_val.startswith('${'):
                    # 提取依赖节点名
                    ref = input_val[2:-1].split('.')[0]
                    if ref in nodes and ref != name:
                        in_degree[name] += 1
                        dependents[ref].append(name)
        
        # 拓扑排序
        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []
        
        while queue:
            current = queue.popleft()
            order.append(current)
            for dep in dependents[current]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
        
        if len(order) != len(nodes):
            raise ValueError("DAG存在循环依赖，无法执行")
        
        return order
    
    def _execute_node(self, node: Dict, inputs: Dict, context: Dict) -> NodeResult:
        node_name = node['name']
        node_type = node.get('type', 'python')
        
        cache_key = f"{node_name}:{hash(str(inputs))}"
        if cache_key in self.node_cache:
            return NodeResult(
                node_name=node_name,
                status="cached",
                output=self.node_cache[cache_key],
                execution_time_ms=0
            )
        
        start_time = time.perf_counter()
        
        try:
            if node_type == 'python':
                func = self.node_functions.get(node_name)
                if func is None:
                    raise ValueError(f"未找到节点函数: {node_name}")
                
                output = func(**inputs)
                # 【修复】防御 None
                if output is None:
                    output = {}
                    logger.warning(f"节点 {node_name} 返回None，使用空字典")
                
            elif node_type == 'llm':
                from src.models.llm_client import get_llm_client
                llm = get_llm_client()
                
                prompt = inputs.get('prompt', '')
                max_tokens = inputs.get('max_tokens', 512)
                
                output = llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                if output is None:
                    output = ""
                    
            else:
                raise ValueError(f"未知节点类型: {node_type}")
            
            self.node_cache[cache_key] = output
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return NodeResult(
                node_name=node_name,
                status="success",
                output=output,
                execution_time_ms=round(execution_time, 2)
            )
            
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"❌ 节点执行失败 [{node_name}]: {error_msg}")
            
            fallback = self._get_fallback_output(node_name, node_type)
            
            return NodeResult(
                node_name=node_name,
                status="error",
                output=fallback,
                error=error_msg,
                execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2)
            )

    def _get_fallback_output(self, node_name: str, node_type: str) -> Any:
        """节点失败时的默认回退输出"""
        fallbacks = {
            'detect_intent': {
                'intent': '标准客服',
                'category': 'general',
                'priority': 4,
                'keywords': [],
                'confidence': 1.0
            },
            'load_memory': '',
            'retrieve_knowledge': [],
            'assemble_prompt': '请回答用户问题。',
            'generate_response': '抱歉，系统暂时无法处理您的请求，请稍后重试。',
            'save_memory': None
        }
        return fallbacks.get(node_name, None)
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行完整流程
        
        Args:
            inputs: 流程输入，如 {"user_message": "...", "user_id": "..."}
        
        Returns:
            流程输出，包含response、intent、context等
        """
        logger.info(f"🚀 开始执行流程，输入: {inputs}")
        
        # 初始化上下文
        context = {
            "inputs": inputs,
            "flow_outputs": {}
        }
        
        # 获取执行顺序
        execution_order = self._get_execution_order()
        logger.info(f"📋 执行顺序: {' → '.join(execution_order)}")
        
        # 执行节点
        results = {}
        for node_name in execution_order:
            node = next(n for n in self.dag['nodes'] if n['name'] == node_name)
            
            # 解析输入
            resolved_inputs = self._resolve_inputs(node.get('inputs', {}), context)
            logger.info(f"  ▶️ 执行节点: {node_name}, 输入: {list(resolved_inputs.keys())}")
            
            # 执行
            result = self._execute_node(node, resolved_inputs, context)
            results[node_name] = result
            
            # 更新上下文
            context[node_name] = result.output if result.status != "error" else result.output
            context["flow_outputs"][node_name] = result.output
            
            logger.info(f"  ✅ 节点完成: {node_name} [{result.status}] ({result.execution_time_ms}ms)")
        
        # 组装最终输出
        outputs = self.dag.get('outputs', {})
        final_output = {}
        for out_key, out_ref in outputs.items():
            if isinstance(out_ref, str) and out_ref.startswith('${'):
                path = out_ref[2:-1].strip().split('.')
                current = context
                for part in path:
                    current = current.get(part, {}) if isinstance(current, dict) else {}
                final_output[out_key] = current if current != {} else None
            else:
                final_output[out_key] = out_ref
        
        # 计算总耗时
        total_time = sum(
            r.execution_time_ms 
            for r in results.values() 
            if r.status != "cached"
        )
        
        # 添加执行元信息
        final_output['_execution_meta'] = {
            'nodes_executed': len(execution_order),
            'total_time_ms': round(total_time, 2),
            'node_results': {
                name: {
                    'status': r.status,
                    'time_ms': r.execution_time_ms,
                    'has_error': r.error is not None
                }
                for name, r in results.items()
            }
        }
        
        # 兼容上层对 execution_time_ms 的期望
        final_output['execution_time_ms'] = round(total_time, 2)
        
        logger.info(f"✅ 流程执行完成，输出keys: {list(final_output.keys())}")
        return final_output
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        渲染Jinja2模板
        """
        # 先尝试从文件加载
        try:
            template = self.jinja_env.get_template(f"{template_name}.jinja2")
            # 【修复】传入 system_prompts
            render_ctx = {
                **kwargs,
                "prompts": self.prompts or {},
                "system_prompts": (self.prompts or {}).get("system_prompts", {})
            }
            return template.render(**render_ctx)
        except Exception:
            pass
        
        # 从YAML配置加载
        if self.prompts and isinstance(self.prompts, dict):
            template_config = self.prompts.get('system_prompts', {}).get(template_name)
            if template_config:
                jinja_template = self.jinja_env.from_string(template_config)
                render_ctx = {
                    **kwargs,
                    "prompts": self.prompts,
                    "system_prompts": self.prompts.get("system_prompts", {})
                }
                return jinja_template.render(**render_ctx)
        
        # 最终回退
        logger.warning(f"模板未找到: {template_name}，返回默认提示词")
        return f"请回答用户问题。上下文: {kwargs}"


# 单例
_flow_engine: Optional[PromptFlowEngine] = None

def get_flow_engine() -> PromptFlowEngine:
    """获取Promptflow引擎单例"""
    global _flow_engine
    if _flow_engine is None:
        _flow_engine = PromptFlowEngine()
    return _flow_engine


# """
# Promptflow模拟层
# - 严格对齐Azure Promptflow flow.dag.yaml规范
# - Python解析.yaml，顺序执行节点
# - 支持inputs/outputs/nodes/node_variants
# """
# import yaml
# import jinja2
# from pathlib import Path
# from typing import Dict, Any, Callable
# import importlib


# class PromptFlowEngine:
#     """
#     Promptflow DAG执行引擎（模拟层）
    
#     对齐Azure Promptflow规范：
#     - flow.dag.yaml 格式
#     - ${inputs.xxx} / ${node_name.output} 变量引用
#     - prompt/llm/python 节点类型
#     """
    
#     def __init__(self, flow_dir: str):
#         self.flow_dir = Path(flow_dir)
#         self.flow_file = self.flow_dir / "flow.dag.yaml"
#         self.flow_def = None
#         self.nodes = {}
#         self.context = {}  # 运行时上下文
        
#         self._load_flow()
    
#     def _load_flow(self):
#         with open(self.flow_file, 'r', encoding='utf-8') as f:
#             self.flow_def = yaml.safe_load(f)
#         print(f"[Promptflow] 加载工作流: {self.flow_file}")
    
#     def _resolve_reference(self, ref: str) -> Any:
#         """解析变量引用，如 ${inputs.text} 或 ${node_name.output}"""
#         if not isinstance(ref, str) or not ref.startswith("${") or not ref.endswith("}"):
#             return ref
        
#         path = ref[2:-1].strip()  # 去掉 ${ 和 }
#         parts = path.split(".")
        
#         if parts[0] == "inputs":
#             return self.context.get("inputs", {}).get(parts[1])
#         elif parts[0] in self.context:
#             node_output = self.context[parts[0]]
#             if isinstance(node_output, dict) and len(parts) > 1:
#                 return node_output.get(parts[1])
#             return node_output
#         return ref
    
#     def _resolve_inputs(self, inputs_def: Dict) -> Dict:
#         """解析节点输入定义"""
#         resolved = {}
#         for key, value in inputs_def.items():
#             resolved[key] = self._resolve_reference(value)
#         return resolved
    
#     def _execute_prompt_node(self, node: Dict, inputs: Dict) -> str:
#         """执行prompt节点（Jinja2模板渲染）"""
#         template_path = self.flow_dir / node["source"]["path"]
#         with open(template_path, 'r', encoding='utf-8') as f:
#             template_str = f.read()
        
#         template = jinja2.Template(template_str)
#         return template.render(**inputs)
    
#     def _execute_python_node(self, node: Dict, inputs: Dict) -> Any:
#         """执行python节点"""
#         # source.path 是相对于 flow.dag.yaml 所在目录的路径
#         module_path = self.flow_dir / node["source"]["path"]
        
#         if not module_path.exists():
#             raise FileNotFoundError(
#                 f"节点文件不存在: {module_path}\n"
#                 f"请确认 source.path \"{node['source']['path']}\" 是相对于 flow.dag.yaml 的正确路径\n"
#                 f"flow.dag.yaml 位置: {self.flow_file}\n"
#                 f"提示: 在 flows/ 目录下创建 nodes/ 子目录存放代理节点文件"
#             )
        
#         # 动态加载Python模块
#         spec = importlib.util.spec_from_file_location(
#             f"flow_node_{node['name']}", 
#             module_path
#         )
#         module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(module)
        
#         # 查找入口函数（优先使用tool装饰器或source中指定的tool）
#         tool_name = node["source"].get("tool")
        
#         if tool_name and hasattr(module, tool_name):
#             func = getattr(module, tool_name)
#         else:
#             # 尝试常见入口函数名
#             for candidate in ["main", "detect_intent", "load_memory", 
#                               "retrieve_knowledge", "generate_response"]:
#                 if hasattr(module, candidate):
#                     func = getattr(module, candidate)
#                     break
#             else:
#                 # 查找第一个非私有函数
#                 for attr_name in dir(module):
#                     attr = getattr(module, attr_name)
#                     if callable(attr) and not attr_name.startswith("_"):
#                         func = attr
#                         break
#                 else:
#                     raise ValueError(f"节点 {node['name']} 未找到可调用函数")
        
#         return func(**inputs)
    
#     def _execute_llm_node(self, node: Dict, inputs: Dict) -> str:
#         """执行llm节点（调用vLLM）"""
#         from src.models.llm_client import LLMClient
#         client = LLMClient()
        
#         prompt = inputs.get("prompt", "")
#         return client.generate(prompt, max_tokens=inputs.get("max_tokens", 512))
    
#     def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         执行完整工作流
        
#         Args:
#             inputs: 工作流输入字典
        
#         Returns:
#             工作流输出字典
#         """
#         self.context = {"inputs": inputs}
        
#         # 按顺序执行节点
#         for node in self.flow_def.get("nodes", []):
#             node_name = node["name"]
#             node_type = node["type"]
            
#             print(f"[Promptflow] 执行节点: {node_name} ({node_type})")
            
#             # 解析输入
#             resolved_inputs = self._resolve_inputs(node.get("inputs", {}))
            
#             # 执行节点
#             if node_type == "prompt":
#                 output = self._execute_prompt_node(node, resolved_inputs)
#             elif node_type == "python":
#                 output = self._execute_python_node(node, resolved_inputs)
#             elif node_type == "llm":
#                 output = self._execute_llm_node(node, resolved_inputs)
#             else:
#                 raise ValueError(f"未知节点类型: {node_type}")
            
#             self.context[node_name] = output
#             print(f"[Promptflow] 节点 {node_name} 完成")
        
#         # 构建输出
#         outputs = {}
#         for out_name, out_def in self.flow_def.get("outputs", {}).items():
#             ref = out_def.get("reference")
#             outputs[out_name] = self._resolve_reference(ref)
        
#         return outputs
    
# if __name__ == '__main__':
#     flow = PromptFlowEngine("flows")
#     result = flow.run({
#     "query": "铅酸蓄电池正确使用的注意事项有哪些？",
#         "session_id": "test_001"
#     })
#     print(f"工作流输出: {result}")