"""
Promptflow模拟层
- 严格对齐Azure Promptflow flow.dag.yaml规范
- Python解析.yaml，顺序执行节点
- 支持inputs/outputs/nodes/node_variants
"""
import yaml
import jinja2
from pathlib import Path
from typing import Dict, Any, Callable
import importlib


class PromptFlowEngine:
    """
    Promptflow DAG执行引擎（模拟层）
    
    对齐Azure Promptflow规范：
    - flow.dag.yaml 格式
    - ${inputs.xxx} / ${node_name.output} 变量引用
    - prompt/llm/python 节点类型
    """
    
    def __init__(self, flow_dir: str):
        self.flow_dir = Path(flow_dir)
        self.flow_file = self.flow_dir / "flow.dag.yaml"
        self.flow_def = None
        self.nodes = {}
        self.context = {}  # 运行时上下文
        
        self._load_flow()
    
    def _load_flow(self):
        with open(self.flow_file, 'r', encoding='utf-8') as f:
            self.flow_def = yaml.safe_load(f)
        print(f"[Promptflow] 加载工作流: {self.flow_file}")
    
    def _resolve_reference(self, ref: str) -> Any:
        """解析变量引用，如 ${inputs.text} 或 ${node_name.output}"""
        if not isinstance(ref, str) or not ref.startswith("${") or not ref.endswith("}"):
            return ref
        
        path = ref[2:-1].strip()  # 去掉 ${ 和 }
        parts = path.split(".")
        
        if parts[0] == "inputs":
            return self.context.get("inputs", {}).get(parts[1])
        elif parts[0] in self.context:
            node_output = self.context[parts[0]]
            if isinstance(node_output, dict) and len(parts) > 1:
                return node_output.get(parts[1])
            return node_output
        return ref
    
    def _resolve_inputs(self, inputs_def: Dict) -> Dict:
        """解析节点输入定义"""
        resolved = {}
        for key, value in inputs_def.items():
            resolved[key] = self._resolve_reference(value)
        return resolved
    
    def _execute_prompt_node(self, node: Dict, inputs: Dict) -> str:
        """执行prompt节点（Jinja2模板渲染）"""
        template_path = self.flow_dir / node["source"]["path"]
        with open(template_path, 'r', encoding='utf-8') as f:
            template_str = f.read()
        
        template = jinja2.Template(template_str)
        return template.render(**inputs)
    
    def _execute_python_node(self, node: Dict, inputs: Dict) -> Any:
        """执行python节点"""
        # source.path 是相对于 flow.dag.yaml 所在目录的路径
        module_path = self.flow_dir / node["source"]["path"]
        
        if not module_path.exists():
            raise FileNotFoundError(
                f"节点文件不存在: {module_path}\n"
                f"请确认 source.path \"{node['source']['path']}\" 是相对于 flow.dag.yaml 的正确路径\n"
                f"flow.dag.yaml 位置: {self.flow_file}\n"
                f"提示: 在 flows/ 目录下创建 nodes/ 子目录存放代理节点文件"
            )
        
        # 动态加载Python模块
        spec = importlib.util.spec_from_file_location(
            f"flow_node_{node['name']}", 
            module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 查找入口函数（优先使用tool装饰器或source中指定的tool）
        tool_name = node["source"].get("tool")
        
        if tool_name and hasattr(module, tool_name):
            func = getattr(module, tool_name)
        else:
            # 尝试常见入口函数名
            for candidate in ["main", "detect_intent", "load_memory", 
                              "retrieve_knowledge", "generate_response"]:
                if hasattr(module, candidate):
                    func = getattr(module, candidate)
                    break
            else:
                # 查找第一个非私有函数
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if callable(attr) and not attr_name.startswith("_"):
                        func = attr
                        break
                else:
                    raise ValueError(f"节点 {node['name']} 未找到可调用函数")
        
        return func(**inputs)
    
    def _execute_llm_node(self, node: Dict, inputs: Dict) -> str:
        """执行llm节点（调用vLLM）"""
        from src.models.llm_client import LLMClient
        client = LLMClient()
        
        prompt = inputs.get("prompt", "")
        return client.generate(prompt, max_tokens=inputs.get("max_tokens", 512))
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行完整工作流
        
        Args:
            inputs: 工作流输入字典
        
        Returns:
            工作流输出字典
        """
        self.context = {"inputs": inputs}
        
        # 按顺序执行节点
        for node in self.flow_def.get("nodes", []):
            node_name = node["name"]
            node_type = node["type"]
            
            print(f"[Promptflow] 执行节点: {node_name} ({node_type})")
            
            # 解析输入
            resolved_inputs = self._resolve_inputs(node.get("inputs", {}))
            
            # 执行节点
            if node_type == "prompt":
                output = self._execute_prompt_node(node, resolved_inputs)
            elif node_type == "python":
                output = self._execute_python_node(node, resolved_inputs)
            elif node_type == "llm":
                output = self._execute_llm_node(node, resolved_inputs)
            else:
                raise ValueError(f"未知节点类型: {node_type}")
            
            self.context[node_name] = output
            print(f"[Promptflow] 节点 {node_name} 完成")
        
        # 构建输出
        outputs = {}
        for out_name, out_def in self.flow_def.get("outputs", {}).items():
            ref = out_def.get("reference")
            outputs[out_name] = self._resolve_reference(ref)
        
        return outputs
    
if __name__ == '__main__':
    flow = PromptFlowEngine("flows")
    result = flow.run({
    "query": "铅酸蓄电池正确使用的注意事项有哪些？",
        "session_id": "test_001"
    })
    print(f"工作流输出: {result}")