"""
vLLM API客户端 - 增强版
支持自动检测端点、双模式回退(chat/completions)、自动获取模型名
"""
import requests
import json
from typing import Optional, Dict, List
from src.config import MODELS


class LLMClient:
    """vLLM OpenAI兼容API客户端 - 增强版"""
    
    def __init__(self):
        self.url = MODELS["llm"]["url"]
        self.api_key = MODELS["llm"]["api_key"]
        self.model = MODELS["llm"]["name"]
        self.max_tokens = MODELS["llm"]["max_tokens"]
        self.temperature = MODELS["llm"]["temperature"]
        
        # 自动检测 vLLM 能力
        self._detect_capabilities()
    
    def _detect_capabilities(self):
        """自动检测 vLLM 支持的端点和模型"""
        self.chat_completions_available = False
        self.completions_available = False
        self.available_models = []
        self.actual_model_name = self.model
        
        try:
            # 检测 /v1/models
            resp = requests.get(
                f"{self.url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            if resp.status_code == 200:
                models_data = resp.json().get("data", [])
                self.available_models = [m["id"] for m in models_data]
                print(f"[LLMClient] vLLM可用模型: {self.available_models}")
                
                # 自动匹配模型名
                if self.available_models:
                    if self.model not in self.available_models:
                        self.actual_model_name = self.available_models[0]
                        print(f"[LLMClient] 模型名不匹配，自动切换: {self.model} -> {self.actual_model_name}")
            
            # 检测 /v1/chat/completions 可用性
            test_resp = requests.post(
                f"{self.url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.actual_model_name,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1
                },
                timeout=5
            )
            self.chat_completions_available = test_resp.status_code in [200, 400]
            print(f"[LLMClient] chat/completions 可用: {self.chat_completions_available}")
            
            # 检测 /v1/completions 可用性
            test_resp2 = requests.post(
                f"{self.url}/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.actual_model_name,
                    "prompt": "test",
                    "max_tokens": 1
                },
                timeout=5
            )
            self.completions_available = test_resp2.status_code in [200, 400]
            print(f"[LLMClient] completions 可用: {self.completions_available}")
            
        except Exception as e:
            print(f"[LLMClient] 检测vLLM能力失败: {e}")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        调用vLLM生成文本
        优先使用 chat/completions，不可用时回退到 completions
        """
        if self.chat_completions_available:
            return self._generate_chat(prompt, max_tokens, temperature, system_prompt)
        elif self.completions_available:
            return self._generate_completion(prompt, max_tokens, temperature)
        else:
            return "[错误] vLLM 未运行或不支持任何已知端点"
    
    def _generate_chat(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """使用 /v1/chat/completions 生成"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.actual_model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.chat_completions_available = False
                return self._generate_completion(prompt, max_tokens, temperature)
            return f"[错误] vLLM HTTP错误: {e.response.status_code} - {e.response.text[:200]}"
        except requests.exceptions.ConnectionError:
            return "[错误] 无法连接到vLLM服务，请确认Docker容器正在运行（端口8000）"
        except requests.exceptions.Timeout:
            return "[错误] vLLM响应超时，请检查服务状态"
        except Exception as e:
            return f"[错误] LLM调用失败: {str(e)}"
    
    def _generate_completion(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """使用 /v1/completions 生成（回退模式）"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.actual_model_name,
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.url}/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["text"]
        
        except requests.exceptions.HTTPError as e:
            return f"[错误] vLLM HTTP错误: {e.response.status_code} - {e.response.text[:200]}"
        except requests.exceptions.ConnectionError:
            return "[错误] 无法连接到vLLM服务"
        except requests.exceptions.Timeout:
            return "[错误] vLLM响应超时"
        except Exception as e:
            return f"[错误] LLM调用失败: {str(e)}"
    
    def generate_stream(self, prompt: str, **kwargs):
        """流式生成（SSE推送）"""
        if not self.chat_completions_available:
            yield "[错误] 流式生成需要 chat/completions 端点"
            return
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.actual_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True
        }
        
        response = requests.post(
            f"{self.url}/chat/completions",
            headers=headers,
            json=data,
            stream=True,
            timeout=60
        )
        
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except:
                        pass
    
    def health_check(self) -> Dict:
        """检查vLLM服务健康状态"""
        try:
            response = requests.get(
                f"{self.url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                return {
                    "status": "healthy",
                    "models": [m["id"] for m in models],
                    "url": self.url,
                    "chat_completions": self.chat_completions_available,
                    "completions": self.completions_available,
                    "actual_model": self.actual_model_name
                }
            return {"status": "unhealthy", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}