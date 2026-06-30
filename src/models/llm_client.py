"""
vLLM API客户端 - 修复版
- 强制重试检测（解决启动时序问题）
- 模型名自动修正（对齐docker-compose）
- 请求时动态检测（不依赖初始化时状态）
"""

import requests
import json
import time
import logging
from typing import Optional, Dict

from src.config import MODELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """vLLM OpenAI兼容API客户端 - 修复版"""
    
    def __init__(self):
        self.url = MODELS["llm"]["url"]
        self.api_key = MODELS["llm"]["api_key"]
        self.model = MODELS["llm"]["name"]
        self.max_tokens = MODELS["llm"]["max_tokens"]
        self.temperature = MODELS["llm"]["temperature"]
        
        # 状态
        self.chat_completions_available = False
        self.completions_available = False
        self.available_models = []
        self.actual_model_name = self.model
        
        # 初始化检测（带重试）
        self._detect_with_retry()
    
    def _detect_with_retry(self, max_retries: int = 3):
        """带重试的vLLM能力检测"""
        for attempt in range(max_retries):
            try:
                self._detect_capabilities()
                if self.chat_completions_available or self.completions_available:
                    logger.info(f"✅ vLLM检测成功（第{attempt+1}次尝试）")
                    return
            except Exception as e:
                logger.warning(f"⚠️ vLLM检测失败（第{attempt+1}次）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
        
        logger.error("❌ vLLM检测全部失败，后续请求将尝试动态检测")
    
    def _detect_capabilities(self):
        """检测 vLLM 支持的端点和模型"""
        # 检测 /v1/models
        resp = requests.get(
            f"{self.url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=10
        )
        resp.raise_for_status()
        
        models_data = resp.json().get("data", [])
        self.available_models = [m["id"] for m in models_data]
        logger.info(f"[LLMClient] vLLM可用模型: {self.available_models}")
        
        # 自动匹配模型名（关键修复）
        if self.available_models:
            if self.model in self.available_models:
                self.actual_model_name = self.model
            else:
                # 尝试模糊匹配
                for avail in self.available_models:
                    if "qwen" in avail.lower() or "1.5b" in avail.lower():
                        self.actual_model_name = avail
                        logger.info(f"[LLMClient] 模型名自动修正: {self.model} -> {avail}")
                        break
                else:
                    self.actual_model_name = self.available_models[0]
                    logger.info(f"[LLMClient] 使用第一个可用模型: {self.actual_model_name}")
        
        # 检测 chat/completions
        test_resp = requests.post(
            f"{self.url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.actual_model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1
            },
            timeout=10
        )
        self.chat_completions_available = test_resp.status_code == 200
        logger.info(f"[LLMClient] chat/completions: {self.chat_completions_available}")
        
        # 检测 completions（回退）
        test_resp2 = requests.post(
            f"{self.url}/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.actual_model_name,
                "prompt": "hi",
                "max_tokens": 1
            },
            timeout=10
        )
        self.completions_available = test_resp2.status_code == 200
        logger.info(f"[LLMClient] completions: {self.completions_available}")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        生成文本（带动态检测）
        """
        # 如果之前检测失败，每次请求前尝试重新检测（最多1次）
        if not self.chat_completions_available and not self.completions_available:
            logger.warning("🔄 之前检测失败，尝试动态重连...")
            self._detect_with_retry(max_retries=1)
        
        if self.chat_completions_available:
            return self._generate_chat(prompt, max_tokens, temperature, system_prompt)
        elif self.completions_available:
            return self._generate_completion(prompt, max_tokens, temperature)
        else:
            return "[错误] vLLM 未运行或不支持任何已知端点"
    
    def _generate_chat(self, prompt, max_tokens, temperature, system_prompt):
        """chat/completions 生成"""
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
            resp = requests.post(
                f"{self.url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=data,
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.chat_completions_available = False
                return self._generate_completion(prompt, max_tokens, temperature)
            return f"[错误] HTTP {e.response.status_code}: {e.response.text[:100]}"
        except Exception as e:
            return f"[错误] {str(e)[:100]}"
    
    def _generate_completion(self, prompt, max_tokens, temperature):
        """completions 回退生成"""
        data = {
            "model": self.actual_model_name,
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "stream": False
        }
        
        try:
            resp = requests.post(
                f"{self.url}/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=data,
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"]
        except Exception as e:
            return f"[错误] {str(e)[:100]}"
    
    def health_check(self) -> Dict:
        """健康检查"""
        try:
            resp = requests.get(
                f"{self.url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            if resp.status_code == 200:
                return {
                    "status": "healthy",
                    "models": self.available_models,
                    "actual_model": self.actual_model_name,
                    "chat_available": self.chat_completions_available,
                    "completions_available": self.completions_available
                }
            return {"status": "unhealthy", "code": resp.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}


_llm_client_instance: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance