"""
bert-base-chinese 意图识别模型 - ModelScope本地路径版
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from pathlib import Path

from src.config import MODELS, INTENT_LABELS


class IntentClassifier:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, device: str = "cuda", max_length: int = 128):
        if self._initialized:
            return
        
        model_path = MODELS["intent"]["local_path"]
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        
        # 检查本地路径
        if Path(model_path).exists():
            print(f"[Intent] 从ModelScope本地路径加载: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=MODELS["intent"]["num_labels"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        else:
            # 回退到在线加载
            model_name = MODELS["intent"]["name"]
            print(f"[Intent] 本地路径不存在，从HuggingFace下载: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=MODELS["intent"]["num_labels"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        
        # 标签映射
        self.id2label = INTENT_LABELS
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # 关键词规则
        self.keyword_rules = {
            "price_inquiry": ["价格", "多少钱", "费用", "报价", "怎么卖", "贵不贵", "便宜"],
            "purchase_intent": ["购买", "下单", "买", "成交", "订购", "我要", "订一个"],
            "technical_issue": ["故障", "报错", "怎么修", "无法", "坏了", "不工作", "出问题"],
            "complaint": ["投诉", "差评", "退款", "售后", "退货", "不满意", "坑人"],
        }
        
        self.model.eval()
        print(f"[Intent] 加载完成 | 显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        self._initialized = True
    
    def predict_by_rule(self, text: str) -> str:
        for intent, keywords in self.keyword_rules.items():
            if any(kw in text for kw in keywords):
                return intent
        return "general_query"
    
    @torch.no_grad()
    def predict(self, text: str, use_rule: bool = True) -> Dict:
        if use_rule:
            rule_result = self.predict_by_rule(text)
            if rule_result != "general_query":
                return {
                    "intent": rule_result,
                    "confidence": 0.95,
                    "method": "rule",
                    "probabilities": {rule_result: 0.95, "general_query": 0.05}
                }
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()
        
        prob_dict = {
            self.id2label[i]: probs[0][i].item() 
            for i in range(len(self.id2label))
        }
        
        return {
            "intent": self.id2label[pred_id],
            "confidence": confidence,
            "method": "model",
            "probabilities": prob_dict
        }
    
    def batch_predict(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = [self.predict(t, use_rule=True) for t in batch]
            results.extend(batch_results)
        return results


# 全局实例
intent_classifier = IntentClassifier()