"""
意图分类器
- L1: 业务意图（多分类）— 关键词规则 + bert-base-chinese GPU
- L2: 优先级（二分类）— 规则判定

输出格式对齐 flow.dag.yaml:
{
    "intent": "销售转化/技术支持/投诉处理/标准客服",
    "category": "sales/tech/complaint/general",
    "priority": 1-4,
    "keywords": ["触发词"],
    "confidence": 0.95
}
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ 配置加载 ============

def _load_intent_config() -> Dict:
    """从 prompts.yaml 加载意图配置"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "prompts.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('intent_routing', {})
    
    # 兜底配置
    return {
        "销售转化": {
            "keywords": ["价格", "优惠", "活动", "推荐", "对比", "买", "下单", "付款", "多少钱"],
            "template": "sales",
            "priority": 2
        },
        "技术支持": {
            "keywords": ["故障", "无法", "报错", "怎么修", "安装", "使用", "不工作", "坏了"],
            "template": "technical_support",
            "priority": 3
        },
        "投诉处理": {
            "keywords": ["投诉", "不满", "差评", "退款", "赔偿", "垃圾", "骗人", "骗子", "退钱"],
            "template": "complaint",
            "priority": 1
        },
        "标准客服": {
            "keywords": [],
            "template": "customer_service",
            "priority": 4
        }
    }


# ============ 关键词规则 ============

class KeywordIntentClassifier:
    """基于关键词的意图分类器（快速路径）"""
    
    def __init__(self):
        self.config = _load_intent_config()
    
    def classify(self, message: str) -> Optional[Dict]:
        """
        关键词匹配分类
        
        Returns:
            匹配结果 或 None（未命中）
        """
        if not message:
            return None
        
        detected_keywords = []
        matched_intent = None
        highest_priority = 999
        
        for intent_name, intent_config in self.config.items():
            keywords = intent_config.get('keywords', [])
            for kw in keywords:
                if kw in message:
                    detected_keywords.append(kw)
                    priority = intent_config.get('priority', 4)
                    if priority < highest_priority:
                        highest_priority = priority
                        matched_intent = intent_name
        
        if matched_intent:
            return {
                "intent": matched_intent,
                "category": self._intent_to_category(matched_intent),
                "priority": highest_priority,
                "keywords": list(set(detected_keywords)),  # 去重
                "confidence": 0.85,
                "source": "keyword"
            }
        
        return None
    
    @staticmethod
    def _intent_to_category(intent: str) -> str:
        """意图名称 → category编码"""
        mapping = {
            "销售转化": "sales",
            "技术支持": "tech",
            "投诉处理": "complaint",
            "标准客服": "general"
        }
        return mapping.get(intent, "general")


# ============ BERT模型分类 ============

class BERTIntentClassifier:
    """
    基于 bert-base-chinese 的意图分类器（GPU）
    
    【演示实现】：使用预训练模型 + 简单分类头
    【生产目标】：在11157条数据上微调，4分类输出
    """
    
    _instance: Optional['BERTIntentClassifier'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.label_map = {
            0: ("标准客服", "general"),
            1: ("销售转化", "sales"),
            2: ("技术支持", "tech"),
            3: ("投诉处理", "complaint")
        }
        
        self._load_model()
    
    def _load_model(self):
        """加载模型（GPU优先）"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            model_path = os.getenv(
                "BERT_INTENT_MODEL_PATH",
                "/mnt/e/modelscope/google-bert/bert-base-chinese"
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🚀 加载意图模型: {model_path} → {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 【演示实现】：使用预训练模型，假设有4分类头
            # 【生产目标】：加载微调后的分类模型
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=4
                )
            except Exception:
                # 无分类头时，创建简单分类器
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=4,
                    ignore_mismatched_sizes=True
                )
            
            self.model.to(self.device)
            self.model.eval()
            
            if self.device == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"✅ 意图模型加载完成，显存占用: {mem:.2f}GB")
            else:
                logger.info("✅ 意图模型加载完成（CPU模式）")
                
        except Exception as e:
            logger.error(f"❌ 意图模型加载失败: {e}")
            self.model = None
    
    def predict(self, message: str) -> Dict:
        """
        BERT模型预测意图
        
        Returns:
            {
                "intent": "销售转化",
                "category": "sales",
                "priority": 2,
                "keywords": [],
                "confidence": 0.92,
                "source": "bert"
            }
        """
        if self.model is None:
            # 模型未加载，返回兜底
            return {
                "intent": "标准客服",
                "category": "general",
                "priority": 4,
                "keywords": [],
                "confidence": 1.0,
                "source": "fallback"
            }
        
        import torch
        
        # 编码
        inputs = self.tokenizer(
            message,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()
        
        intent_name, category = self.label_map.get(pred_id, ("标准客服", "general"))
        
        # 获取优先级
        config = _load_intent_config()
        priority = config.get(intent_name, {}).get('priority', 4)
        
        return {
            "intent": intent_name,
            "category": category,
            "priority": priority,
            "keywords": [],
            "confidence": round(confidence, 4),
            "source": "bert"
        }


# ============ 统一入口 ============

class IntentClassifier:
    """
    意图分类器统一入口
    
    策略：
    1. 先关键词匹配（快速，覆盖高频场景）
    2. 未命中时调用BERT模型（准确，覆盖长尾）
    3. 模型失败时兜底标准客服
    """
    
    def __init__(self):
        self.keyword_clf = KeywordIntentClassifier()
        self.bert_clf = BERTIntentClassifier()
    
    def classify(self, message: str) -> Dict:
        """
        分类用户意图
        
        Args:
            message: 用户输入消息
        
        Returns:
            意图结果字典（对齐 flow.dag.yaml 变量引用）
        """
        # 1. 关键词匹配
        keyword_result = self.keyword_clf.classify(message)
        if keyword_result:
            logger.info(f"🔑 关键词命中: {keyword_result['intent']}, 词: {keyword_result['keywords']}")
            return keyword_result
        
        # 2. BERT模型预测
        logger.info("🤖 BERT模型预测意图...")
        bert_result = self.bert_clf.predict(message)
        logger.info(f"🤖 BERT预测: {bert_result['intent']}, 置信度: {bert_result['confidence']}")
        return bert_result


# 单例
_intent_classifier: Optional[IntentClassifier] = None

def get_intent_classifier() -> IntentClassifier:
    """获取意图分类器单例"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


# 兼容 Promptflow 节点入口
def run(message: str) -> Dict:
    """Promptflow节点入口函数"""
    clf = get_intent_classifier()
    return clf.classify(message)