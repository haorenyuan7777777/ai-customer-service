"""
Alpaca风格JSON数据加载与处理
支持多种JSON格式：标准JSON数组、JSON Lines、多JSON对象拼接
支持80/20划分、批量向量化、意图标签生成
"""
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from src.config import DATA_SPLIT
from src.models.embedding_model import BGEEmbedding


class AlpacaDataLoader:
    """Alpaca格式数据加载器"""
    
    # 意图关键词规则（用于半监督标注）
    INTENT_KEYWORDS = {
        "price_inquiry": ["价格", "多少钱", "费用", "报价", "怎么卖", "贵不贵", "便宜", "折扣", "优惠"],
        "purchase_intent": ["购买", "下单", "买", "成交", "订购", "我要", "订一个", "来一份", "怎么买"],
        "technical_issue": ["故障", "报错", "怎么修", "无法", "坏了", "不工作", "出问题", "维修", "保养"],
        "complaint": ["投诉", "差评", "退款", "售后", "退货", "不满意", "坑人", "骗人", "质量差"],
    }
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.raw_data = []
        self.train_data = []
        self.test_data = []
    
    def load(self) -> List[Dict]:
        """加载原始数据，自动检测格式"""
        content = self.data_path.read_text(encoding='utf-8-sig')  # 自动处理BOM
        
        # 尝试1：标准JSON数组
        try:
            self.raw_data = json.loads(content)
            print(f"[DataLoader] 标准JSON格式，加载 {len(self.raw_data)} 条")
            self._annotate_intent()
            return self.raw_data
        except json.JSONDecodeError:
            pass
        
        # 尝试2：JSON Lines格式（每行一个JSON对象）
        try:
            self.raw_data = []
            for line_num, line in enumerate(content.strip().split('\n'), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self.raw_data.append(obj)
                except json.JSONDecodeError:
                    print(f"[DataLoader] 跳过无效行 {line_num}: {line[:50]}...")
            
            if self.raw_data:
                print(f"[DataLoader] JSON Lines格式，加载 {len(self.raw_data)} 条")
                self._annotate_intent()
                return self.raw_data
        except Exception:
            pass
        
        # 尝试3：多个JSON对象拼接（提取所有JSON对象）
        try:
            self.raw_data = self._extract_json_objects(content)
            if self.raw_data:
                print(f"[DataLoader] 多对象拼接格式，加载 {len(self.raw_data)} 条")
                self._annotate_intent()
                return self.raw_data
        except Exception:
            pass
        
        raise ValueError(f"无法解析数据文件: {self.data_path}，请检查格式")
    
    def _extract_json_objects(self, content: str) -> List[Dict]:
        """从文本中提取所有JSON对象"""
        objects = []
        decoder = json.JSONDecoder()
        idx = 0
        content = content.strip()
        
        while idx < len(content):
            try:
                # 跳过空白字符
                while idx < len(content) and content[idx] in ' \t\n\r':
                    idx += 1
                if idx >= len(content):
                    break
                
                obj, end_idx = decoder.raw_decode(content, idx)
                if isinstance(obj, dict):
                    objects.append(obj)
                idx += end_idx
            except json.JSONDecodeError:
                idx += 1
        
        return objects
    
    def _annotate_intent(self):
        """基于关键词规则自动标注意图"""
        annotated_count = 0
        
        for item in self.raw_data:
            text = item.get("instruction", "") + " " + item.get("input", "")
            intent = self._classify_by_keywords(text)
            item["intent"] = intent
            if intent != "general_query":
                annotated_count += 1
        
        print(f"[DataLoader] 意图标注完成: {annotated_count}条有明确意图, "
              f"{len(self.raw_data) - annotated_count}条为一般咨询")
    
    def _classify_by_keywords(self, text: str) -> str:
        """基于关键词分类意图"""
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return intent
        return "general_query"
    
    def split(self, train_ratio: float = None, seed: int = None) -> Tuple[List, List]:
        """划分训练集和测试集（分层抽样，保持意图分布）"""
        train_ratio = train_ratio or DATA_SPLIT["train_ratio"]
        seed = seed or DATA_SPLIT["random_seed"]
        
        random.seed(seed)
        
        # 按意图分组
        intent_groups = {}
        for item in self.raw_data:
            intent = item.get("intent", "general_query")
            intent_groups.setdefault(intent, []).append(item)
        
        self.train_data = []
        self.test_data = []
        
        # 每层意图按比例划分
        for intent, items in intent_groups.items():
            random.shuffle(items)
            split_idx = int(len(items) * train_ratio)
            self.train_data.extend(items[:split_idx])
            self.test_data.extend(items[split_idx:])
        
        # 再次打乱
        random.shuffle(self.train_data)
        random.shuffle(self.test_data)
        
        print(f"[DataLoader] 分层划分完成:")
        print(f"  训练集: {len(self.train_data)}条")
        print(f"  测试集: {len(self.test_data)}条")
        
        # 统计意图分布
        self._print_intent_distribution()
        
        return self.train_data, self.test_data
    
    def _print_intent_distribution(self):
        """打印意图分布统计"""
        def count_intents(data):
            counts = {}
            for item in data:
                intent = item.get("intent", "general_query")
                counts[intent] = counts.get(intent, 0) + 1
            return counts
        
        train_dist = count_intents(self.train_data)
        test_dist = count_intents(self.test_data)
        
        print(f"[DataLoader] 意图分布:")
        for intent in sorted(set(list(train_dist.keys()) + list(test_dist.keys()))):
            t = train_dist.get(intent, 0)
            v = test_dist.get(intent, 0)
            print(f"  {intent:20s}: 训练{t:5d} | 测试{v:5d}")
    
    def save_split(self, output_dir: str):
        """保存划分后的数据"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
            json.dump(self.train_data, f, ensure_ascii=False, indent=2)
        
        with open(output_dir / "test.json", 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f, ensure_ascii=False, indent=2)
        
        print(f"[DataLoader] 数据已保存到 {output_dir}")
    
    def vectorize(self, data: List[Dict], batch_size: int = 500) -> np.ndarray:
        """批量向量化instruction字段"""
        embedding = BGEEmbedding()
        
        texts = [item["instruction"] for item in data]
        print(f"[DataLoader] 开始向量化 {len(texts)} 条数据，批次大小: {batch_size}...")
        
        import time
        start = time.time()
        vectors = embedding.encode(texts, batch_size=batch_size, show_progress=True)
        elapsed = time.time() - start
        
        print(f"[DataLoader] 向量化完成: {len(texts)}条 / {elapsed:.2f}s "
              f"({len(texts)/elapsed:.1f}条/秒)")
        print(f"[DataLoader] 向量形状: {vectors.shape}")
        
        return vectors
    
    def prepare_for_milvus(self, data: List[Dict], vectors: np.ndarray) -> Dict:
        """准备Milvus插入数据"""
        return {
            "ids": [str(item.get("id", f"doc_{i}")) for i, item in enumerate(data)],
            "vectors": vectors,
            "instructions": [item["instruction"] for item in data],
            "outputs": [item["output"] for item in data],
            "intents": [item.get("intent", "general_query") for item in data],
        }


def build_knowledge_base(data_path: str, output_dir: str = "data/processed"):
    """
    一键构建知识库
    
    流程:
    1. 加载Alpaca JSON（自动检测格式）
    2. 自动意图标注（关键词规则）
    3. 80/20分层划分
    4. 训练集GPU向量化（批量500）
    5. 插入Milvus
    6. 验证入库数量
    """
    from src.rag.milvus_store import MilvusKnowledgeStore
    import torch
    
    print("=" * 60)
    print("阶段2：知识库构建")
    print("=" * 60)
    
    # 1. 加载和标注（自动检测格式）
    loader = AlpacaDataLoader(data_path)
    loader.load()
    
    # 2. 划分
    train_data, test_data = loader.split()
    loader.save_split(output_dir)
    
    # 3. 向量化训练集
    print(f"\n[KnowledgeBase] GPU显存状态: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    train_vectors = loader.vectorize(train_data, batch_size=500)
    
    # 4. 插入Milvus
    print(f"\n[KnowledgeBase] 开始插入Milvus...")
    store = MilvusKnowledgeStore()
    store.clear_collection()  # ← 新增：清空旧数据
    milvus_data = loader.prepare_for_milvus(train_data, train_vectors)
    
    # 分批插入（避免单次请求过大）
    batch_size = 1000
    total = len(milvus_data["ids"])
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        store.insert(
            ids=milvus_data["ids"][i:end],
            vectors=milvus_data["vectors"][i:end],
            instructions=milvus_data["instructions"][i:end],
            outputs=milvus_data["outputs"][i:end],
            intents=milvus_data["intents"][i:end]
        )
        print(f"[KnowledgeBase] 已插入 {end}/{total} 条")
    
    # 5. 验证
    stats = store.get_stats()
    print(f"\n[KnowledgeBase] 构建完成!")
    print(f"  Milvus实体数: {stats['num_entities']}")
    print(f"  预期训练集: {len(train_data)}")
    print(f"  验证: {'✅ 通过' if stats['num_entities'] == len(train_data) else '❌ 不匹配'}")
    
    return stats


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/knowledge.json"
    build_knowledge_base(data_path)