"""
知识库增量更新模块
支持：全量重建、增量追加、差异更新
"""
import json
from pathlib import Path
from typing import List, Dict, Set
import numpy as np

from src.rag.milvus_store import MilvusKnowledgeStore
from src.models.embedding_model import BGEEmbedding
from src.rag.data_loader import AlpacaDataLoader


class KnowledgeBaseUpdater:
    """知识库增量更新器"""
    
    def __init__(self, milvus_store: MilvusKnowledgeStore = None):
        self.store = milvus_store or MilvusKnowledgeStore()
        self.embedding = BGEEmbedding()
    
    def incremental_update(
        self,
        new_data_path: str,
        id_field: str = "id",
        batch_size: int = 500
    ) -> Dict:
        """
        增量更新知识库
        
        流程：
        1. 加载新数据
        2. 对比已有ID，筛选出新增数据
        3. 向量化新增数据
        4. 插入Milvus
        
        Args:
            new_data_path: 新增数据文件路径
            id_field: ID字段名
            batch_size: 批大小
        
        Returns:
            更新统计信息
        """
        print("=" * 60)
        print("知识库增量更新")
        print("=" * 60)
        
        # 1. 加载新数据
        loader = AlpacaDataLoader(new_data_path)
        new_data = loader.load()
        
        # 2. 获取已有ID
        existing_ids = self._load_id_cache()
        
        # 3. 筛选新增数据
        added_data = []
        updated_data = []
        
        for item in new_data:
            item_id = str(item.get(id_field, ""))
            if not item_id:
                added_data.append(item)
            elif item_id not in existing_ids:
                added_data.append(item)
            else:
                updated_data.append(item)
        
        print(f"[Updater] 新数据: {len(new_data)}条")
        print(f"[Updater] 新增: {len(added_data)}条")
        print(f"[Updater] 可能更新: {len(updated_data)}条")
        
        if not added_data and not updated_data:
            print("[Updater] 无变化，跳过更新")
            return {"status": "no_change", "added": 0, "updated": 0}
        
        # 4. 处理新增数据
        if added_data:
            self._process_additions(added_data, batch_size)
        
        # 5. 处理更新数据（删除旧数据后重新插入）
        if updated_data:
            self._process_updates(updated_data, batch_size)
        
        # 6. 更新ID缓存
        self._save_id_cache(new_data, id_field)
        
        # 7. 验证
        stats = self.store.get_stats()
        print(f"\n[Updater] 更新完成!")
        print(f"  当前总实体数: {stats['num_entities']}")
        
        return {
            "status": "success",
            "added": len(added_data),
            "updated": len(updated_data),
            "total": stats["num_entities"]
        }
    
    def _process_additions(self, data: List[Dict], batch_size: int):
        """处理新增数据：向量化并插入"""
        print(f"\n[Updater] 开始向量化 {len(data)} 条新增数据...")
        
        texts = [item["instruction"] for item in data]
        vectors = self.embedding.encode(texts, batch_size=batch_size, show_progress=True)
        
        print(f"[Updater] 插入Milvus...")
        milvus_data = self._prepare_milvus_data(data, vectors)
        
        # 分批插入
        insert_batch = 1000
        total = len(milvus_data["ids"])
        for i in range(0, total, insert_batch):
            end = min(i + insert_batch, total)
            self.store.insert(
                ids=milvus_data["ids"][i:end],
                vectors=milvus_data["vectors"][i:end],
                instructions=milvus_data["instructions"][i:end],
                outputs=milvus_data["outputs"][i:end],
                intents=milvus_data["intents"][i:end]
            )
        
        print(f"[Updater] 新增 {len(data)} 条完成")
    
    def _process_updates(self, data: List[Dict], batch_size: int):
        """处理更新数据：删除旧数据后重新插入"""
        print(f"\n[Updater] 处理 {len(data)} 条更新数据...")
        
        ids_to_delete = [str(item.get("id", "")) for item in data if item.get("id")]
        if ids_to_delete:
            self.store.delete_by_ids(ids_to_delete)
            print(f"[Updater] 已删除旧数据: {len(ids_to_delete)}条")
        
        self._process_additions(data, batch_size)
    
    def _prepare_milvus_data(self, data: List[Dict], vectors: np.ndarray) -> Dict:
        """准备Milvus数据"""
        return {
            "ids": [str(item.get("id", f"doc_{i}")) for i, item in enumerate(data)],
            "vectors": vectors,
            "instructions": [item["instruction"] for item in data],
            "outputs": [item["output"] for item in data],
            "intents": [item.get("intent", "general_query") for item in data],
        }
    
    def _load_id_cache(self) -> Set[str]:
        """加载ID缓存"""
        cache_file = Path("data/processed/id_cache.json")
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()
    
    def _save_id_cache(self, data: List[Dict], id_field: str):
        """保存ID缓存"""
        cache_file = Path("data/processed/id_cache.json")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        ids = [str(item.get(id_field, "")) for item in data if item.get(id_field)]
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)
        
        print(f"[Updater] ID缓存已更新: {len(ids)}条")
    
    def full_rebuild(self, data_path: str):
        """全量重建（兜底方案）"""
        print("[Updater] 执行全量重建...")
        self.store.clear_collection()
        
        from src.rag.data_loader import build_knowledge_base
        return build_knowledge_base(data_path)


def update_knowledge_base(
    data_path: str,
    mode: str = "incremental",
    id_field: str = "id"
):
    """
    更新知识库入口函数
    
    Args:
        data_path: 数据文件路径
        mode: "incremental"(增量) 或 "full"(全量)
        id_field: ID字段名
    """
    updater = KnowledgeBaseUpdater()
    
    if mode == "full":
        return updater.full_rebuild(data_path)
    else:
        return updater.incremental_update(data_path, id_field)


if __name__ == "__main__":
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/knowledge_new.json"
    mode = sys.argv[2] if len(sys.argv) > 2 else "incremental"
    
    result = update_knowledge_base(data_path, mode)
    print(f"\n更新结果: {result}")