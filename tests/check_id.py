# check_id.py
import json
from src.rag.milvus_store import get_milvus_store

with open("data/processed/test.json") as f:
    test_data = json.load(f)

store = get_milvus_store()
store.load_collection()

for item in test_data[:10]:
    # 【修复】id 在 JSON 中是 int，但 Milvus 中可能是 VARCHAR
    # 先尝试 int 格式，失败则尝试字符串格式
    id_val = item["id"]
    try:
        expr = f'id == {id_val}'
        result = store.collection.query(expr=expr, output_fields=["id", "instruction"])
    except Exception as e:
        expr = f'id == "{id_val}"'
        result = store.collection.query(expr=expr, output_fields=["id", "instruction"])
    
    print(f"id={id_val}: {'✅ 存在' if result else '❌ 缺失'} | {item['instruction'][:40]}...")
    if result:
        print(f"    Milvus中存储的instruction: {result[0].get('instruction', 'N/A')[:40]}...")