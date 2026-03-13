"""
retriever.py — 向量检索模块
使用 Ollama Embedding + JSON 文件存储
零编译依赖，Windows 友好
"""

import os
import json
import math
import requests

# ---------- 配置 ----------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DB_PATH = os.getenv("DB_PATH", "./vector_store.json")

# ---------- 内存缓存 ----------
_store = None


def _load_store():
    global _store
    if _store is not None:
        return _store
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            _store = json.load(f)
    else:
        _store = {"documents": []}
    return _store


def _save_store(store):
    global _store
    _store = store
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def embed_texts(texts):
    """调用 Ollama /api/embed 接口生成向量"""
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"]


def _cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def insert(texts, metadatas=None):
    """将文本片段写入向量库"""
    store = _load_store()
    if metadatas is None:
        metadatas = [{"source": "unknown"}] * len(texts)

    batch_size = 10
    total = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        vectors = embed_texts(batch_texts)

        for text, vec, meta in zip(batch_texts, vectors, batch_metas):
            store["documents"].append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "vector": vec,
            })
            total += 1
        print(f"   已处理 {min(i + batch_size, len(texts))}/{len(texts)} ...")

    _save_store(store)
    print(f"✅  已写入 {total} 条记录，总计 {len(store['documents'])} 条")


def search(query, top_k=3):
    """查询 → 返回最相关的 top_k 条文档片段"""
    store = _load_store()

    if not store["documents"]:
        return []

    query_vec = embed_texts([query])[0]

    scored = []
    for doc in store["documents"]:
        score = _cosine_sim(query_vec, doc["vector"])
        scored.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
