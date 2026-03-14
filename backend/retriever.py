"""
retriever.py — 向量检索模块
使用 Ollama Embedding + JSON 文件存储 + 中文关键词匹配
"""

import os
import re
import json
import math
import requests

# ---------- 配置 ----------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DB_PATH = os.getenv("DB_PATH", "./vector_store.json")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))

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


def _chinese_tokenize(text):
    """
    中文分词：滑动窗口切 2-gram 和 3-gram
    比如 "什么设备" → {"什么", "么设", "设备", "什么设", "么设备"}
    不依赖 jieba，零依赖
    """
    tokens = set()
    # 提取纯中文部分
    cn_parts = re.findall(r'[\u4e00-\u9fff]+', text)
    for part in cn_parts:
        # 单字也加入（短文本很重要）
        for ch in part:
            tokens.add(ch)
        # 2-gram
        for i in range(len(part) - 1):
            tokens.add(part[i:i+2])
        # 3-gram
        for i in range(len(part) - 2):
            tokens.add(part[i:i+3])
    # 英文和数字整词
    en_parts = re.findall(r'[a-zA-Z0-9]+', text.lower())
    tokens.update(en_parts)
    return tokens


def _extract_question_text(doc_text):
    """从 QA 片段中提取问题部分"""
    match = re.match(r'Q\d*\s*[：:、.\.\s]\s*(.+?)[\n]', doc_text)
    if match:
        return match.group(1).strip()
    # 取第一行
    first_line = doc_text.split("\n")[0]
    return re.sub(r'^Q\d*\s*[：:、.\.\s]\s*', '', first_line).strip()


def _keyword_score(query, doc_text):
    """
    关键词匹配打分
    分别匹配文档的问题部分和全文，问题部分权重更高
    """
    query_tokens = _chinese_tokenize(query)
    if not query_tokens:
        return 0.0

    # 问题部分匹配（权重高）
    question_part = _extract_question_text(doc_text)
    q_tokens = _chinese_tokenize(question_part)
    q_overlap = query_tokens & q_tokens
    q_score = sum(len(w) for w in q_overlap) / max(sum(len(w) for w in query_tokens), 1)

    # 全文匹配（权重低，兜底用）
    full_tokens = _chinese_tokenize(doc_text)
    f_overlap = query_tokens & full_tokens
    f_score = sum(len(w) for w in f_overlap) / max(sum(len(w) for w in query_tokens), 1)

    # 问题部分占 70%，全文占 30%
    return q_score * 0.7 + f_score * 0.3


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


def search(query, top_k=5):
    """
    混合检索：向量相似度 (50%) + 关键词匹配 (50%)
    过滤低分结果
    """
    store = _load_store()

    if not store["documents"]:
        return []

    query_vec = embed_texts([query])[0]

    scored = []
    for doc in store["documents"]:
        vec_score = _cosine_sim(query_vec, doc["vector"])
        kw_score = _keyword_score(query, doc["text"])
        # 混合打分：各占一半
        final_score = vec_score * 0.5 + kw_score * 0.5

        scored.append({
            "text": doc["text"],
            "source": doc["source"],
            "vec_score": round(vec_score, 4),
            "kw_score": round(kw_score, 4),
            "score": round(final_score, 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # 过滤低于阈值的结果
    filtered = [s for s in scored[:top_k] if s["score"] >= SCORE_THRESHOLD]

    return filtered