"""
retriever.py — 向量检索模块
使用 ChromaDB (嵌入式) + sentence-transformers 做 embedding
兼容 Windows / macOS / Linux
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

# ---------- 配置 ----------
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


# ---------- 单例 ----------
_client = None
_collection = None
_embedder = None

def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embedder()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def insert(texts: list[str], metadatas: list[dict] | None = None):
    collection = _get_collection()
    vectors = embed_texts(texts)
    if metadatas is None:
        metadatas = [{"source": "unknown"}] * len(texts)

    existing_count = collection.count()
    ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"✅  已写入 {len(texts)} 条记录到 [{COLLECTION_NAME}]")


def search(query: str, top_k: int = 3) -> list[dict]:
    collection = _get_collection()

    if collection.count() == 0:
        return []

    query_vec = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "score": 1 - results["distances"][0][i],
        })
    return hits