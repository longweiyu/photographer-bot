"""
FastAPI 入口 — 提供 /chat 接口 + 静态前端文件服务
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import search
from llm import ask_llm

app = FastAPI(title="客服机器人 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    1. 用户问题 → 向量检索相关文档片段
    2. 片段 + 问题 → LLM 生成回答
    """
    try:
        hits = search(req.question, top_k=req.top_k)
        context_chunks = [h["text"] for h in hits]
        sources = [h["source"] for h in hits]
        answer = ask_llm(req.question, context_chunks)
        return ChatResponse(answer=answer, sources=list(set(sources)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- 静态文件服务 ----------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
