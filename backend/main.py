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
    try:
        hits = search(req.question, top_k=req.top_k)

        # ========== 命令行日志输出 ==========
        print("\n" + "=" * 60)
        print(f"📝 用户问题: {req.question}")
        print("-" * 60)
        if hits:
            for i, h in enumerate(hits):
                preview = h["text"][:80].replace("\n", " ")
                print(f"  [{i+1}] 综合={h['score']}  向量={h.get('vec_score','-')}  关键词={h.get('kw_score','-')}")
                print(f"      {preview}...")
        else:
            print("  ⚠️  未检索到相关文档（全部低于阈值）")
        print("=" * 60)

        context_chunks = [h["text"] for h in hits]
        sources = [h["source"] for h in hits]
        answer = ask_llm(req.question, context_chunks)

        print(f"🤖 回答: {answer[:100]}...")
        print()

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