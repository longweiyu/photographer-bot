"""
llm.py — Ollama LLM 调用封装
"""

import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

SYSTEM_PROMPT = """你是一个专业的客服助手。请根据提供的参考资料和规则来回答用户的问题。
规则：
1. 只根据参考资料中的信息来回答，不要编造。
2. 回答要简洁、准确、友好，非常的客气（比如使用"你好呀宝宝~"，"（＾Ｏ＾☆♪"，"(^_^)v" 等）。
3. 如果参考资料中没有相关信息，请诚实地说"抱歉，我暂时无法回答这个问题，建议您加微信lovetange77联系我"。
4. 使用中文回答。"""


def ask_llm(question: str, context_chunks: list[str]) -> str:
    context_text = "\n---\n".join(context_chunks) if context_chunks else "（无相关参考资料）"

    user_prompt = f"""参考资料：
{context_text}

用户问题：{question}

请根据上面的参考资料回答用户的问题。"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        },
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except requests.ConnectionError:
        return "⚠️ 无法连接到 Ollama 服务，请确认 Ollama 已启动。"
    except Exception as e:
        return f"⚠️ LLM 调用出错：{e}"
