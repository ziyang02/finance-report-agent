"""单轮 RAG 问答：检索 -> 带引用回答。评估的被测对象（区别于完整多智能体研报）。"""
from __future__ import annotations

from src.config import LLMConfig
from src.llm import chat

_PROMPT = """仅根据下列资料回答问题，不要使用资料外的知识；资料不足就回答"资料不足"。

资料：
{context}

问题：{question}
简洁作答："""


def answer_question(question: str, pipeline, use_rerank: bool = True) -> dict:
    """返回 {answer, contexts}，contexts 为检索到的文本列表（供指标计算）。"""
    hits = pipeline.retrieve(question, use_rerank=use_rerank) if pipeline else []
    contexts = [h.get("text", "") for h in hits]
    ctx_block = "\n".join(f"- {c}" for c in contexts) or "（无）"

    if not LLMConfig.available():
        return {"answer": f"[离线stub] 命中{len(contexts)}条资料", "contexts": contexts}

    msg = chat([{"role": "user", "content": _PROMPT.format(context=ctx_block, question=question)}])
    return {"answer": msg.content, "contexts": contexts}
