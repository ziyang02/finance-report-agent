"""Retriever：对每个子问题检索->重排，产出证据 + 全局引用表。

反思回路：Critic 判 revise 回来时，state 里带着 critique.notes——本节点会把 notes
作为一条额外的补充检索 query（Self-RAG 式反思重检索），让第二轮拿到更有针对性的证据。
"""
from __future__ import annotations

from src.agents.state import ReportState


def make_retriever_node(pipeline):
    def retriever_node(state: ReportState) -> dict:
        queries = list(state.get("subquestions", []))

        # 反思重检索：把 Critic 的意见变成一条补充 query
        notes = (state.get("critique") or {}).get("notes", "")
        if notes:
            queries.append(f"针对以下不足补充证据：{notes}")

        evidence, seen, citations = [], {}, []
        for q in queries:
            chunks = pipeline.retrieve(q) if pipeline else []
            evidence.append({"q": q, "chunks": chunks})
            # 去重建全局引用表：同一 source 只占一个编号
            for c in chunks:
                src = c.get("source", "?")
                if src not in seen:
                    seen[src] = len(citations) + 1
                    citations.append({"n": seen[src], "source": src,
                                      "text": c.get("text", "")})

        return {"evidence": evidence, "citations": citations}

    return retriever_node
