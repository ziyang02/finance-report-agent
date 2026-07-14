"""Analyst：基于证据分点分析，每个结论必须挂引用编号 [n]（防幻觉抓手）。

[n] 来自 state.citations 的全局编号，Writer 会用同一套编号渲染引用列表，保证前后一致。
"""
from __future__ import annotations

from src.agents.state import ReportState
from src.config import LLMConfig
from src.llm import chat

_PROMPT = """你是资深股票分析师。基于下列带编号的证据，对标的「{target}」逐子问题分析。
硬性规则：
1) 每个结论后必须用 [编号] 标注支撑证据，可多个如 [1][3]；
2) 证据未覆盖的内容不要臆测，宁可写"证据不足"；
3) 涉及数字时直接引用证据里的数字。

子问题：
{subqs}

证据：
{context}

输出分点分析（Markdown，每个子问题一节）。"""


def format_citations(citations: list[dict]) -> str:
    if not citations:
        return "（无证据）"
    return "\n".join(f"[{c['n']}] ({c['source']}) {c['text'][:220]}" for c in citations)


def analyst_node(state: ReportState) -> dict:
    citations = state.get("citations", [])
    subqs = state.get("subquestions", [])
    context = format_citations(citations)

    if not LLMConfig.available():
        avail = "".join(f"[{c['n']}]" for c in citations[:3]) or "[无]"
        body = "\n".join(f"### {q}\n- 基于证据的分析（离线占位）{avail}" for q in subqs)
        return {"analysis": body}

    msg = chat([{"role": "user", "content": _PROMPT.format(
        target=state["target"], subqs="\n".join(f"- {q}" for q in subqs), context=context)}])
    return {"analysis": msg.content}
