"""Planner：把"分析某标的"拆成子问题清单。"""
from __future__ import annotations

import json

from src.agents.state import ReportState
from src.config import LLMConfig, settings
from src.llm import chat

_DEFAULT_SUBQS = ["财务健康度如何", "成长性如何", "估值是否合理", "主要风险有哪些"]

_PROMPT = """你是研报规划师。针对分析标的「{target}」，拆解出 3-5 个互不重叠的分析子问题，
覆盖财务健康度、成长性、估值、风险等维度。只输出 JSON 数组，如 ["...","..."]。"""


def planner_node(state: ReportState) -> dict:
    target = state["target"]
    if not LLMConfig.available():
        return {"subquestions": [f"{target}：{q}" for q in _DEFAULT_SUBQS], "revise_count": 0}

    msg = chat([{"role": "user", "content": _PROMPT.format(target=target)}])
    try:
        subqs = json.loads(msg.content)
    except Exception:
        subqs = _DEFAULT_SUBQS
    subqs = subqs[: settings()["agent"]["subquestion_max"]]
    return {"subquestions": subqs, "revise_count": 0}
