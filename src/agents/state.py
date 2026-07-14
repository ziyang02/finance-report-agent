"""LangGraph 共享状态（黑板）。各节点读写这个 TypedDict。

反思回路里 evidence/citations 每轮**整体替换**（不累加），所以不用 operator.add reducer——
LangGraph 默认后写覆盖，正好符合"重新检索得到新证据"的语义。
"""
from __future__ import annotations

from typing import TypedDict


class ReportState(TypedDict, total=False):
    target: str                 # "贵州茅台 600519" 或行业名
    subquestions: list[str]     # Planner 产出
    evidence: list[dict]        # [{q, chunks:[{id,text,source,score}]}] 每轮重检索覆盖
    citations: list[dict]       # 全局引用表 [{n, source, text}]，Analyst/Writer 共用同一编号
    analysis: str               # Analyst 产出（结论挂 [n] 引用）
    critique: dict              # {"verdict": "pass"|"revise", "notes": str}
    revise_count: int           # 反思回路计数，防死循环
    report: str                 # Writer 最终研报
