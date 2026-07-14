"""Critic：审查分析质量，输出 pass/revise + 具体 notes。触发条件边（反思回路核心）。

检查：①有无未被证据[编号]支撑的结论 ②遗漏的子问题 ③数据前后矛盾。
notes 会被 Retriever 当作补充检索 query，形成闭环。
"""
from __future__ import annotations

from src.agents.state import ReportState
from src.config import LLMConfig
from src.llm import chat
from src.utils import parse_json

_PROMPT = """你是严格的研报质检员。审查下面的分析，重点看：
1) 是否存在未被 [编号] 证据支撑的结论；
2) 是否遗漏了某个子问题；
3) 数据前后是否矛盾。

只输出 JSON（不要多余文字）：
{{"verdict": "pass" 或 "revise", "notes": "若 revise，用一句话说明最需要补充/核实什么；若 pass 则空字符串"}}

子问题：{subqs}

分析：
{analysis}"""


def critic_node(state: ReportState) -> dict:
    revise_count = state.get("revise_count", 0)

    if not LLMConfig.available():
        # 离线演示：第一轮判 revise（触发一次反思重检索），第二轮 pass
        crit = ({"verdict": "revise", "notes": "补充最新一期盈利能力与估值数据"}
                if revise_count == 0 else {"verdict": "pass", "notes": ""})
        return {"critique": crit, "revise_count": revise_count + 1}

    msg = chat([{"role": "user", "content": _PROMPT.format(
        subqs=state.get("subquestions", []), analysis=state.get("analysis", ""))}])
    crit = parse_json(msg.content, default={"verdict": "pass", "notes": "解析失败，默认通过"})
    if crit.get("verdict") not in ("pass", "revise"):
        crit["verdict"] = "pass"
    return {"critique": crit, "revise_count": revise_count + 1}
