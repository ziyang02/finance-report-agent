"""Writer：整合成结构化研报（摘要->基本面->估值->风险->结论 + 引用列表）。

引用列表用 state.citations 的全局编号渲染，与 Analyst 正文里的 [n] 一一对应。
"""
from __future__ import annotations

from src.agents.state import ReportState
from src.config import LLMConfig
from src.llm import chat

_PROMPT = """你是研报撰稿人。把下列分析整理成结构化研报，包含这五节：
一、投资摘要（3-4 句结论先行） 二、基本面 三、估值 四、风险提示 五、投资结论。

严格要求：
- 直接输出研报正文，第一行就是标题「# {target} 研究报告」，不要任何开场白/客套/结尾语/分隔线；
- 不要编造分析师、机构、日期等占位信息；
- 保留分析中的 [编号] 引用；只用分析里已有的信息，不要新增未引用的数字。

标的：{target}
分析：
{analysis}"""


def _render_refs(citations: list[dict]) -> str:
    if not citations:
        return "（无）"
    return "\n".join(f"[{c['n']}] {c['source']}" for c in citations)


def writer_node(state: ReportState) -> dict:
    target = state["target"]
    analysis = state.get("analysis", "")
    refs = _render_refs(state.get("citations", []))

    if not LLMConfig.available():
        report = (f"# {target} 研报（离线占位）\n\n## 一、投资摘要\n（占位）\n\n"
                  f"## 基本面 / 估值 / 风险 / 结论\n{analysis}\n\n## 引用来源\n{refs}\n")
        return {"report": report}

    msg = chat([{"role": "user", "content": _PROMPT.format(target=target, analysis=analysis)}])
    return {"report": f"{msg.content}\n\n## 引用来源\n{refs}\n"}
