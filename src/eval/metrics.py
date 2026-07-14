"""LLM-as-Judge 指标（不依赖 embedding，只需一个 LLM，DeepSeek 即可跑）。

- faithfulness       忠实度/防幻觉：答案里被资料支撑的 claim 占比
- context_precision  上下文精度：检索到的资料里与问题相关的占比
- answer_correctness 答案正确性：答案与参考答案的一致程度

每个返回 [0,1] 浮点。离线(无 key)返回 None，跳过统计。
面试点：faithfulness = 把答案拆成原子 claim，逐条判断是否被 context entail。
"""
from __future__ import annotations

from src.config import LLMConfig
from src.llm import chat
from src.utils import parse_json

_FAITH = """把"答案"拆成若干原子事实陈述(claim)，逐条判断是否能被"资料"支撑(entailed)。
只输出JSON：{{"total": 整数, "supported": 整数}}
资料：
{context}
答案：
{answer}"""

_PREC = """判断下列每条"资料"是否与"问题"相关(能帮助回答)。
只输出JSON：{{"relevant": 相关条数(整数), "total": 总条数(整数)}}
问题：{question}
资料：
{context}"""

_CORR = """对照"参考答案"评估"实际答案"的正确性，给0到1的分数(数字越大越准确、越完整)。
只输出JSON：{{"score": 0到1的小数}}
问题：{question}
参考答案：{ground_truth}
实际答案：{answer}"""


def _judge(prompt: str) -> dict | None:
    if not LLMConfig.available():
        return None
    msg = chat([{"role": "user", "content": prompt}], temperature=0.0)
    return parse_json(msg.content, default={})


def faithfulness(answer: str, contexts: list[str]) -> float | None:
    r = _judge(_FAITH.format(context="\n".join(contexts) or "（无）", answer=answer))
    if not r or not r.get("total"):
        return None if r is None else 0.0
    return round(r["supported"] / r["total"], 3)


def context_precision(question: str, contexts: list[str]) -> float | None:
    if not contexts:
        return None if not LLMConfig.available() else 0.0
    body = "\n".join(f"- {c}" for c in contexts)
    r = _judge(_PREC.format(question=question, context=body))
    if not r or not r.get("total"):
        return None if r is None else 0.0
    return round(r["relevant"] / r["total"], 3)


def answer_correctness(question: str, answer: str, ground_truth: str) -> float | None:
    r = _judge(_CORR.format(question=question, ground_truth=ground_truth, answer=answer))
    if not r or "score" not in r:
        return None
    return round(float(r["score"]), 3)
