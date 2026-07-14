"""阶段 A 演示：手写一轮 Function Calling loop，理解裸协议（不套框架）。

跑法：
    cp .env.example .env   # 填入 LLM_API_KEY
    python demo_function_calling.py

没填 key 也能跑：会走离线 stub，直接演示"手动调用工具"这一步的机制。
"""
from __future__ import annotations

import json

from src.config import LLMConfig
from src.llm import chat
from src.tools.finance_tools import TOOLS_SCHEMA, dispatch


def run(question: str, max_turns: int = 5) -> str:
    messages = [
        {"role": "system", "content": "你是金融分析助手。需要数据时调用工具，不要编造数字。"},
        {"role": "user", "content": question},
    ]

    for turn in range(max_turns):
        msg = chat(messages, tools=TOOLS_SCHEMA)

        # 模型没要求调工具 -> 拿到最终答案
        if not getattr(msg, "tool_calls", None):
            return msg.content

        # 把 assistant 的 tool_calls 记入对话历史
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        # 逐个执行工具，结果回填（关键：执行权在我们手里，模型只出意图）
        for tc in msg.tool_calls:
            result = dispatch(tc.function.name, tc.function.arguments)
            print(f"  [turn {turn}] 调用 {tc.function.name}({tc.function.arguments}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "[达到最大轮次上限]"


def _offline_demo():
    """无 key 时，直接演示工具本身能跑（Function Calling 的"执行"这一半）。"""
    from src.tools.finance_tools import dispatch as d

    print("离线模式：直接演示工具执行（配置 LLM_API_KEY 后可看模型自主调用）")
    print("  get_financials('600519') ->", d("get_financials", {"code": "600519"}))
    print("  calc_metrics ->", d("calc_metrics",
                                  {"revenue": 1000, "net_profit": 340, "equity": 1000}))


if __name__ == "__main__":
    if LLMConfig.available():
        print("问题：贵州茅台(600519)去年 ROE 和毛利率大概多少？值得买吗？\n")
        print("\n最终回答：\n", run("贵州茅台(600519)去年 ROE 和毛利率大概多少？值得买吗？"))
    else:
        _offline_demo()
