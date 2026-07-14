"""统一的 LLM 调用入口。换模型只改 .env 的三个字段。

没有 API key 时 chat() 返回一个确定性 stub，让整套图/测试离线也能跑通——
真正接入时把 .env 填好即可，业务代码零改动。
"""
from __future__ import annotations

from functools import lru_cache

from src.config import LLMConfig


@lru_cache
def _client():
    from openai import OpenAI

    return OpenAI(base_url=LLMConfig.base_url, api_key=LLMConfig.api_key)


def chat(messages: list[dict], tools: list | None = None, temperature: float = 0.2):
    """返回 OpenAI ChatCompletion message 对象（含 .content / .tool_calls）。

    离线模式（无 key）返回一个仿造的对象，content 为占位文本，无 tool_calls。
    """
    if not LLMConfig.available():
        return _StubMessage(_stub_reply(messages))

    kwargs = dict(model=LLMConfig.model, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
    resp = _client().chat.completions.create(**kwargs)
    return resp.choices[0].message


def _stub_reply(messages: list[dict]) -> str:
    last = messages[-1]["content"] if messages else ""
    return f"[OFFLINE-STUB] 未配置 LLM_API_KEY，这是占位回答。收到的最后一条输入：{str(last)[:80]}"


class _StubMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None
