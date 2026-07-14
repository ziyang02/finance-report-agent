"""通用小工具。"""
from __future__ import annotations

import json
import re


def parse_json(text: str, default=None):
    """从 LLM 输出里稳健地抽 JSON：容忍 ```json 代码块、前后噪声。"""
    if not text:
        return default
    # 去掉 ```json ... ``` 围栏
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate.strip())
    except (json.JSONDecodeError, TypeError):
        pass
    # 兜底：抓第一个 {...} 或 [...] 片段
    m = re.search(r"(\{.*\}|\[.*\])", candidate, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return default
