"""集中读取环境变量 + settings.yaml。其它模块统一从这里拿配置。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@lru_cache
def settings() -> dict:
    with open(ROOT / "configs" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LLMConfig:
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    @classmethod
    def available(cls) -> bool:
        """没配 key 时走离线 stub，保证脚手架/CI 能跑通。"""
        return bool(cls.api_key) and cls.api_key not in ("", "EMPTY", "sk-xxxxxxxxxxxxxxxx")
