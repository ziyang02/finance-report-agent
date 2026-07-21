"""bge-m3 向量化封装。阶段 B 实现。

离线/未装 FlagEmbedding 时用一个确定性的 hash 伪向量，保证 index/pipeline 能跑通流程，
真接入时把 encode() 换成真模型即可（接口不变）。
"""
from __future__ import annotations

import hashlib
import os
import warnings

# 国内直连 huggingface.co 常常很慢/超时，默认走镜像（未手动设置时才生效）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np  # noqa: E402

from src.config import settings  # noqa: E402

_DIM_FALLBACK = 256


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings()["rag"]["embed_model"]
        self._model = None
        # CI / 离线：置 RAG_FAKE_EMBED=1 强制用伪向量，避免下载 ~2GB 模型
        if os.getenv("RAG_FAKE_EMBED") == "1":
            return
        try:
            # bge-m3 用专用类；其它 bge 系列（如 bge-small-zh，网速受限时的轻量选择）用通用类
            if "m3" in self.model_name.lower():
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(self.model_name, use_fp16=True)
            else:
                from FlagEmbedding import FlagModel

                self._model = FlagModel(self.model_name, use_fp16=True)
        except Exception as exc:
            message = (
                f"无法加载 embedding 模型 {self.model_name!r}；将回退到不具备语义能力的伪向量。"
                "评测或正式运行请安装模型，或设置 RAG_STRICT_MODE=1 直接失败。"
            )
            if os.getenv("RAG_STRICT_MODE") == "1":
                raise RuntimeError(message) from exc
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            self._model = None

    @property
    def real(self) -> bool:
        return self._model is not None

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        if self._model is not None:
            out = self._model.encode(texts)
            if isinstance(out, dict):  # BGEM3FlagModel 返回 {"dense_vecs": ...}
                out = out["dense_vecs"]
            vecs = np.asarray(out, dtype="float32")
        else:
            vecs = np.stack([self._fake_vec(t) for t in texts]).astype("float32")
        if normalize:
            vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        return vecs

    @staticmethod
    def _fake_vec(text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(_DIM_FALLBACK)
