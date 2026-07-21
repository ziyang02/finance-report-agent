"""bge-reranker 精排。RAG 效果杠杆最大的一环：召回 20 -> 精排取 5。

阶段 E 进阶：用 FlagEmbedding 构造 (query, 正样本, 难负样本) 微调，
报告 context_precision 前后提升——这条直接对口 MLE"会训模型"。

离线/未装时按召回分数保序返回，保证流程可跑。
"""
from __future__ import annotations

import os
import warnings

from src.config import settings


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings()["rag"]["rerank_model"]
        self._model = None
        # CI / 离线：置 RAG_FAKE_EMBED=1 跳过加载，按召回分数保序（不下载模型）
        if os.getenv("RAG_FAKE_EMBED") == "1":
            return
        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.model_name, use_fp16=True)
        except Exception as exc:
            message = (
                f"无法加载 reranker 模型 {self.model_name!r}；将按召回分数保序，不执行真实重排。"
                "评测或正式运行请安装模型，或设置 RAG_STRICT_MODE=1 直接失败。"
            )
            if os.getenv("RAG_STRICT_MODE") == "1":
                raise RuntimeError(message) from exc
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            self._model = None

    @property
    def real(self) -> bool:
        return self._model is not None

    def rank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        """candidates: [{text, ...}] -> 重排后前 top_n 条（附 rerank_score）。"""
        if self._model is not None:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self._model.compute_score(pairs, normalize=True)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        else:
            ranked = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)
        return ranked[:top_n]
