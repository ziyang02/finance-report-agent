"""检索 pipeline：召回 -> 重排 ->（阶段 E：Self-RAG 反思重检索）。

Retriever 节点直接调 retrieve()。Self-RAG 的"证据不足则改写 query 重检"逻辑，
在阶段 E 从这里或 Critic 分支接入。
"""
from __future__ import annotations

from src.config import settings
from src.rag.index import VectorIndex
from src.rag.reranker import Reranker


class RagPipeline:
    def __init__(self, index: VectorIndex, reranker: Reranker | None = None):
        cfg = settings()["rag"]
        self.index = index
        self.reranker = reranker or Reranker()
        self.k_recall = cfg["k_recall"]
        self.k_final = cfg["k_final"]

    def retrieve(self, query: str, use_rerank: bool = True) -> list[dict]:
        """use_rerank=False 时走基线：直接取召回 top-k_final（用于评估对比）。"""
        if not use_rerank:
            return self.index.search(query, k=self.k_final)
        recalled = self.index.search(query, k=self.k_recall)
        if not recalled:
            return []
        return self.reranker.rank(query, recalled, top_n=self.k_final)

    @classmethod
    def from_dir(cls, index_dir: str | None = None) -> "RagPipeline":
        index_dir = index_dir or settings()["rag"]["index_dir"]
        return cls(VectorIndex().load(index_dir))
