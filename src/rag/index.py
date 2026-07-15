"""FAISS 索引封装。先用 IndexFlatIP（精确、内积=归一化后的 cos）；
数据量大再换 IVF/HNSW（面试点：nlist/nprobe 的召回-速度权衡）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# macOS 下 faiss 与 torch 各带一份 OpenMP 运行时，同进程共存会段错误（exit 139）。
# 两个都要：允许重复运行时 + 单线程避免两套线程池打架。必须在 import faiss 前设置。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss  # noqa: E402

from src.rag.embedder import Embedder


class VectorIndex:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []  # [{id, text, source}]

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        vecs = self.embedder.encode([c["text"] for c in chunks], normalize=True)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, k: int) -> list[dict]:
        q = self.embedder.encode([query], normalize=True)
        scores, idx = self.index.search(q, min(k, len(self.chunks)))
        out = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            c = dict(self.chunks[i])
            c["score"] = float(score)
            out.append(c)
        return out

    def save(self, index_dir: str) -> None:
        p = Path(index_dir)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "faiss.index"))
        (p / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, index_dir: str) -> "VectorIndex":
        p = Path(index_dir)
        self.index = faiss.read_index(str(p / "faiss.index"))
        self.chunks = json.loads((p / "chunks.json").read_text(encoding="utf-8"))
        return self
