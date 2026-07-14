"""离线建库入口：拉数据 -> 切块 -> bge-m3 编码 -> FAISS 落盘。

    python scripts/build_index.py 600519 000858
"""
from __future__ import annotations

import sys

from src.config import settings
from src.data.akshare_loader import load_company_docs
from src.data.pdf_loader import chunk_text
from src.rag.index import VectorIndex


def build(codes: list[str]) -> None:
    cfg = settings()["rag"]
    chunks: list[dict] = []
    for code in codes:
        for doc in load_company_docs(code):
            chunks += chunk_text(doc["text"], source=doc["source"],
                                 size=cfg["chunk_size"], overlap=cfg["chunk_overlap"])
    print(f"共 {len(chunks)} 个 chunk，开始建索引…")
    idx = VectorIndex()
    idx.build(chunks)
    idx.save(cfg["index_dir"])
    print(f"索引已保存到 {cfg['index_dir']}（embedder.real={idx.embedder.real}）")


if __name__ == "__main__":
    codes = sys.argv[1:] or ["600519", "000858"]
    build(codes)
