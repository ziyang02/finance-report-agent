"""研报 PDF -> 文本块。阶段 B 实现（pymupdf）。

版权提醒：PDF 仅个人 demo 用，README 别放来源、别商用。
"""
from __future__ import annotations

from src.config import settings


def load_pdf_chunks(pdf_path: str) -> list[dict]:
    cfg = settings()["rag"]
    size, overlap = cfg["chunk_size"], cfg["chunk_overlap"]
    try:
        import fitz  # pymupdf

        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
    except Exception:
        return [{"text": "（未装 pymupdf 或文件缺失，占位）", "source": pdf_path}]
    return chunk_text(text, source=pdf_path, size=size, overlap=overlap)


def chunk_text(text: str, source: str, size: int = 400, overlap: int = 50) -> list[dict]:
    """定长滑窗切块，overlap 防止把一句话切断丢语义。"""
    text = text.replace("\r", "")
    chunks, start, cid = [], 0, 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append({"id": f"{source}#{cid}", "text": piece, "source": source})
            cid += 1
        start += size - overlap
    return chunks
