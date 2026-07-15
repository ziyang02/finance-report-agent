"""阶段 E：构造 reranker 微调数据集 (query, pos, hard_neg) 三元组。

    python scripts/make_rerank_dataset.py            # 需要 LLM_API_KEY + 已建好的真实索引

流程：
  1. 每个 chunk 让 LLM 生成 N 个"这段资料能回答的自然问题" -> (query, pos) 对；
  2. 拿 query 去 FAISS 召回 top-k，剔除正样本同源的 -> 难负样本（长得像但答不上）；
  3. 按 chunk 划分 train/dev（防泄漏），输出 FlagEmbedding 标准格式 JSONL：
     {"query": str, "pos": [str], "neg": [str, ...]}

难负样本是重排微调的关键：随机负样本太容易，模型学不到"区分相似但不相关"。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import LLMConfig, settings
from src.llm import chat
from src.rag.index import VectorIndex
from src.utils import parse_json

N_QUERIES_PER_CHUNK = 3
N_HARD_NEG = 4
DEV_RATIO = 0.15
OUT_DIR = Path("data/train")

_GEN_PROMPT = """下面是一段A股财务资料。请生成 {n} 个自然、多样的中文问题，要求：
- 每个问题都能且只能用这段资料回答（问题里要带上公司代码或时间等限定，避免歧义）；
- 覆盖不同角度（如具体数值、增长趋势、水平判断），口语化一点，像真实用户提问。
只输出 JSON 数组，如 ["问题1", "问题2", "问题3"]。

资料：
{text}"""


def gen_queries(text: str, n: int) -> list[str]:
    msg = chat([{"role": "user", "content": _GEN_PROMPT.format(n=n, text=text)}],
               temperature=0.7)
    qs = parse_json(msg.content, default=[])
    return [q for q in qs if isinstance(q, str) and len(q) >= 5][:n]


def mine_hard_negatives(index: VectorIndex, query: str, pos_source: str,
                        top_n: int) -> list[str]:
    """召回相似 chunk，剔除与正样本同源的，剩下即难负样本。"""
    hits = index.search(query, k=top_n + 4)
    return [h["text"] for h in hits if h.get("source") != pos_source][:top_n]


def main():
    assert LLMConfig.available(), "需要配置 LLM_API_KEY 才能生成 query"
    index = VectorIndex().load(settings()["rag"]["index_dir"])
    assert index.embedder.real, "需要真实 bge-m3（伪向量挖出的负样本没有意义）"
    chunks = index.chunks
    print(f"语料 {len(chunks)} 个 chunk，每个生成 {N_QUERIES_PER_CHUNK} 个 query…")

    samples = []  # [(chunk_i, {query, pos, neg})]
    for i, c in enumerate(chunks):
        try:
            queries = gen_queries(c["text"], N_QUERIES_PER_CHUNK)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] query 生成失败({type(e).__name__})，跳过")
            continue
        for q in queries:
            negs = mine_hard_negatives(index, q, c["source"], N_HARD_NEG)
            if negs:
                samples.append((i, {"query": q, "pos": [c["text"]], "neg": negs}))
        print(f"  [{i+1}/{len(chunks)}] {c['source']}: +{len(queries)} query")

    # 按 chunk 划分 train/dev，同一 chunk 的 query 不跨集（防泄漏）
    chunk_ids = sorted({i for i, _ in samples})
    random.seed(42)
    dev_ids = set(random.sample(chunk_ids, max(1, int(len(chunk_ids) * DEV_RATIO))))
    train = [s for i, s in samples if i not in dev_ids]
    dev = [s for i, s in samples if i in dev_ids]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("rerank_train.jsonl", train), ("rerank_dev.jsonl", dev)):
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\ntrain={len(train)} dev={len(dev)} -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
