"""阶段 E：构造 SFT 数据集（蒸馏 DeepSeek 的金融 RAG 问答能力）。

    python scripts/make_sft_dataset.py       # 本地跑，需要 LLM_API_KEY + 真实索引

思路（蒸馏 distillation）：
  复用 reranker 数据集里的 query（LLM 生成的自然问题）
  -> 检索真实证据 -> 让 DeepSeek（老师模型）生成"仅基于资料、带出处约束"的回答
  -> (instruction, input, output) 三元组，Alpaca 格式，供 LLaMA-Factory LoRA 微调。

产出 sft/data/finance_rag_qa.json；云端训练与部署见 sft/README.md。
SFT 后的模型经 vLLM 起 OpenAI 兼容服务，改 .env 的 LLM_BASE_URL 即可无缝替换 DeepSeek。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import LLMConfig
from src.eval.rag_qa import _PROMPT, answer_question
from src.rag.pipeline import RagPipeline

TRAIN_QUERIES = Path("data/train/rerank_train.jsonl")
OUT = Path("sft/data/finance_rag_qa.json")
INSTRUCTION = "你是金融数据问答助手。仅根据提供的资料回答问题，不要使用资料外的知识；资料不足就回答\"资料不足\"。"


def main():
    assert LLMConfig.available(), "需要 LLM_API_KEY（DeepSeek 当老师模型）"
    assert TRAIN_QUERIES.exists(), "先跑 scripts/make_rerank_dataset.py 生成 query"
    pipeline = RagPipeline.from_dir()

    queries = [json.loads(ln)["query"] for ln in
               TRAIN_QUERIES.read_text(encoding="utf-8").splitlines() if ln]
    print(f"{len(queries)} 个 query，逐个 检索->老师模型作答…")

    samples = []
    for i, q in enumerate(queries, 1):
        try:
            qa = answer_question(q, pipeline, use_rerank=True)  # 老师：检索+DeepSeek 作答
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] 失败({type(e).__name__})，跳过")
            continue
        ctx_block = "\n".join(f"- {c}" for c in qa["contexts"]) or "（无）"
        samples.append({
            "instruction": INSTRUCTION,
            # input 与推理时的 rag_qa prompt 同构，保证训练/推理分布一致
            "input": _PROMPT.format(context=ctx_block, question=q),
            "output": qa["answer"],
        })
        if i % 20 == 0:
            print(f"  {i}/{len(queries)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"共 {len(samples)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
