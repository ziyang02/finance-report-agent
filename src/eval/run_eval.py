"""评估入口：对 EVAL_SET 跑 RAG-QA，用 LLM-as-Judge 打分，输出对比表到 eval_results.md。

    python -m src.eval.run_eval                # 跑全部配置对比
    python -m src.eval.run_eval --config rerank

对比配置：
  baseline   召回后直接取 top-k（不重排）
  rerank     召回 k_recall -> bge-reranker-v2-m3 精排 -> top-k
  rerank_ft  召回 -> 微调后的 reranker（models/bge-reranker-ft，阶段 E 产物）
（rerank_ft 行仅在训练产物存在时出现；伪向量模式下各组接近，结果不可比）
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

from src.eval import metrics
from src.eval.dataset import EVAL_SET
from src.eval.rag_qa import answer_question
from src.rag.pipeline import RagPipeline


def _mean(xs: list) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(statistics.mean(vals), 3) if vals else None


def eval_config(name: str, pipeline, use_rerank: bool) -> dict:
    rows = []
    for item in EVAL_SET:
        qa = answer_question(item["question"], pipeline, use_rerank=use_rerank)
        rows.append({
            "faithfulness": metrics.faithfulness(qa["answer"], qa["contexts"]),
            "context_precision": metrics.context_precision(item["question"], qa["contexts"]),
            "answer_correctness": metrics.answer_correctness(
                item["question"], qa["answer"], item["ground_truth"]),
        })
        print(f"  [{name}] {item['question'][:20]}… "
              f"faith={rows[-1]['faithfulness']} corr={rows[-1]['answer_correctness']}")
    return {
        "config": name,
        "n": len(rows),
        "faithfulness": _mean([r["faithfulness"] for r in rows]),
        "context_precision": _mean([r["context_precision"] for r in rows]),
        "answer_correctness": _mean([r["answer_correctness"] for r in rows]),
    }


def render_table(results: list[dict]) -> str:
    head = ("| 配置 | 样本数 | faithfulness | context_precision | answer_correctness |\n"
            "|---|---|---|---|---|\n")
    rows = "".join(
        f"| {r['config']} | {r['n']} | {r['faithfulness']} | "
        f"{r['context_precision']} | {r['answer_correctness']} |\n"
        for r in results)
    return head + rows


def main():
    only = None
    if "--config" in sys.argv:
        only = sys.argv[sys.argv.index("--config") + 1]

    pipeline = RagPipeline.from_dir()
    configs = [("baseline", pipeline, False), ("rerank", pipeline, True)]

    ft_dir = Path("models/bge-reranker-ft")   # 阶段 E 微调产物，存在才对比
    if ft_dir.exists():
        from src.rag.reranker import Reranker

        ft_pipe = RagPipeline(pipeline.index, reranker=Reranker(model_name=str(ft_dir)))
        configs.append(("rerank_ft", ft_pipe, True))

    if only:
        configs = [c for c in configs if c[0] == only]

    results = [eval_config(name, pipe, use_rerank) for name, pipe, use_rerank in configs]
    table = render_table(results)

    out = Path("eval_results.md")
    analysis = (
        "\n## 指标含义\n"
        "- **faithfulness**：答案里被检索资料支撑的 claim 占比（防幻觉）。\n"
        "- **context_precision**：检索到的资料中与问题相关的占比（检索质量）。\n"
        "- **answer_correctness**：答案对照参考答案的正确性（端到端质量）。\n"
        "\n## 配置说明\n"
        "- **baseline**：bge-m3 召回后直接取 top-k，不重排。\n"
        "- **rerank**：召回 20 条 -> 预训练 bge-reranker-v2-m3 精排取 5。\n"
        "- **rerank_ft**：召回 20 条 -> 领域微调后的 bge-reranker-base（阶段 E，"
        "训练数据为 LLM 生成 query + FAISS 难负样本挖掘）。\n"
        "\n## 解读\n"
        "- baseline vs rerank 的差值 = 引入重排的收益；"
        "rerank vs rerank_ft 的差值 = 领域微调的收益。\n"
        "- 评测集仅 8 问且同源语料较小，数字有波动，趋势比绝对值更有意义。\n"
    )
    out.write_text("# 评估结果\n\n" + table + analysis, encoding="utf-8")
    print("\n" + table)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
