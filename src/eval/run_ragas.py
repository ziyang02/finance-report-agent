"""RAGAS 评估入口（可选/进阶）。

主力评估用轻量自研 `src/eval/run_eval.py`（LLM-as-Judge，只需 DeepSeek，无需 embedding）。
本文件是标准库 RAGAS 的接入，供想要业界通用指标口径时使用——但它的 answer_relevancy
需要 embedding 模型，且对 langchain 版本敏感，装好 bge-m3 与 ragas 后再启用。

指标：faithfulness(忠实度/防幻觉)、answer_relevancy、context_precision、context_recall。
面试点：faithfulness = 把答案拆成 claims，逐条判断是否被 context 支撑。
"""
from __future__ import annotations

from src.eval.dataset import EVAL_SET


def run():
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception:
        print("[提示] 未装 ragas：pip install -e '.[eval]'。当前仅打印评测集规模。")
        print(f"评测集条数：{len(EVAL_SET)}")
        return

    # TODO(阶段 F): 用 RagPipeline + graph 生成 answer/contexts，再喂给 ragas
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in EVAL_SET:
        rows["question"].append(item["question"])
        rows["answer"].append("（TODO: 接入系统输出）")
        rows["contexts"].append(item["contexts"])
        rows["ground_truth"].append(item["ground_truth"])

    ds = Dataset.from_dict(rows)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy,
                                   context_precision, context_recall])
    print(result)


if __name__ == "__main__":
    run()
