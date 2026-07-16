"""对比 老师(DeepSeek) vs 学生(微调后 Qwen2.5-7B) 的 RAG 问答质量。

    python scripts/compare_sft.py           # 需 .env 指向 DeepSeek + 隧道 localhost:8000 通

设计：两个模型答同一批题（同样的检索证据），评委统一用 DeepSeek（.env）打分，
保证公平。学生模型走 OpenAI 兼容接口（LLaMA-Factory api / vLLM 均可）。
"""
from __future__ import annotations

import os

from openai import OpenAI

from src.eval import metrics
from src.eval.dataset import EVAL_SET
from src.eval.rag_qa import _PROMPT, answer_question
from src.rag.pipeline import RagPipeline

STUDENT_BASE = os.getenv("STUDENT_BASE_URL", "http://localhost:8000/v1")
STUDENT_MODEL = os.getenv("STUDENT_MODEL", "qwen")
_student = OpenAI(base_url=STUDENT_BASE, api_key="EMPTY")


def student_answer(question: str, pipeline) -> dict:
    """学生模型：同样先检索，再用微调模型作答（与 rag_qa 同构 prompt）。"""
    hits = pipeline.retrieve(question, use_rerank=True)
    contexts = [h.get("text", "") for h in hits]
    ctx = "\n".join(f"- {c}" for c in contexts) or "（无）"
    resp = _student.chat.completions.create(
        model=STUDENT_MODEL, temperature=0.2,
        messages=[{"role": "user", "content": _PROMPT.format(context=ctx, question=question)}])
    return {"answer": resp.choices[0].message.content, "contexts": contexts}


def score(qa: dict, item: dict) -> dict:
    return {
        "faithfulness": metrics.faithfulness(qa["answer"], qa["contexts"]),
        "answer_correctness": metrics.answer_correctness(
            item["question"], qa["answer"], item["ground_truth"]),
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def main():
    pipeline = RagPipeline.from_dir()
    teacher_rows, student_rows = [], []
    for item in EVAL_SET:
        q = item["question"]
        t = score(answer_question(q, pipeline, use_rerank=True), item)   # 老师 DeepSeek
        s = score(student_answer(q, pipeline), item)                     # 学生 Qwen-ft
        teacher_rows.append(t)
        student_rows.append(s)
        print(f"  {q[:22]}… 老师corr={t['answer_correctness']} 学生corr={s['answer_correctness']}")

    print("\n| 模型 | faithfulness | answer_correctness |")
    print("|---|---|---|")
    for name, rows in (("老师 DeepSeek", teacher_rows), ("学生 Qwen2.5-7B-ft", student_rows)):
        print(f"| {name} | {mean([r['faithfulness'] for r in rows])} | "
              f"{mean([r['answer_correctness'] for r in rows])} |")


if __name__ == "__main__":
    main()
