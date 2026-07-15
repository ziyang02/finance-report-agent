"""跟踪脚本：一步步看每个 Agent 节点产出了什么（学习用，不影响主流程）。

    python scripts/trace_run.py "贵州茅台 600519"

用 LangGraph 的 .stream() 在每个节点跑完后打印它写进黑板(state)的内容。
"""
from __future__ import annotations

import sys

from src.agents.graph import build_graph
from src.config import LLMConfig


def _short(s: str, n: int = 300) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + " …"


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "贵州茅台 600519"
    print(f"{'='*70}\n目标: {target}   |   LLM 真实可用: {LLMConfig.available()}\n{'='*70}")

    # 加载检索 pipeline（和 main.py 一样）
    try:
        from src.rag.pipeline import RagPipeline
        pipeline = RagPipeline.from_dir()
    except Exception as e:
        print(f"[无索引，用空检索占位] {e}")
        pipeline = None

    app = build_graph(pipeline)

    step = 0
    for update in app.stream({"target": target}):
        for node, out in update.items():
            step += 1
            print(f"\n{'─'*70}\n【第 {step} 步】节点: {node.upper()}\n{'─'*70}")

            if node == "planner":
                print("拆出的子问题:")
                for i, q in enumerate(out.get("subquestions", []), 1):
                    print(f"   {i}. {q}")

            elif node == "retriever":
                ev = out.get("evidence", [])
                cites = out.get("citations", [])
                print(f"检索了 {len(ev)} 个 query，建了 {len(cites)} 条全局引用:")
                for q in ev:
                    print(f"   • query: {_short(q['q'], 40)}  → 命中 {len(q['chunks'])} 段")
                print("引用表:")
                for c in cites:
                    print(f"   [{c['n']}] ({c['source']}) {_short(c['text'], 60)}")

            elif node == "analyst":
                print("分析(节选):\n   " + _short(out.get("analysis", ""), 500))

            elif node == "critic":
                crit = out.get("critique", {})
                mark = "✅ 通过" if crit.get("verdict") == "pass" else "🔁 打回重检索"
                print(f"质检结论: {mark}   (第 {out.get('revise_count')} 轮)")
                if crit.get("notes"):
                    print(f"意见: {crit['notes']}  ← 这句会变成 Retriever 下一轮的补充检索词")

            elif node == "writer":
                print("最终研报(开头):\n   " + _short(out.get("report", ""), 400))

    print(f"\n{'='*70}\n完成，共 {step} 步。注意看有没有出现「打回重检索」——那就是反思回路。\n{'='*70}")


if __name__ == "__main__":
    main()
