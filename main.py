"""端到端入口：生成一份研报。

    python main.py "贵州茅台 600519"

无索引/无 key 时走占位流程，也能看到完整的多智能体链路跑通。
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.agents.graph import run_report


def _load_pipeline():
    try:
        from src.rag.pipeline import RagPipeline

        return RagPipeline.from_dir()
    except Exception:
        print("[提示] 未找到索引，先跑 scripts/build_index.py；本次用空检索占位。")
        return None


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "贵州茅台 600519"
    report = run_report(target, pipeline=_load_pipeline())

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    safe = target.replace(" ", "_").replace("/", "_")
    path = out_dir / f"report_{safe}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\n已生成：{path}\n")
    print(report[:800])


if __name__ == "__main__":
    main()
