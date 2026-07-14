"""图能端到端跑通（离线 stub 模式），且反思回路会触发。"""
from src.agents.graph import build_graph, run_report
from src.data.pdf_loader import chunk_text
from src.rag.index import VectorIndex
from src.rag.pipeline import RagPipeline


def _pipeline():
    idx = VectorIndex()
    idx.build(chunk_text("贵州茅台毛利率约91%。" * 20, source="t", size=40, overlap=10))
    return RagPipeline(idx)


def test_graph_compiles():
    app = build_graph(_pipeline())
    assert app is not None


def test_end_to_end_produces_report():
    report = run_report("贵州茅台 600519", pipeline=_pipeline())
    assert isinstance(report, str)
    assert "引用来源" in report


def test_reflection_loop_triggers_offline():
    """离线 stub 下 Critic 第一轮判 revise，应回到 retriever 至少一次。"""
    app = build_graph(_pipeline())
    final = app.invoke({"target": "贵州茅台 600519"})
    # max_revise=2，离线逻辑第一轮 revise、第二轮 pass -> revise_count 应达到 2
    assert final["revise_count"] >= 2
