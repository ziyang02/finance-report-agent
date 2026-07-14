from src.agents.graph import build_graph
from src.data.pdf_loader import chunk_text
from src.rag.index import VectorIndex
from src.rag.pipeline import RagPipeline
from src.utils import parse_json


def _pipeline():
    idx = VectorIndex()
    idx.build(chunk_text("贵州茅台毛利率约91%。" * 20, source="600519", size=40, overlap=10))
    return RagPipeline(idx)


def test_parse_json_plain():
    assert parse_json('{"verdict":"pass","notes":""}')["verdict"] == "pass"


def test_parse_json_fenced():
    text = '这是分析：\n```json\n{"verdict": "revise", "notes": "补数据"}\n```\n'
    assert parse_json(text)["notes"] == "补数据"


def test_parse_json_default_on_garbage():
    assert parse_json("no json here", default={"x": 1}) == {"x": 1}


def test_citation_registry_is_global_and_deduped():
    """所有引用编号唯一、连续，同一 source 不重复占号。"""
    app = build_graph(_pipeline())
    final = app.invoke({"target": "贵州茅台 600519"})
    ns = [c["n"] for c in final["citations"]]
    sources = [c["source"] for c in final["citations"]]
    assert ns == list(range(1, len(ns) + 1))       # 连续编号
    assert len(sources) == len(set(sources))       # source 去重


def test_reflection_adds_supplementary_retrieval():
    """离线下 Critic 首轮 revise，Retriever 应据 notes 多做一次补充检索。"""
    app = build_graph(_pipeline())
    final = app.invoke({"target": "贵州茅台 600519"})
    # 4 个子问题 + 1 条反思补充检索 = 5 个证据块
    assert len(final["evidence"]) == len(final["subquestions"]) + 1
