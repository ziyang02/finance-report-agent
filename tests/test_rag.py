from src.data.pdf_loader import chunk_text
from src.rag.index import VectorIndex
from src.rag.pipeline import RagPipeline
from src.rag.reranker import Reranker


def _sample_chunks():
    text = ("贵州茅台毛利率约91%，白酒龙头。" * 10 +
            "五粮液毛利率约76%，行业第二。" * 10)
    return chunk_text(text, source="test", size=40, overlap=10)


def test_chunking_overlap():
    chunks = chunk_text("a" * 100, source="t", size=40, overlap=10)
    assert len(chunks) >= 3
    assert all(c["source"] == "t" for c in chunks)


def test_index_build_and_search():
    idx = VectorIndex()
    idx.build(_sample_chunks())
    hits = idx.search("茅台毛利率", k=3)
    assert len(hits) == 3
    assert "score" in hits[0]


def test_pipeline_recall_then_rerank():
    idx = VectorIndex()
    idx.build(_sample_chunks())
    pipe = RagPipeline(idx, Reranker())
    out = pipe.retrieve("茅台")
    assert 0 < len(out) <= pipe.k_final
