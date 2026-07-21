from src.data.pdf_loader import chunk_text
from src.eval import metrics
from src.eval.dataset import EVAL_SET
from src.eval.rag_qa import answer_question
from src.eval.run_eval import eval_config, render_table
from src.rag.index import VectorIndex
from src.rag.pipeline import RagPipeline


def _pipeline():
    idx = VectorIndex()
    idx.build(chunk_text("贵州茅台2025年营业总收入约1720.54亿元。" * 8,
                         source="600519-摘要", size=50, overlap=10))
    return RagPipeline(idx)


def test_dataset_has_ground_truth():
    assert len(EVAL_SET) >= 5
    assert all(item["question"] and item["ground_truth"] for item in EVAL_SET)


def test_eval_dataset_has_15_unique_questions():
    questions = [item["question"] for item in EVAL_SET]
    assert len(questions) == 15
    assert len(questions) == len(set(questions))


def test_rag_qa_returns_contexts():
    qa = answer_question(EVAL_SET[0]["question"], _pipeline())
    assert "answer" in qa and isinstance(qa["contexts"], list)


def test_metrics_offline_return_none():
    # 离线(测试无 key)时指标应返回 None，被均值统计跳过而不报错
    assert metrics.faithfulness("答案", ["资料"]) is None
    assert metrics.answer_correctness("q", "a", "gt") is None


def test_eval_config_and_table_render():
    res = eval_config("baseline", _pipeline(), use_rerank=False)
    assert res["n"] == len(EVAL_SET)
    table = render_table([res])
    assert "faithfulness" in table and "| baseline |" in table
