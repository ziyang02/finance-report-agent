# 评估结果

| 配置 | 样本数 | faithfulness | context_precision | answer_correctness |
|---|---|---|---|---|
| baseline | 15 | 0.8 | 0.333 | 0.793 |
| rerank | 15 | 0.644 | 0.28 | 0.767 |
| rerank_ft | 15 | 0.8 | 0.347 | 0.793 |

## 指标含义
- **faithfulness**：答案里被检索资料支撑的 claim 占比（防幻觉）。
- **context_precision**：检索到的资料中与问题相关的占比（检索质量）。
- **answer_correctness**：答案对照参考答案的正确性（端到端质量）。

## 配置说明
- **baseline**：bge-m3 召回后直接取 top-k，不重排。
- **rerank**：召回 20 条 -> 预训练 bge-reranker-v2-m3 精排取 5。
- **rerank_ft**：召回 20 条 -> 领域微调后的 bge-reranker-base（阶段 E，训练数据为 LLM 生成 query + FAISS 难负样本挖掘）。

## 解读
- baseline vs rerank 的差值 = 引入重排的收益；rerank vs rerank_ft 的差值 = 领域微调的收益。
- 评测集包含 17 家公司的 15 道题且规模较小，数字有波动，趋势比绝对值更有意义。
- SFT 数据由 DeepSeek 蒸馏，当前评委也使用 DeepSeek，可能存在同源模型偏好；结果仅用于内部配置对比。
- 机器可读汇总见 [`artifacts/eval_summary.json`](artifacts/eval_summary.json)。
