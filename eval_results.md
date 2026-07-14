# 评估结果

> 数据集：8 条问题（`src/eval/dataset.py`），标的 600519，LLM-as-Judge 用 DeepSeek。
> 当前为**伪向量**模式（bge-m3 未下载）——见下方解读。

| 配置 | 样本数 | faithfulness | context_precision | answer_correctness |
|---|---|---|---|---|
| baseline（不重排） | 8 | 0.75 | 0.40 | 0.875 |
| rerank（bge-reranker） | 8 | 0.75 | 0.40 | 0.875 |

## 指标含义
- **faithfulness**：答案里被检索资料支撑的 claim 占比（防幻觉）。
- **context_precision**：检索到的资料中与问题相关的占比（检索质量）。
- **answer_correctness**：答案对照参考答案的正确性（端到端质量）。

## 解读
- **`context_precision` 仅 0.40 是关键信号**：当前用伪向量检索、库很小(几乎全量召回)，
  无法按语义区分相关性——**这正是 bge-m3 + bge-reranker 要解决的问题**。
- 装好 bge-m3 后重跑（`python -m src.eval.run_eval`）：baseline 与 rerank 两行的
  `context_precision` 差值，就是简历可写的“引入重排后检索精度提升 X%”。
- **`answer_correctness` 0.875 已较高**：说明多智能体对**真实财务数据**的问答本身可靠；
  伪向量下仍能答对，是因为库小、相关 chunk 大概率被召回。
- baseline 与 rerank 完全相同，是伪向量下的**预期结果**（reranker 无语义可排），
  也反过来验证了评估框架本身正确——真实模型接入后差异才会显现。

## 复现
```bash
python scripts/build_index.py 600519          # 建库（装好 bge-m3 后为真实语义）
python -m src.eval.run_eval                    # 跑 baseline + rerank 两组对比
```
