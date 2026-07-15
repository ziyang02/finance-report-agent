# 阶段 E（下）：LLaMA-Factory SFT + vLLM 部署（云端 GPU）

> 本地 Mac 跑不了 vLLM（需 CUDA/Linux），7B SFT 也不现实。
> 本目录是完整的云端材料包：数据在本地造好，训练/部署在云端一条龙。

## 故事线（简历/面试怎么讲）

用 DeepSeek（老师）蒸馏出金融 RAG 问答数据，LoRA 微调 Qwen2.5-7B（学生），
vLLM 起 OpenAI 兼容服务后**主系统零改动切换**（改 `.env` 即可）——
把"任意 OpenAI 兼容端点"这个架构预留真正用上，收益是成本/延迟/数据可控。

## 0. 本地先造数据（已可跑）

```bash
python scripts/make_rerank_dataset.py   # 若还没跑过（产出 query）
python scripts/make_sft_dataset.py      # 蒸馏 -> sft/data/finance_rag_qa.json
```

## 1. 租云端 GPU

AutoDL / 仙宫云等，单卡 **RTX 4090 24GB** 即够（7B LoRA bf16）。
镜像选 PyTorch 2.x + CUDA 12.x。把本仓库（至少 `sft/` 目录）传上去。

## 2. 装 LLaMA-Factory 并训练

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]"

# 国内下载 Qwen 权重走 modelscope（可选）
export USE_MODELSCOPE_HUB=1

cd /path/to/finance-report-agent
llamafactory-cli train sft/qwen2_5_7b_lora_sft.yaml
# 4090 上约 10-20 分钟（数据几百条、3 epoch）；loss 曲线在 saves/ 下
```

## 3. 合并 LoRA 并用 vLLM 起服务

```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path saves/qwen2.5-7b-finance-lora \
  --template qwen --finetuning_type lora \
  --export_dir models/qwen2.5-7b-finance

pip install vllm
vllm serve models/qwen2.5-7b-finance --port 8000 --max-model-len 4096
```

## 4. 主系统无缝切换 + 量化收益

本地项目 `.env` 改三行（把 <ip> 换成云端公网/内网穿透地址）：

```
LLM_BASE_URL=http://<ip>:8000/v1
LLM_API_KEY=EMPTY-but-not-empty     # vLLM 默认不校验，非空即可
LLM_MODEL=models/qwen2.5-7b-finance
```

然后跑同一套评估，对比 DeepSeek vs 微调后 Qwen 的三项指标：

```bash
python -m src.eval.run_eval
```

| 对比项 | 说明 |
|---|---|
| answer_correctness | 微调模型 vs 老师模型，差距多少 |
| faithfulness | 蒸馏时的"仅基于资料"约束是否被学到 |
| 成本/延迟 | API 计费 -> 自托管；实测 tokens/s |

## 常见坑

- vLLM 显存不够：加 `--gpu-memory-utilization 0.9` 或 `--max-model-len 2048`。
- 训练 OOM：`per_device_train_batch_size: 1` + `gradient_accumulation_steps: 16`。
- 数据太少（<300 条）：回本地把 `build_index.py` 的股票池扩大、
  `make_rerank_dataset.py` 的 `N_QUERIES_PER_CHUNK` 调大再蒸馏。
