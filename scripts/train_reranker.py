"""阶段 E：微调 bge-reranker-base（cross-encoder，二分类打分）。

    python scripts/train_reranker.py                     # 默认 2 epoch, batch 8
    python scripts/train_reranker.py --epochs 3 --batch 4

原理：reranker = AutoModelForSequenceClassification(num_labels=1)。
输入 (query, passage) 拼接后过模型出一个 logit 分数；训练目标是让
正样本对分数高、难负样本对分数低（BCEWithLogitsLoss，正=1 负=0）。

在 Apple Silicon 上走 MPS 加速；训练前后各评一次 dev（MRR / Acc@1），
提升幅度就是"微调有效"的直接证据。产出模型可被 FlagReranker 直接加载：
    Reranker(model_name="models/bge-reranker-ft")
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL = "models/bge-reranker-base"   # 278M 本地权重，M2 16GB 可训；v2-m3 留给云端
DATA_DIR = Path("data/train")
OUT_DIR = Path("models/bge-reranker-ft")
MAX_LEN = 384


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class PairDataset(Dataset):
    """把 {query, pos[], neg[]} 展开成 (query, passage, label) 对。"""

    def __init__(self, rows: list[dict]):
        self.items: list[tuple[str, str, float]] = []
        for r in rows:
            for p in r["pos"]:
                self.items.append((r["query"], p, 1.0))
            for n in r["neg"]:
                self.items.append((r["query"], n, 0.0))
        random.seed(42)
        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(tokenizer):
    def collate(batch):
        queries = [b[0] for b in batch]
        passages = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        enc = tokenizer(queries, passages, truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        return enc, labels
    return collate


@torch.no_grad()
def eval_ranking(model, tokenizer, rows: list[dict], device) -> dict:
    """对每个 dev query：给 pos+neg 打分，看 pos 是否排第一（Acc@1）/ 排名倒数（MRR）。"""
    model.eval()
    ranks = []
    for r in rows:
        passages = r["pos"] + r["neg"]
        enc = tokenizer([r["query"]] * len(passages), passages, truncation=True,
                        max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        scores = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        rank = int(np.argsort(-scores).tolist().index(0)) + 1  # 正样本固定在第0位
        ranks.append(rank)
    return {
        "acc@1": round(sum(r == 1 for r in ranks) / len(ranks), 3),
        "mrr": round(float(np.mean([1 / r for r in ranks])), 3),
        "n": len(ranks),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  base={BASE_MODEL}")

    train_rows = load_jsonl(DATA_DIR / "rerank_train.jsonl")
    dev_rows = load_jsonl(DATA_DIR / "rerank_dev.jsonl")
    print(f"train_queries={len(train_rows)}  dev_queries={len(dev_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.to(device)

    before = eval_ranking(model, tokenizer, dev_rows, device)
    print(f"[微调前] dev: {before}")

    ds = PairDataset(train_rows)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=make_collate(tokenizer))
    total_steps = len(dl) * args.epochs
    warmup = max(1, int(total_steps * 0.1))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(   # 线性 warmup + 线性衰减
        opt, lambda s: s / warmup if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"pairs={len(ds)}  steps={total_steps}  开始训练…")
    best_mrr = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (enc, labels) in enumerate(dl, 1):
            enc, labels = enc.to(device), labels.to(device)
            loss = loss_fn(model(**enc).logits.squeeze(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            if step % 20 == 0:
                print(f"  epoch{epoch+1} step {step}/{len(dl)} loss={running/20:.4f}")
                running = 0.0

        m = eval_ranking(model, tokenizer, dev_rows, device)
        print(f"[epoch {epoch+1}] dev: {m}")
        if m["mrr"] >= best_mrr:
            best_mrr = m["mrr"]
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
            print(f"  已保存最优模型 -> {OUT_DIR}")

    print(f"\n[微调前] {before}\n[微调后] dev best_mrr={best_mrr}")
    print(f"用法：configs/settings.yaml 的 rerank_model 改为 {OUT_DIR}，或评估里直接对比。")


if __name__ == "__main__":
    main()
