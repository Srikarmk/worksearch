"""Fine-tune bge-small on (query, chunk, hard negative) triples from the training repos only.

Positives:
- commit subjects from the training repos: for each file the commit changed, that file's chunk that best
  matches the subject by BM25 (the commit doesn't say which part of the file it was about);
- synthetic queries (generate.py), each with the chunk it was written for.
Hard negative: the best BM25 chunk from any other file, so the model must learn more than word overlap.
Loss: in-batch negatives (MultipleNegativesRankingLoss), no duplicate texts in a batch.

    uv run python -m worksearch.train --name e1-lr2e-5 --epochs 1 --lr 2e-5 [--no-synthetic] [--no-commits]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .data import split_of
from .search import BASE, BM25, QUERY_PREFIX, load_chunks


def triples(use_commits: bool, use_synthetic: bool, seed: int = 0) -> list[dict]:
    chunks = load_chunks()
    idx = {c["id"]: i for i, c in enumerate(chunks)}
    fkey = [f"{c['repo']}/{c['path']}" for c in chunks]
    by_file: dict[str, list[int]] = {}
    for i, k in enumerate(fkey):
        by_file.setdefault(k, []).append(i)
    bm = BM25([c["text"] for c in chunks])
    out = []

    def negative(s: np.ndarray, avoid: set[str]) -> int | None:
        for j in np.argsort(-s)[:50]:
            if fkey[j] not in avoid:
                return int(j)
        return None

    if use_commits:
        for q in map(json.loads, open("data/queries.jsonl")):
            if q["split"] != "train":
                continue
            s = bm.scores(q["query"])
            neg = negative(s, set(q["files"]))
            for f in q["files"][:3]:
                pos = max(by_file[f], key=lambda i: s[i])
                out.append({"q": q["query"], "pos": pos, "neg": neg, "src": "commit"})
    if use_synthetic and Path("data/synthetic.jsonl").exists():
        for r in map(json.loads, open("data/synthetic.jsonl")):
            if split_of(r["repo"]) != "train" or r["chunk"] not in idx:
                continue
            pos = idx[r["chunk"]]
            neg = negative(bm.scores(r["query"]), {fkey[pos]})
            out.append({"q": r["query"], "pos": pos, "neg": neg, "src": "synthetic"})
    random.Random(seed).shuffle(out)
    texts = [c["text"] for c in chunks]
    return [{"anchor": QUERY_PREFIX + t["q"], "positive": texts[t["pos"]],
             "negative": texts[t["neg"] if t["neg"] is not None else random.Random(t["pos"]).randrange(len(texts))],
             "src": t["src"]} for t in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--no-commits", action="store_true")
    a = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import (BatchSamplers,
                                                                          SentenceTransformerTrainingArguments)

    rows = triples(not a.no_commits, not a.no_synthetic)
    src = {s: sum(r["src"] == s for r in rows) for s in ("commit", "synthetic")}
    print(f"{len(rows)} training triples {src}", flush=True)
    ds = Dataset.from_list([{k: r[k] for k in ("anchor", "positive", "negative")} for r in rows])
    model = SentenceTransformer(BASE, local_files_only=True)
    model.max_seq_length = 512
    out = Path("models") / a.name
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, warmup_ratio=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no", logging_steps=20, report_to="none", seed=0)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                               loss=MultipleNegativesRankingLoss(model)).train()
    model.save(str(out))
    (out / "train.json").write_text(json.dumps(vars(a) | {"triples": len(rows), "by_source": src}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
