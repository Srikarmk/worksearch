"""Score a retriever on the commit queries: does a file the commit changed come back in the top 10?

    uv run python -m worksearch.evaluate bm25 --split valid
    uv run python -m worksearch.evaluate base --split test
    uv run python -m worksearch.evaluate models/run1 --split valid
    uv run python -m worksearch.evaluate hybrid:models/run1 --split valid   # BM25 and the model, fused by rank
    uv run python -m worksearch.evaluate compare results/test-a.json results/test-b.json

Per query: recall@10 is the share of the commit's files in the top 10, hit@10 is whether any is, and MRR@10 is
1 / the rank of the first one. Intervals bootstrap over queries within repo, so a repo with many commits is not
resampled as if its queries were independent of each other more than they are.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .search import BASE, BM25, Dense, load_chunks, rank_files

K = 10


def per_query(ranked: list[str], gold: list[str]) -> dict:
    top = ranked[:K]
    first = next((i + 1 for i, f in enumerate(top) if f in gold), None)
    return {"recall": sum(f in top for f in gold) / len(gold), "hit": float(first is not None),
            "mrr": 1 / first if first else 0.0}


def run(retriever: str, split: str) -> dict:
    chunks = load_chunks()
    qs = [json.loads(line) for line in open("data/queries.jsonl")]
    qs = [q for q in qs if q["split"] == split]
    texts = [c["text"] for c in chunks]
    hybrid = retriever.startswith("hybrid:")
    dense = retriever.removeprefix("hybrid:")
    if retriever == "bm25" or hybrid:
        bm = BM25(texts)
        B = np.stack([bm.scores(q["query"]) for q in qs])
    if retriever != "bm25":
        path = BASE if dense == "base" else dense
        cache = Path("data/emb") / (("base" if dense == "base" else Path(dense).name) + ".npy")
        D = Dense(texts, path, cache).scores_many([q["query"] for q in qs])
    S = B if retriever == "bm25" else rrf(B, D) if hybrid else D
    items = []
    for q, s in zip(qs, S):
        ranked = [f for f, _, _ in rank_files(s, chunks, K)]
        items.append({"id": q["id"], "repo": q["repo"], "query": q["query"], "gold": q["files"], "top": ranked,
                      **per_query(ranked, q["files"])})
    return {"retriever": retriever, "split": split, "n": len(items), **summary(items), "items": items}


def rrf(*score_sets: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion per chunk: sum of 1 / (k + rank) over the retrievers. No weights to tune."""
    out = np.zeros_like(score_sets[0], dtype=float)
    for S in score_sets:
        ranks = np.argsort(np.argsort(-S, axis=1), axis=1) + 1
        out += 1.0 / (k + ranks)
    return out


def boot(items: list[dict], key: str, n: int = 4000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    by: dict[str, list[float]] = {}
    for it in items:
        by.setdefault(it["repo"], []).append(it[key])
    groups = [np.array(v) for v in by.values()]
    vals = [np.mean(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(n)]
    return [round(float(x), 3) for x in np.percentile(vals, [2.5, 97.5])]


def summary(items: list[dict]) -> dict:
    return {k: {"mean": round(float(np.mean([i[k] for i in items])), 3), "ci": boot(items, k)}
            for k in ("recall", "hit", "mrr")}


def compare(a: dict, b: dict, n: int = 4000, seed: int = 0) -> dict:
    """b minus a, paired by query."""
    A = {i["id"]: i for i in a["items"]}
    pairs = [{"repo": i["repo"], **{k: i[k] - A[i["id"]][k] for k in ("recall", "hit", "mrr")}}
             for i in b["items"] if i["id"] in A]
    return {"a": a["retriever"], "b": b["retriever"], "n": len(pairs),
            **{k: {"diff": round(float(np.mean([p[k] for p in pairs])), 3), "ci": boot(pairs, k, n, seed),
                   "b_better": sum(p[k] > 0 for p in pairs), "a_better": sum(p[k] < 0 for p in pairs)}
               for k in ("recall", "hit", "mrr")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("retriever", help="bm25, base, a model folder, or 'compare'")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    a = ap.parse_args()
    if a.retriever == "compare":
        x, y = (json.loads(Path(f).read_text()) for f in a.files)
        print(json.dumps(compare(x, y), indent=1))
        return
    out = run(a.retriever, a.split)
    name = a.retriever if a.retriever in ("bm25", "base") else (
        "hybrid-" if a.retriever.startswith("hybrid:") else "") + Path(a.retriever.removeprefix("hybrid:")).name
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{a.split}-{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "items"}))


if __name__ == "__main__":
    main()
