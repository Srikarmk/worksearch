"""Search the chunks with BM25 or an embedding model, and rank files by their best chunk."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

BASE = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge's own instruction for queries
TOKEN = re.compile(r"[a-z0-9]+")


def load_chunks(path: Path = Path("data/chunks.jsonl")) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def tokens(text: str) -> list[str]:
    """Lowercase words, with camelCase and snake_case split, so `staged_diff` and `stagedDiff` both match
    'staged diff'. The whole identifier is kept too."""
    out = []
    for w in re.findall(r"[A-Za-z0-9_]+", text):
        parts = [p for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", w) if p]
        out += [p.lower() for p in parts]
        if len(parts) > 1:
            out.append(w.lower())
    return out


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        docs = [Counter(tokens(t)) for t in texts]
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        n, lens = len(docs), np.array([sum(d.values()) for d in docs], dtype=float)
        df = Counter(w for d in docs for w in d)
        self.idf = np.array([math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in self.vocab])
        rows, cols, vals = [], [], []
        for i, d in enumerate(docs):
            norm = k1 * (1 - b + b * lens[i] / lens.mean())
            for w, tf in d.items():
                rows.append(i)
                cols.append(self.vocab[w])
                vals.append(tf * (k1 + 1) / (tf + norm))
        self.m = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab)))

    def scores(self, query: str) -> np.ndarray:
        q = np.zeros(len(self.vocab))
        for w in set(tokens(query)):
            if w in self.vocab:
                q[self.vocab[w]] = self.idf[self.vocab[w]]
        return self.m @ q


class Dense:
    def __init__(self, texts: list[str], model_path: str = BASE, cache: Path | None = None, batch: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path, local_files_only=model_path == BASE)
        self.model.max_seq_length = 512
        if cache and cache.exists():
            self.E = np.load(cache)
        else:
            self.E = self.model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.E)

    def scores_many(self, queries: list[str]) -> np.ndarray:
        Q = self.model.encode([QUERY_PREFIX + q for q in queries], normalize_embeddings=True, batch_size=64,
                              show_progress_bar=False)
        return Q @ self.E.T

    def scores(self, query: str) -> np.ndarray:
        return self.scores_many([query])[0]


def rank_files(scores: np.ndarray, chunks: list[dict], k: int = 10) -> list[tuple[str, float, int]]:
    """Files by their best chunk: (repo/path, score, index of that chunk)."""
    best: dict[str, tuple[float, int]] = {}
    for i in np.argsort(-scores)[: max(200, 20 * k)]:
        if not np.isfinite(scores[i]):  # outside the searched repo
            break
        key = f"{chunks[i]['repo']}/{chunks[i]['path']}"
        if key not in best:
            best[key] = (float(scores[i]), int(i))
        if len(best) >= k:
            break
    return [(f, s, i) for f, (s, i) in best.items()]
