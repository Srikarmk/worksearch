"""Synthetic training queries: for a sample of chunks from the training repos only, a local model (qwen3:8b on
Ollama) writes the search a developer might type to find that chunk. Validation and test repos are never sent.

    uv run python -m worksearch.generate --n 3000      # appends to data/synthetic.jsonl; resumable
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import httpx

from .data import split_of
from .search import load_chunks

MODEL = "qwen3:8b"
PROMPT = ("Here is a chunk of a file from a developer's own repositories. Write one short search query, 4 to 14 "
          "words, that the developer might type months later to find this code again: name what it does or the "
          "change it supports, in plain words, the way a commit subject or a question would. Do not quote the "
          "file path. Reply with the query only.\n\n{text}")


def sample(chunks: list[dict], n: int, per_file: int = 3, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by = defaultdict(list)
    for c in chunks:
        if split_of(c["repo"]) == "train":
            by[(c["repo"], c["path"])].append(c)
    pool = []
    for cs in by.values():
        rng.shuffle(cs)
        pool += cs[:per_file]
    rng.shuffle(pool)
    return pool[:n]


def ask(client: httpx.Client, text: str) -> str:
    r = client.post("http://localhost:11434/api/chat", json={
        "model": MODEL, "think": False, "stream": False,
        "options": {"temperature": 0.7, "num_predict": 40, "num_ctx": 4096, "seed": 0},
        "messages": [{"role": "user", "content": PROMPT.format(text=text[:2500])}]}, timeout=120)
    r.raise_for_status()
    q = r.json()["message"]["content"].strip().strip('"').splitlines()[0].strip()
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    out = Path("data/synthetic.jsonl")
    done = {json.loads(line)["chunk"] for line in out.open()} if out.exists() else set()
    todo = [c for c in sample(load_chunks(), a.n) if c["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    with httpx.Client() as client, out.open("a") as f:
        for k, c in enumerate(todo, 1):
            q = ask(client, c["text"])
            if 3 <= len(q.split()) <= 25:
                f.write(json.dumps({"chunk": c["id"], "repo": c["repo"], "query": q}) + "\n")
                f.flush()
            if k % 100 == 0:
                print(k, q, flush=True)


if __name__ == "__main__":
    main()
