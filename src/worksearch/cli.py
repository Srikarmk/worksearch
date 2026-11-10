"""worksearch: search my public repos by what a change did, not only by the words in it.

    worksearch "where does the pet render its share card"
    worksearch "chrf checker" --repo evalpair
    worksearch "council vote tally mismatch" --mode hybrid
    worksearch index            # rebuild the chunks after the repos change
    worksearch mcp              # MCP server on stdio, with one tool: search

The default mode is BM25, which won on the validation repos; `hybrid` fuses BM25 with the fine-tuned
bge-small model (results in README.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data import REPOS
from .search import BM25, Dense, load_chunks, rank_files

HOME = Path(__file__).resolve().parents[2]
MODEL = HOME / "models" / "all-e3-lr5e-5"


@lru_cache(maxsize=1)
def _chunks():
    return load_chunks(HOME / "data" / "chunks.jsonl")


@lru_cache(maxsize=1)
def _bm25():
    return BM25([c["text"] for c in _chunks()])


@lru_cache(maxsize=1)
def _dense():
    return Dense([c["text"] for c in _chunks()], str(MODEL), HOME / "data" / "emb" / f"{MODEL.name}.npy")


def search(query: str, repo: str | None = None, mode: str = "bm25", k: int = 10) -> list[dict]:
    from .evaluate import rrf
    chunks = _chunks()
    b = _bm25().scores(query)
    s = b if mode == "bm25" else rrf(b[None], _dense().scores(query)[None])[0]
    if repo:
        names = {r for r, gh in REPOS.items() if repo.lower() in (r.lower(), gh.lower())}
        if not names:
            raise SystemExit(f"unknown repo {repo!r}; one of: {', '.join(sorted(REPOS.values()))}")
        s = np.where(np.isin([c["repo"] for c in chunks], list(names)), s, -np.inf)
    out = []
    for f, score, i in rank_files(s, chunks, k):
        c = chunks[i]
        snippet = c["text"].split("\n", 1)[1][:300] if "\n" in c["text"] else ""
        out.append({"repo": REPOS[c["repo"]], "path": c["path"], "line": c["line"], "score": round(score, 4),
                    "local": str(Path.home() / "Programming" / c["repo"] / c["path"]), "snippet": snippet})
    return out


def serve():
    from mcp.server.fastmcp import FastMCP
    app = FastMCP("worksearch")

    @app.tool()
    def search_work(query: str, repo: str | None = None, mode: str = "bm25", k: int = 10) -> str:
        """Find files in Srikar's public repos for a description of code or of a change ("where the pet's share
        card is drawn", "chrf checker"). Returns up to k files, best first, each with the line its best chunk
        starts on and a snippet. `repo` limits the search to one repo; `mode` is "bm25" (default, best on the
        validation repos) or "hybrid" (BM25 fused with the fine-tuned embedding model). Read-only."""
        return json.dumps(search(query, repo, mode, k), indent=1)

    app.run()


def main(argv=None):
    if (argv or sys.argv[1:])[:1] == ["index"]:
        import os
        from .data import build
        os.chdir(HOME)  # build() writes data/ relative to the project
        print(build())
        return
    if (argv or sys.argv[1:])[:1] == ["mcp"]:
        serve()
        return
    ap = argparse.ArgumentParser(prog="worksearch")
    ap.add_argument("query")
    ap.add_argument("--repo")
    ap.add_argument("--mode", default="bm25", choices=["bm25", "hybrid"])
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    hits = search(a.query, a.repo, a.mode, a.k)
    if a.json:
        print(json.dumps(hits, indent=1))
        return
    for h in hits:
        print(f"{h['repo']}/{h['path']}:{h['line']}")
        first = next((l.strip() for l in h["snippet"].splitlines() if l.strip()), "")
        if first:
            print(f"    {first[:110]}")
