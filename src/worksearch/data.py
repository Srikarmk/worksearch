"""The corpus (every indexed text file in my public repos, cut into chunks) and the queries (my commit subjects,
each paired with the files that commit changed).

The split is by repo and is the same as diffnote's: queries from the test repos are never used for training or
for choosing anything. Every repo's files are in the index at every stage, because the product searches all of
them at once.

    uv run python -m worksearch.data      # data/chunks.jsonl, data/queries.jsonl
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path.home() / "Programming"
OUT = Path("data")

# folder -> GitHub repo; all public and owned by Srikarmk
REPOS = {
    "Aurora-GP": "region-aware-gp", "Gitgotchi": "Gitgotchi", "Keystone": "Keystone",
    "MLOps-Lab2": "MLOps-Lab2", "MLOps-Lab3": "MLOps-Lab3", "Silent-RAG-Failures": "Silent-RAG-Failures",
    "Srikarmk-profile": "Srikarmk", "bostonmove": "bostonmove", "council-votes": "council-votes",
    "datasniff": "datasniff", "evalpair": "evalpair", "experiment-forensics": "experiment-forensics",
    "fantasy-racer": "duskline", "format-tax": "format-tax", "groupthink": "groupthink", "hearth": "hearth",
    "llm-judge-bias": "llm-judge-bias", "long-context-recall": "long-context-recall", "memory-rot": "memory-rot",
    "prompt-brittleness": "prompt-brittleness", "quant-cliff": "quant-cliff", "rag-poisoning": "rag-poisoning",
    "tool-call-reliability": "tool-call-reliability", "whygit": "whygit",
}
TEST = {"Gitgotchi", "whygit", "council-votes", "hearth", "bostonmove", "evalpair"}
VALID = {"Aurora-GP", "groupthink", "format-tax", "quant-cliff"}
NO_QUERIES = {"Srikarmk-profile"}  # one file: every commit "finds" it

TEXT = re.compile(r"\.(py|ts|tsx|js|mjs|cjs|md|toml|ya?ml|json|sh|txt|css|html|sql|cfg|ini|tex|swift|kt|r)$", re.I)
SKIP = re.compile(r"(^|/)(package-lock\.json|pnpm-lock\.yaml|yarn\.lock|uv\.lock|poetry\.lock|LICENSE(\.\w+)?)$"
                  r"|(^|/)(node_modules|dist|build|\.next|vendor)/", re.I)
MAX_CHARS = 150_000       # bigger files are data dumps, not something to search for
MAX_JSON = 20_000         # small JSON (package.json, server.json, configs) is indexed; data files are not
CHUNK, STEP = 1500, 1200  # characters; bge-small reads 512 tokens, about 1,500-2,000 characters of code


def split_of(repo: str) -> str:
    return "test" if repo in TEST else "valid" if repo in VALID else "train"


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT / repo), *args], capture_output=True, text=True, check=True,
                          errors="replace").stdout


def indexed_files(repo: str) -> list[str]:
    out = []
    for path in git(repo, "ls-files").splitlines():
        if not TEXT.search(path) or SKIP.search(path):
            continue
        f = ROOT / repo / path
        try:
            n = f.stat().st_size
        except OSError:
            continue
        if n == 0 or n > MAX_CHARS or (path.endswith(".json") and n > MAX_JSON):
            continue
        out.append(path)
    return out


def chunks(repo: str, path: str, text: str) -> list[dict]:
    """Windows of whole lines, about CHUNK characters, each starting about STEP after the last. Each chunk
    carries its repo and path in its text, which is how a person would describe where it is."""
    lines = text.splitlines(keepends=True)
    out, start = [], 0
    while start < len(lines):
        size, end = 0, start
        while end < len(lines) and (size < CHUNK or end == start):
            size += len(lines[end])
            end += 1
        body = "".join(lines[start:end]).strip()
        if body:
            out.append({"id": f"{repo}/{path}#L{start + 1}", "repo": repo, "path": path, "line": start + 1,
                        "text": f"{REPOS[repo]}/{path}\n{body}"})
        if end >= len(lines):
            break
        back, nxt = 0, end  # step back so neighbouring chunks overlap by about CHUNK - STEP characters
        while nxt > start + 1 and back < CHUNK - STEP:
            nxt -= 1
            back += len(lines[nxt])
        start = max(nxt, start + 1)
    return out


TRAILER = re.compile(r"^(Co-Authored-By|Signed-off-by):", re.I)


def queries(repo: str, indexed: set[str]) -> list[dict]:
    """One query per commit authored as Srikarmk: the subject, and the indexed files it changed that still exist.
    Commits touching more than 8 indexed files are dropped (a rename or a sweep has no single answer)."""
    if repo in NO_QUERIES:
        return []
    log = git(repo, "log", "--no-merges", "--author=Srikarmk", "--format=%x1e%H%x1f%s", "--name-only")
    out, seen = [], set()
    for rec in log.split("\x1e")[1:]:
        head, *names = rec.strip().split("\n")
        sha, subject = head.split("\x1f", 1)
        files = sorted({n for n in names if n and n in indexed})
        if not files or len(files) > 8 or len(subject) < 15 or subject.lower() in seen:
            continue
        seen.add(subject.lower())
        out.append({"id": f"{repo}:{sha[:10]}", "repo": repo, "split": split_of(repo), "query": subject.strip(),
                    "files": [f"{repo}/{f}" for f in files]})
    return out


def build() -> dict:
    OUT.mkdir(exist_ok=True)
    n_chunks, qs, files = 0, [], 0
    with (OUT / "chunks.jsonl").open("w") as f:
        for repo in sorted(REPOS):
            paths = indexed_files(repo)
            files += len(paths)
            for p in paths:
                for c in chunks(repo, p, (ROOT / repo / p).read_text(errors="replace")):
                    f.write(json.dumps(c) + "\n")
                    n_chunks += 1
            qs += queries(repo, set(paths))
    with (OUT / "queries.jsonl").open("w") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    by = {s: sum(q["split"] == s for q in qs) for s in ("train", "valid", "test")}
    return {"files": files, "chunks": n_chunks, "queries": by}


if __name__ == "__main__":
    print(build())
