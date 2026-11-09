import numpy as np

from worksearch import data
from worksearch.evaluate import per_query, rrf
from worksearch.search import BM25, rank_files, tokens


def test_chunks_cover_every_line_and_overlap():
    text = "".join(f"line {i} " + "x" * 90 + "\n" for i in range(100))
    cs = data.chunks("whygit", "a.py", text)
    starts = [c["line"] for c in cs]
    assert starts[0] == 1 and starts == sorted(starts) and len(cs) > 3
    assert all(c["text"].startswith("whygit/a.py\n") for c in cs)
    assert "line 99 " in cs[-1]["text"]
    assert starts[1] - starts[0] < 1500 // 100  # the second chunk starts before the first one ends


def test_tokens_split_identifiers():
    t = tokens("def staged_diff(): stagedDiff")
    assert {"staged", "diff", "staged_diff", "stageddiff"} <= set(t)


def test_bm25_prefers_matching_chunk():
    bm = BM25(["the pet renders a card", "parse the arxiv abstract page", "nothing here"])
    assert int(np.argmax(bm.scores("arxiv abstract"))) == 1


def test_rank_files_keeps_best_chunk_per_file():
    chunks = [{"repo": "r", "path": "a"}, {"repo": "r", "path": "a"}, {"repo": "r", "path": "b"}]
    out = rank_files(np.array([0.9, 0.8, 0.5]), chunks, k=10)
    assert [f for f, _, _ in out] == ["r/a", "r/b"]


def test_per_query_metrics():
    m = per_query(["x", "a", "y", "b"], ["a", "b", "c"])
    assert m == {"recall": 2 / 3, "hit": 1.0, "mrr": 0.5}


def test_rrf_rewards_agreement():
    a = np.array([[3.0, 2.0, 1.0]])
    b = np.array([[1.0, 3.0, 2.0]])
    assert int(np.argmax(rrf(a, b))) == 1


def test_splits_match_diffnote():
    assert not data.TEST & data.VALID and data.TEST | data.VALID <= set(data.REPOS)
