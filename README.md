# worksearch

Search my 24 public repos by what a change did ("where the pet's share card is drawn", "chrf checker"), from
the command line or as an MCP tool. I also fine-tuned the embedding model on my own history to see whether
that beats keyword search, and scored it on six repos it never saw.

**Result:** on those repos, fine-tuning bge-small didn't measurably help. Fusing keyword search (BM25)
with an embedding model did: it put the right file higher (MRR@10 0.67–0.69 against BM25's 0.60), and within one
repo the gain is clear (+0.10, 95% interval +0.01 to +0.19), whether or not the model was tuned.

```bash
worksearch "where does the pet render its share card"
worksearch "chrf checker" --repo evalpair
worksearch "council vote tally mismatch" --mode hybrid
worksearch mcp          # MCP server on stdio, one tool: search_work
```

## Data

- **Corpus:** every tracked text file in the 24 repos (code, Markdown, configs; no lockfiles, no files over
  150k characters, no JSON over 20k), 1,269 files cut into 15,066 overlapping chunks of about 1,500
  characters. A file's score is its best chunk's score. Each chunk starts with its repo and path.
- **Queries:** my commit subjects. The right answer to each is the set of indexed files that commit changed
  (commits touching more than 8 files are dropped). Nothing is labelled by hand or by a model.
- **Split by repo**, the same as [diffnote](https://github.com/Srikarmk/diffnote)'s:

| Split | Queries | Repos |
|---|---|---|
| train | 269 | 14, mostly duskline (173) and Keystone (52) |
| validation | 34 | region-aware-gp (22), groupthink, format-tax, quant-cliff |
| test | 52 | Gitgotchi (25), council-votes (7), evalpair (6), hearth (6), whygit (5), bostonmove (3) |

Every repo's files are in the index at every stage, because the tool searches all of them at once; only
the queries are split.

## Method

- **Base model:** [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (33M parameters, MIT),
  with its query instruction.
- **Training pairs (training repos only):**
  - each training commit's subject paired with the best-matching chunk of each file it changed (545 pairs);
  - 1,567 synthetic queries: qwen3:8b, run locally, wrote the search a developer might type for a sample of
    chunks from the training repos.
- **Hard negatives:** each pair's negative is the best BM25 chunk from a different file.
- **Loss:** in-batch negatives (MultipleNegativesRankingLoss), batch 32.
- **Settings tried, all on validation:**
  - 1 epoch at 2e-5;
  - 3 epochs at 5e-5;
  - 3 epochs at 5e-5 on the commit pairs alone.
- **Taken to the test, chosen on validation:**
  - the best tuned model on its own (commit pairs, 3 epochs);
  - the best hybrid (all pairs, 3 epochs, fused with BM25).
- **Hybrid:** reciprocal rank fusion of BM25 and the embedding model per chunk (k = 60). It has no weight to
  tune.
- **Metrics:**
  - recall@10: the share of the commit's files in the top 10;
  - hit@10: whether any of them is;
  - MRR@10: 1 / the rank of the first one.
- **Intervals:** bootstrap over queries within each repo. Comparisons are paired by query.
- **Two settings:**
  - searching all 24 repos;
  - searching only the commit's own repo (`--repo`). This second setting was added after reading validation
    misses, before the test was run: many subjects ("README: link the case study") fit a dozen repos
    equally well.

## Results

Held-out test repos, 52 queries:

| Search | All repos: recall@10 | All repos: MRR@10 | Own repo: recall@10 | Own repo: MRR@10 |
|---|---|---|---|---|
| BM25 | 0.58 [0.49, 0.67] | 0.60 [0.50, 0.69] | 0.77 | 0.78 |
| bge-small, untuned | 0.50 | 0.52 | 0.73 | 0.75 |
| bge-small, tuned | 0.50 | 0.48 | 0.70 | 0.72 |
| BM25 + untuned bge-small | 0.59 | 0.67 | 0.78 | **0.88** |
| BM25 + tuned bge-small | **0.63** [0.54, 0.71] | **0.69** [0.59, 0.78] | **0.79** | 0.86 |

Paired differences on the test repos:

| Comparison | Recall@10 | MRR@10 |
|---|---|---|
| tuned vs untuned bge-small | −0.01 [−0.06, +0.05] | −0.04 [−0.11, +0.02] |
| tuned vs untuned, both fused with BM25 | +0.03 [−0.01, +0.09] | +0.01 [−0.05, +0.07] |
| BM25 + tuned vs BM25 | +0.05 [−0.00, +0.10] | +0.09 [−0.01, +0.19] |
| BM25 + untuned vs BM25, own repo | +0.02 [−0.02, +0.05] | **+0.10 [+0.01, +0.19]** |

What this says:
- **Fine-tuning didn't transfer.** On validation it lifted the model from 0.36 to 0.43 recall; on the
  test repos the tuned model is level with the untuned one. 2,100 pairs, a third of them from one repo
  (Keystone), taught it more about those repos than about finding code in general; on validation it kept
  answering paper questions with Keystone's LaTeX parser.
- **Keyword search is hard to beat on these queries.** Commit subjects name the things they change
  (`chrf`, `haunt`, `server.json`), and BM25 matches them exactly. Neither embedding model beats it alone.
- **Fusion is what helps:** BM25 finds the files that share a rare word, the embedding model finds ones
  that are about the same thing, and putting them together ranks the right file first more often.
- The default mode is BM25, the winner on validation, chosen before the test. `--mode hybrid` is the one I
  would now use, on the test numbers above, and nothing further is claimed for that choice.

## Limits

- 52 test queries, half from one repo (Gitgotchi); 34 validation queries, two-thirds from one. One training
  run per setting, one seed.
- Commit subjects are a proxy for searches: they describe a change, not a question, and the answer is the
  files at their current version, which may have moved on since the commit. Most were written with an AI
  coding assistant in the loop.
- Some labels are noisy: "0.2.0: chrf checker" changed only `pyproject.toml`, so the checker's own file
  counts as a miss.

## Reproduce

```bash
uv sync
uv run python -m worksearch.data                       # chunks and queries from the local clones
uv run python -m worksearch.generate --n 3000          # synthetic queries, needs Ollama with qwen3:8b
uv run python -m worksearch.train --name all-e3-lr5e-5 --epochs 3 --lr 5e-5
uv run python -m worksearch.train --name commits-e3-lr5e-5 --epochs 3 --lr 5e-5 --no-synthetic
uv run python -m worksearch.evaluate bm25 --split valid
uv run python -m worksearch.evaluate hybrid:models/all-e3-lr5e-5 --split test --scope repo
uv run python -m worksearch.evaluate compare results/test-bm25.json results/test-hybrid-all-e3-lr5e-5.json
```

Every query's top 10 for every run is in `results/`, and the training logs are in `results/logs/`.

## License

MIT. The corpus and queries come from my own public repos.
