# Contributing to mind-mem

Contributions are welcome. Please follow these guidelines.

## Getting Started

1. Fork the repo
2. Clone your fork
3. Create a branch: `git checkout -b feature/your-feature`
4. Make changes
5. Run tests: `python -m pytest tests/ -x`
6. Run lint: `ruff check scripts/ tests/ mcp_server.py`
7. Commit and push
8. Open a pull request

## Development Setup

```bash
git clone https://github.com/star-ga/mind-mem.git
cd mind-mem
python -m pip install -e ".[dev]"
python -m pytest tests/ -x
```

## Module Map

The codebase is organized into focused modules under `scripts/`:

| Module | Purpose | Lines |
|--------|---------|------:|
| `_recall_core.py` | Main recall pipeline: BM25 scoring loop, graph boost, PRF, RM3, CLI | ~1200 |
| `_recall_scoring.py` | BM25F helper, date scores, negation, categories, entity extraction | ~260 |
| `_recall_constants.py` | All named constants: BM25 params, field weights, boost factors | ~240 |
| `_recall_detection.py` | Query type detection, intent routing, tokenization helpers | ~350 |
| `_recall_expansion.py` | RM3 expansion, synonym expansion, month normalization | ~200 |
| `_recall_temporal.py` | Temporal query parsing, date range resolution, filtering | ~180 |
| `_recall_reranking.py` | Deterministic reranker, optional LLM reranker | ~250 |
| `_recall_context.py` | Context packing: adjacency, diversity, pronoun rescue | ~200 |
| `_recall_tokenization.py` | Porter stemming, lemmatization, tokenizer | ~150 |
| `sqlite_index.py` | FTS5 index: build, incremental update, fact extraction, query | ~1100 |
| `retrieval_graph.py` | Co-retrieval logging, PageRank propagation, hard negatives | ~300 |
| `mcp_server.py` | MCP server: 18 tools, 8 resources, stdio + HTTP transports | ~600 |

## Code Standards

- **Python 3.10+** — use modern syntax (type hints, `|` unions, etc.)
- **Zero external dependencies** for core modules (scripts/*.py)
- **ruff** for linting — zero errors required
- **pytest** for testing — all tests must pass
- **Named constants** — no magic numbers in scoring logic (use `_recall_constants.py`)
- **Single source of truth** — BM25 formula lives in `bm25f_score_terms()` only

## Test Guidelines

- Every new module needs a corresponding test file in `tests/`
- Tests should be self-contained (use temp directories, mock externals)
- Target: 100% pass rate across Ubuntu/macOS/Windows x Python 3.10/3.12/3.13/3.14
- CI runs 14 matrix jobs (3 OSes x 4-5 Python versions)

## Pull Request Checklist

- [ ] All tests pass (`python -m pytest tests/ -x`)
- [ ] `ruff check` reports zero errors
- [ ] New features have tests
- [ ] No new external dependencies added to core

## Architecture Decisions

Before proposing significant architectural changes, please open an issue first to discuss the approach. See [docs/architecture.md](docs/architecture.md) for the current system design.

## MIND Kernels

If modifying `.mind` files:
- Follow MIND syntax (see [mind/README.md](mind/README.md))
- Ensure pure Python fallback exists for every MIND function
- Test both with and without compiled `.so`

## Reporting Issues

Please include:
- Python version
- OS
- Minimal reproduction steps
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
