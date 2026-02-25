# Development Guide

## Setup

```bash
git clone https://github.com/star-ga/mind-mem.git
cd mind-mem
pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_recall_edge_cases.py

# With coverage
pytest tests/ --cov=scripts --cov-report=term-missing

# Benchmarks only
pytest tests/ -k "benchmark or perf" --benchmark-only
```

## Code Style

- Python 3.10+ compatible
- Use `from __future__ import annotations` in all files
- Run `ruff check` before committing
- Run `ruff format` for formatting

## Project Structure

```
scripts/           # Core library modules
tests/             # Test suite
mcp_server.py      # MCP server entry point
docs/              # Documentation
.github/           # CI workflows and templates
```

## Adding New Features

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Write tests first (TDD encouraged)
3. Implement the feature
4. Run full test suite: `pytest tests/`
5. Create a PR with description

## Commit Messages

Follow conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Test additions
- `ci:` CI changes
- `refactor:` Code refactoring

## CI Pipeline

- **CI**: Runs tests on Python 3.10/3.12/3.13/3.14 across Ubuntu/macOS/Windows
- **Benchmark**: Measures recall latency
- **Security Review**: CodeQL + Claude Code review
- **Docs**: Validates documentation
