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

## Code Standards

- **Python 3.10+** — use modern syntax (type hints, `|` unions, etc.)
- **Zero external dependencies** for core modules (scripts/*.py)
- **ruff** for linting — zero errors required
- **pytest** for testing — all tests must pass

## Test Guidelines

- Every new module needs a corresponding test file in `tests/`
- Tests should be self-contained (use temp directories, mock externals)
- Target: 100% pass rate across Ubuntu/macOS/Windows x Python 3.10/3.12/3.13

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
