# Contributing to MIND-Mem

Thank you for your interest in contributing to MIND-Mem!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/star-ga/mind-mem.git
cd mind-mem

# Install in development mode
pip install -e ".[test]"

# Run tests
pytest tests/ -v

# Run linting
ruff check src/ tests/
ruff format --check src/ tests/
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass: `pytest tests/ -v`
4. Ensure linting passes: `ruff check src/ tests/`
5. Update documentation if applicable
6. Submit a pull request with a clear description

## Code Style

- Python 3.10+ with type hints
- Formatting: `ruff format`
- Linting: `ruff check`
- Zero external dependencies in core (stdlib only)
- Tests use pytest with fixtures

## Codebase Layout — Shims vs. Canonical Modules

The MCP surface was decomposed from a single ``mcp_server.py``
monolith into per-domain modules under ``src/mind_mem/mcp/`` over
the v3.2.0–v4.0.0 series. To keep public imports stable, several
top-level modules remain as **shims** — they only re-export from
their canonical location. New code MUST land in the canonical
module, not the shim.

| Shim (do not extend)              | Canonical (target for new code)          |
|-----------------------------------|------------------------------------------|
| ``mind_mem/mcp_server.py``        | ``mind_mem/mcp/server.py`` + ``mcp/tools/*`` |
| Tool decorator + ACL/rate-limit   | ``mind_mem/mcp/infra/observability.py`` |
| Path / workspace guards           | ``mind_mem/mcp/infra/workspace.py``     |
| Encryption helpers                | ``mind_mem/mcp/tools/encryption.py``    |
| Memory ops (delete/export/...)    | ``mind_mem/mcp/tools/memory_ops.py``    |
| Federation HTTP handlers          | ``mind_mem/http_transport.py``          |

When in doubt: if a module is < 80 lines and consists mostly of
``from ... import ...`` lines, it is a shim. Add tests against the
canonical module path, not the shim.

## Test Guidelines

- Colocate tests in `tests/test_<module>.py`
- Use `tmp_path` fixture for filesystem tests
- Mock external dependencies (LLM calls, network)
- Aim for >90% coverage on new code

## Docs-Claim Gate

Public claims (IR version strings, runtime-boundary wording, tool counts)
must not drift out of sync with the wider MIND project. A small regression
gate checks `README.md` and `docs/**/*.md` against the shared MIND
capability manifest:

```bash
# Requires the star-ga/mind repo checked out beside this one (../mind).
scripts/check_claims.sh
```

It runs forbidden-phrase + canonical-IR checks only; it exits 0 (skips
cleanly) if the `mind` repo isn't present, so it's safe to run anywhere.
Run it after editing any public docs.

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
