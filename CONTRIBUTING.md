# Contributing to mind-mem

Thank you for your interest in contributing to mind-mem!

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
ruff check scripts/ tests/
ruff format --check scripts/ tests/
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass: `pytest tests/ -v`
4. Ensure linting passes: `ruff check scripts/ tests/`
5. Update documentation if applicable
6. Submit a pull request with a clear description

## Code Style

- Python 3.10+ with type hints
- Formatting: `ruff format`
- Linting: `ruff check`
- Zero external dependencies in core (stdlib only)
- Tests use pytest with fixtures

## Test Guidelines

- Colocate tests in `tests/test_<module>.py`
- Use `tmp_path` fixture for filesystem tests
- Mock external dependencies (LLM calls, network)
- Aim for >90% coverage on new code

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
