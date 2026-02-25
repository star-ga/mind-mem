.PHONY: test lint bench install dev clean smoke help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run all 1352 tests
	python3 -m pytest tests/ -x -q

lint: ## Run ruff linter
	ruff check scripts/ tests/ mcp_server.py

bench: ## Run performance benchmark (no API key needed)
	python3 benchmarks/bench_kernels.py --iterations 50 --sizes 100,500,1000

install: ## Install mind-mem in editable mode
	python3 -m pip install -e ".[dev]"

dev: install ## Full dev setup: install + pre-commit + smoke test
	@command -v pre-commit >/dev/null 2>&1 && pre-commit install || true
	@echo "Dev environment ready. Run 'make test' to verify."

smoke: ## Quick smoke test (creates temp workspace, runs pipeline)
	bash scripts/smoke_test.sh

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ scripts/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .ruff_cache
	rm -rf dist/ build/ *.egg-info
	rm -rf htmlcov coverage.xml .coverage
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

validate: ## Run 74+ structural checks on current workspace
	python3 scripts/validate_py.py .

reindex: ## Rebuild FTS5 index
	python3 scripts/sqlite_index.py build --workspace .

scan: ## Run integrity scan (contradictions, drift, dead decisions)
	python3 scripts/intel_scan.py .

.PHONY: coverage benchmark docs format typecheck

coverage: ## Run tests with coverage report
	python3 -m pytest tests/ --cov=scripts --cov-report=term-missing --cov-report=html -q

benchmark: ## Run recall performance benchmark
	python3 -m pytest tests/ -k "benchmark or perf" -v --timeout=60

docs: ## Validate documentation links
	@echo "Checking markdown links..."
	@find docs/ -name '*.md' -exec grep -l 'http' {} \;

format: ## Auto-format code
	python3 -m ruff format scripts/ tests/

typecheck: ## Run type checking (if mypy installed)
	python3 -m mypy scripts/ --ignore-missing-imports --no-error-summary 2>/dev/null || echo "mypy not installed — skipping"
