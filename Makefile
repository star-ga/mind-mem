.PHONY: test lint bench install dev clean smoke help regen-bash-literals

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run the test suite
	python3 -m pytest tests/ -x -q

lint: ## Run ruff linter (whole repo — same scope as the CI lint job)
	ruff check .

bench: ## Run performance benchmark (no API key needed)
	python3 benchmarks/bench_kernels.py --iterations 50 --sizes 100,500,1000

install: ## Install mind-mem in editable mode
	python3 -m pip install -e ".[dev]"

dev: install ## Full dev setup: install + pre-commit + smoke test
	@command -v pre-commit >/dev/null 2>&1 && pre-commit install || true
	@echo "Dev environment ready. Run 'make test' to verify."

smoke: ## Quick smoke test (creates temp workspace, runs pipeline)
	bash src/mind_mem/smoke_test.sh

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ src/mind_mem/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .ruff_cache
	rm -rf dist/ build/ *.egg-info
	rm -rf htmlcov coverage.xml .coverage
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

validate: ## Run 74+ structural checks on current workspace
	python3 -m mind_mem.validate_py .

reindex: ## Rebuild FTS5 index
	python3 -m mind_mem.sqlite_index build --workspace .

scan: ## Run integrity scan (contradictions, drift, dead decisions)
	python3 -m mind_mem.intel_scan .

.PHONY: coverage benchmark repro-niah repro-niah-smoke repro-verify repro-benchmarks docs format typecheck

coverage: ## Run tests with coverage report
	python3 -m pytest tests/ --cov=src/mind_mem --cov-report=term-missing --cov-report=html -q

benchmark: ## Run recall performance benchmark
	python3 -m pytest tests/ -k "benchmark or perf" -v --timeout=60

repro-niah: ## FULL NIAH matrix (250 cells, ~1h) -> repro package: raw NDJSON + metrics + pinned manifest (local, no API key)
	python3 benchmarks/repro_niah.py --out benchmarks/repro/niah

repro-niah-smoke: ## Same machinery over a 7-cell subset spanning every size/depth -- proves the harness, NOT the 250/250 figure
	python3 benchmarks/repro_niah.py --every 37 --out benchmarks/repro/niah-smoke

repro-verify: ## Recompute every committed number from its committed raw rows. Non-zero exit = a published figure does not follow.
	python3 benchmarks/repro_verify.py all

repro-benchmarks: repro-niah repro-verify ## Run all reproducible, third-party-verifiable benchmark packages, then verify them
	@echo "Repro artifacts under benchmarks/repro/ -- rerun this target and diff the manifest.json hashes."

docs: ## Validate documentation links
	@echo "Checking markdown links..."
	@find docs/ -name '*.md' -exec grep -l 'http' {} \;

format: ## Auto-format code (same scope as the CI format check)
	python3 -m ruff format . --exclude benchmarks --exclude train

typecheck: ## Run type checking
# This gate reported success for every outcome. `... 2>/dev/null || echo
# "mypy not installed — skipping"` turned any type error, and any failure to
# run at all, into exit 0 plus a reassuring message.
#
# Measured in ONE environment while fixing it, stated as such: on this machine
# `python3 -m mypy` is not importable, because mypy is installed as a
# standalone executable rather than into this interpreter. There the target
# printed "skipping" and exited 0 -- so the gate reported success without
# type-checking. Whether any other machine's interpreter could import mypy is
# not something that measurement establishes, and the earlier wording claimed
# it did.
#
# So: prefer the executable, fall back to the module, and if neither is
# present say so with a non-zero exit and installation guidance. Absence is
# reported as absence. mypy's own exit code and its stderr propagate.
	@if command -v mypy >/dev/null 2>&1; then \
	  MYPY="mypy"; \
	elif python3 -c 'import mypy' >/dev/null 2>&1; then \
	  MYPY="python3 -m mypy"; \
	else \
	  echo "typecheck: mypy is not available as an executable or as a module." >&2; \
	  echo "           Install it with:  pip install mypy" >&2; \
	  exit 2; \
	fi; \
	echo "typecheck: using $$MYPY"; \
	$$MYPY src/ --ignore-missing-imports --no-error-summary

regen-bash-literals: ## Regenerate src/mind_mem/_task_status_literals.sh from enums.py
	python3 scripts/regen_bash_literals.py
