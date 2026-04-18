# Full Audit of mind-mem Repository

## 1. Architectural Gaps & Structural Misalignments
- **Scaling vs. Format**: The system relies on parsing Markdown files for core operations. This introduces O(N) overhead for file-based scans. While a Postgres adapter (v3.2.0) is planned, it's not fully integrated into the core retrieval path yet.
- **Concurrency Bottlenecks**: Although advisory file locks and SQLite WAL are used, simultaneous writes to the same Markdown file by multiple agents remain a potential point of contention and risk.
- **Component Fragmentation**: The recall engine is split across many private `_recall_*.py` files. While modular, the `RecallBackend` abstraction is still evolving, leading to tight coupling between the facade and implementation submodules.
- **Missing Production Features**: Features like a native REST API, OIDC authentication, and distributed query caching are listed in the roadmap (v3.2.0+) but are not yet implemented in the core. The 'Streaming Ingestion' (v2.5.0) is partially implemented via hooks rather than a robust high-throughput event bus.

## 2. Static Analysis Errors & Warnings

### Type Checking (MyPy)
- **Duplicate Module Error**: `mypy .` fails with: `mcp_server.py: error: Duplicate module named "mcp_server" (also at "./src/mcp_server.py")`. This prevents full type checking across the repository.
  - **Resolution**: Remove or rename the redundant `mcp_server.py` at the project root if it's just a proxy, or configure mypy `--exclude mcp_server.py`.

### Linting & Style (Ruff)
- **Extraneous f-strings**: `generate_mind7b_training.py` has multiple f-strings without any placeholders (e.g., `f"discussed the deployment plan"`).
- **Unused Imports**:
  - `generate_mind7b_training.py`: `import sys`
  - `train/backport_sweep.py`: `import sys`
  - `train/eval_harness.py`: `import re`
  - `train/train_qlora.py`: `from transformers import TrainingArguments`
- **Import Sorting & Formatting**: Multiple files have unsorted or multiple imports on one line (e.g., `train/upload_to_hf.py`, `train_mind7b_runpod.py`).
- **Line Length Violations (E501)**: Several lines exceed the 140-character limit in `train/build_corpus.py`, `train/build_model_card.py`, and `generate_mind7b_training.py`.

## Action Items for Claude Code
1. Refactor imports and remove unused variables/f-strings by running `ruff check . --fix`.
2. Resolve the `mcp_server.py` duplicate module error to re-enable full `mypy` type checking.
3. Review and fix long lines (`E501`) in the `train/` directory.
4. Begin addressing architectural gaps, starting with abstracting the `RecallBackend` to reduce coupling and investigating concurrent write safety for Markdown storage.
