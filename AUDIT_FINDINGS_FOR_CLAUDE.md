# Comprehensive Architectural Audit: mind-mem (Clean Git Repo)

This document contains a deep architectural audit of the `mind-mem` repository, focusing on code-level smells, structural misalignments with `SPEC.md`, and static analysis errors. It serves as an actionable backlog for Claude Code to improve the system's maintainability and correctness.

## 1. Architectural Gaps & Structural Misalignments

### 1.1 Redundant Entry Points & Module Duplication
- **Duplicate MCP Servers**: There are multiple entry points for the MCP server: `mcp_server.py` (at the root), `src/mcp_server.py`, and `src/mind_mem/mcp_entry.py`. This creates significant maintenance overhead, confuses contributors, and completely breaks static type checking across the project.
- **MyPy Blocker**: `mypy .` fails entirely with `error: Duplicate module named "mcp_server" (also at "./src/mcp_server.py")`.

### 1.2 "God Object" Design Flaw
- **Bloated MCP Implementation**: `src/mind_mem/mcp_server.py` is exceptionally large (~158KB) and contains the logic for almost all 57 MCP tools. It violates the single-responsibility principle and should be decomposed into separate domain-specific tool modules (e.g., `tools_recall.py`, `tools_governance.py`).

### 1.3 Validation Duality (Enforcement Drift)
- **Split Validation Logic**: Core system invariants (defined in `SPEC.md`) are currently enforced by both a bash script (`src/mind_mem/validate.sh`) and a Python module (`src/mind_mem/validate_py.py`). `SPEC.md` frequently cites the bash script, but modern environments prefer the Python implementation. This duality risks "enforcement drift" where one validator permits a block that the other forbids.

### 1.4 Storage Protocol Bypass
- **Tight Coupling to Markdown**: `SPEC.md` requires the `BlockStore` protocol to be the single gateway for writes. However, `src/mind_mem/apply_engine.py` directly handles atomic writes, file locks, and Markdown formatting. This bypasses the intended abstraction layer and acts as a major blocker for the `v3.2.0` Postgres adapter implementation.

## 2. Code Smells & Technical Debt

### 2.1 Hardcoded State Machine (DRY Violation)
- **Task Status Duplication**: The internal state machine for Task Status (`todo`, `doing`, `blocked`, `done`, `canceled`) is hardcoded as raw string literals across at least 8 different files:
  - `src/mind_mem/sqlite_index.py`
  - `src/mind_mem/validate_py.py`
  - `src/mind_mem/_recall_core.py`
  - `src/mind_mem/intel_scan.py`
  - `src/mind_mem/recall_vector.py`
  - `src/mind_mem/_recall_constants.py`
  - `src/mind_mem/capture.py`
  - `src/mind_mem/validate.sh`
- **Action Required**: Extract these literal strings into a centralized `TaskStatus(str, Enum)` class to prevent typos and ease future schema upgrades.

### 2.2 Atomicity Scope Risks
- **Snapshot Exclusions**: The Apply Engine (`apply_engine.py`) provides ACID-like guarantees via snapshots. However, the snapshot scope (Section 5 of `SPEC.md`) excludes `maintenance/` and `intelligence/applied/`. If an apply fails during a complex multi-stage migration, this could lead to partial state loss or untrackable failures.

## 3. Static Analysis Errors

Ruff identified 45 linting and style errors across the codebase (18 are auto-fixable).

### 3.1 Unused Imports & Variables
- `train/backport_sweep.py`: Unused `import sys`
- `generate_mind7b_training.py`: Unused `import sys`
- `train/eval_harness.py`: Unused `import re`
- `train/train_qlora.py`: Unused `from transformers import TrainingArguments`
- **Extraneous f-strings**: `generate_mind7b_training.py` contains numerous f-strings with no placeholders (e.g., `f"discussed the deployment plan"`).

### 3.2 Formatting & Line Lengths
- **Import Hygiene**: Multiple files have unsorted imports or multiple imports on one line (`train/upload_to_hf.py`, `train_mind7b_runpod.py`).
- **Line Length Violations (E501)**: Severe line length violations (>140 chars, some >400 chars) in `train/build_corpus.py` and `train/build_model_card.py` that hurt readability.

## 4. Immediate Action Plan for Claude Code

1. **Resolve MyPy Blocker**: Delete or rename the redundant `mcp_server.py` at the project root to fix the duplicate module error and re-enable global `mypy` type checking.
2. **Fix Linting Errors**: Run `ruff check . --fix` to resolve unused imports, extraneous f-strings, and basic formatting issues. Address the `E501` line length violations in the `train/` directory manually.
3. **Decompose MCP Server**: Break down the monolithic `src/mind_mem/mcp_server.py` into logical submodules to manage the 57 MCP tools effectively.
4. **Unify Validation**: Deprecate `validate.sh` in favor of making `validate_py.py` the single source of truth for invariant enforcement, ensuring it fully aligns with `SPEC.md`.
5. **Centralize Enums**: Create a `src/mind_mem/enums.py` file to centralize all hardcoded status strings (like task statuses) and refactor the codebase to use these enums.
