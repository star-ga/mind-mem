# Comprehensive Architectural Audit: mind-mem (Commit 30d8b71)

This document contains a deep architectural audit of the `mind-mem` repository as of commit `30d8b71`. It serves as an actionable backlog for Claude Code to improve maintainability, resolve architectural smells, and align the codebase with its own `SPEC.md`.

## 1. Architectural Gaps & Structural Misalignments

### 1.1 Redundant Entry Points & Module Duplication
- **Duplicate MCP Servers**: There are multiple entry points for the MCP server: `mcp_server.py` (at the root), `src/mcp_server.py`, and `src/mind_mem/mcp_entry.py`. This design creates maintenance overhead and confuses the project layout. While wrappers exist to avoid Python import errors, it remains an anti-pattern.

### 1.2 "God Object" Design Flaw
- **Bloated MCP Implementation**: `src/mind_mem/mcp_server.py` is a massive monolithic file (4,604 lines, ~158KB) housing the logic for all 57 MCP tools, rate limiting, and ACLs. It severely violates the single-responsibility principle and MUST be decomposed into domain-specific tool modules (e.g., `tools_recall.py`, `tools_governance.py`).

### 1.3 Validation Duality (Enforcement Drift)
- **Split Validation Logic**: Core system invariants (as defined in `SPEC.md`) are still enforced by both a bash script (`src/mind_mem/validate.sh`) and a Python module (`src/mind_mem/validate_py.py`). Maintaining dual validation engines poses a high risk of "enforcement drift," where one validator permits a block that the other forbids. `validate_py.py` should become the sole source of truth.

### 1.4 Storage Protocol Bypass
- **Direct File I/O**: `SPEC.md` requires the `BlockStore` protocol to be the single gateway for writes. However, components like `src/mind_mem/capture.py` directly open and write to `SIGNALS.md` using basic file locks before interacting with the `GovernanceGate`. This bypasses the centralized storage abstraction, defeating the purpose of the `BlockStore` and blocking the seamless integration of the planned v3.2.0 `PostgresAdapter`.

## 2. Code Smells & Technical Debt

### 2.1 State Machine DRY Violation (Bash vs Python)
- **Task Status Spread**: The recent introduction of `src/mind_mem/enums.py` successfully centralized `TaskStatus` for Python files (e.g., using `TaskStatus.TODO` instead of literals). However, to support `validate.sh`, a new file `_task_status_literals.sh` was introduced. This means the state machine definition is now duplicated across Python and Bash, preventing a single source of truth for the schema.

### 2.2 Atomicity Scope Risks
- **Snapshot Exclusions**: The Apply Engine (`apply_engine.py`) provides ACID-like guarantees via snapshots. However, the snapshot scope explicitly excludes `maintenance/` and `intelligence/applied/`. If an apply fails during a complex multi-stage migration, excluding these directories from the atomic rollback could lead to partial state loss or broken audit trails.

## 3. Static Analysis 
The static analysis blockers (MyPy and Ruff) have been successfully resolved in this commit. `mypy .` and `ruff check .` both report no issues.

## 4. Immediate Action Plan for Claude Code

1. **Decompose the MCP Server**: This is the highest priority. Break down the monolithic 4,604-line `src/mind_mem/mcp_server.py` into logical, domain-specific submodules to drastically improve maintainability.
2. **Consolidate Validation**: Deprecate `validate.sh` (and its supporting `_task_status_literals.sh`) in favor of making `validate_py.py` the single source of truth for invariant enforcement.
3. **Enforce BlockStore Abstraction**: Refactor `capture.py` and `apply_engine.py` so that all disk I/O passes exclusively through the `BlockStore` interface. Direct file `open()` calls for managed markdown files must be eliminated to unblock the Postgres migration.
4. **Remove Wrapper Redundancy**: Clean up the root directory and `src/` directory by consolidating the MCP entry points into a single, canonical launcher.
