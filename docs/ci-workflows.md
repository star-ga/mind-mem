# CI Workflows

## Overview

mind-mem uses GitHub Actions for continuous integration.

## Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| CI | push, PR | Full test suite across OS/Python matrix |
| Benchmark | push, PR | Recall performance benchmarks |
| Security Review | push, PR | Claude Code security analysis |
| CodeQL | push, PR, weekly | Static analysis |
| Docs | push, PR | Documentation validation |
| Stale | daily | Auto-close stale issues/PRs |
| Release | tags | Auto-create GitHub releases |
| Label Sync | push | Sync repo labels |
| Dependency Review | PR | Check dependency security |

## CI Matrix

| OS | Python 3.10 | Python 3.12 | Python 3.13 | Python 3.14 |
|----|:-----------:|:-----------:|:-----------:|:-----------:|
| Ubuntu | x | x | x | x |
| macOS | | x | | x |
| Windows | | x | | x |

## Adding a Workflow

1. Create `.github/workflows/<name>.yml`
2. Define triggers in `on:` section
3. Set minimal `permissions:`
4. Add to this documentation
