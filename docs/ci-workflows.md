# CI Workflows

## Overview

MIND-Mem uses GitHub Actions for continuous integration.

Every row of both tables below is derived from `.github/workflows/*.yml` and is
gated: `scripts/check_docs_alignment.py` re-reads the workflow directory and the
`test` job's matrix on every run, so a workflow that is added, renamed, retriggered
or dropped fails this page instead of silently outdating it.

## Workflows

| Workflow (`name:`) | File | Trigger | Description |
|--------------------|------|---------|-------------|
| Audit Pinned Models | `audit-pinned.yml` | push to `main` / PR touching `mind-mem.json` or the model-provenance sources | Re-audits the pinned model manifest and its signatures |
| Benchmark | `benchmark.yml` | manual (`workflow_dispatch`) only | Recall performance benchmarks — deliberately not on push/PR, which was slowing CI feedback |
| CI | `ci.yml` | push to `main`, PR to `main` | Lint, format check, typecheck, and the full OS × Python test matrix |
| CodeQL | `codeql.yml` | push to `main`, PR, weekly (Mon 04:00 UTC) | Static analysis |
| Dependency Review | `dependency-review.yml` | PR | Checks dependency security on incoming changes |
| Docs | `docs.yml` | push to `main` / PR touching `docs/**`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`; manual | Documentation validation |
| Label Sync | `label-sync.yml` | push to `main` touching `.github/labels.yml`; manual | Syncs repo labels from the checked-in definition |
| Red Team Audit | `red-team.yml` | tag push (`v*`) | Adversarial `petri-audit` sweep before a tag is promoted |
| Release | `release.yml` | tag push (`v*`); manual | Preflight, build, sign, SBOM, publish, verify-published |
| Supply-Chain Security | `security.yml` | push to `main`, PR, weekly (Mon 06:00 UTC) | SBOM, dependency and supply-chain scanning |
| Stale Issues | `stale.yml` | daily (06:00 UTC); manual | Auto-closes stale issues/PRs |
| Web Console | `web.yml` | PR to `main`; push to `main` touching `web/**` or the workflow | `npm ci`, ESLint (`--max-warnings=0`), `tsc --noEmit` and `next build` for the `web/` console on Node 24 |

## CI Matrix

The `test` job in `ci.yml` runs every supported Python on Ubuntu, and the oldest
and newest (3.10 and 3.14) on macOS and Windows — the two ends where interpreter
and stdlib behaviour diverges; Ubuntu still covers the versions in between. That
is **9 CI jobs** (the number is derived from the matrix and its `exclude:` list,
not typed here). Runner images are pinned (`ubuntu-24.04`, `macos-26`,
`windows-2025`) rather than `-latest`, so the platform a green row proved cannot
move under an unchanged workflow. **No row is advisory**: the `continue-on-error`
carve-out that used to forgive the 3.14 rows is gone, so a red row fails the run
instead of being filed as an advisory.

| OS | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 | Python 3.14 |
|----|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Ubuntu | x | x | x | x | x |
| macOS | x |   |   |   | x |
| Windows | x |   |   |   | x |

Coverage (`--cov`) is instrumented on the `ubuntu-24.04` / 3.12 row only; the
other rows run the same selector without instrumentation to stay inside the
runner's memory budget.

## Adding a Workflow

1. Create `.github/workflows/<name>.yml`
2. Define triggers in `on:` section
3. Set minimal `permissions:`
4. Add it to the table above — `scripts/check_docs_alignment.py` fails until you do
