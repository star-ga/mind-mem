# `maintenance/` namespaces

v3.2.0 splits `maintenance/` into two sibling subdirectories so the
apply-engine's snapshot scope can categorise each file correctly.

This document pins the classification rule and documents the
migration semantics.

## Why two namespaces

Before v3.2.0, `apply_engine` excluded `maintenance/` wholesale from
snapshots. A multi-stage apply that wrote a corpus block *and* touched
`maintenance/dedup-state.json` could crash between the two writes,
roll the block back, and leave the dedup hash — so the next apply
silently skipped re-writing the rolled-back block.

§2.2 of `AUDIT_FINDINGS_FOR_CLAUDE.md` flagged this. v3.2.0 resolves
it by separating *behavioural state* (must survive round-trip with the
corpus) from *append-only observability* (must survive rollback
as-is, because rolling it back discards real signal).

## The two subdirectories

### `maintenance/tracked/` — snapshot-included

Files whose presence or content **changes the next apply's behaviour**.
Restored on rollback so the corpus and the behavioural state stay in
lock-step.

Classification (suffix match):

| Pattern                | Example                        | Rationale                         |
| ---------------------- | ------------------------------ | --------------------------------- |
| `*-state.json`         | `dedup-state.json`             | Dedup / activity cache            |
| `*-checkpoint.json`    | `compaction-checkpoint.json`   | Resumable-run checkpoint          |
| `*.lock`               | `workspace.lock`               | Exclusivity primitive             |

Suffix rules beat prefix rules: `compaction-checkpoint.json` is
**tracked** (suffix wins) even though `compaction-` is an append-only
prefix.

### `maintenance/append-only/` — snapshot-excluded

Observability output. Append-only semantically, which means rolling
it back would discard real signal (validation errors logged during
the failed apply, compaction progress, intel-scan findings).

Classification:

| Pattern                       | Example                                |
| ----------------------------- | -------------------------------------- |
| `*-report.txt` (suffix)       | `validation-report.txt`                |
| `*.log` (suffix)              | `compaction-2026-04-20.log`            |
| `*.ndjson` (suffix)           | `intel-scan-2026-04-20.ndjson`         |
| `compaction-*` (prefix)       | `compaction-summary.txt`               |
| `validation-*` (prefix)       | `validation-errors.txt`                |
| `intel-scan-*` (prefix)       | `intel-scan-index.md`                  |

### Unknown files

When no rule matches, the classifier defaults to **tracked**. Two
reasons:

1. Missing a behavioural-state file from the snapshot is worse than
   snapshotting an append-only file unnecessarily — the first is a
   correctness bug, the second is just snapshot bloat.
2. New file types introduced after v3.2.0 will classify correctly
   the moment a maintainer adds a pattern here; the tracked-default
   means they're safe in the meantime.

## Atomicity contract

Given the split, the apply-engine guarantees:

1. Every file in `maintenance/tracked/` present at snapshot time is
   restored to its snapshot content on rollback.
2. Every file in `maintenance/tracked/` created between snapshot and
   rollback is removed on rollback (orphan cleanup).
3. Every file in `maintenance/append-only/` is left untouched by
   rollback. Content appended during the failed apply survives.
4. Every file in `intelligence/applied/` is untouched by rollback
   (prevents recursive snapshot deletion).

The orphan-cleanup walk explicitly honors these exclusions — file
deletion in step 2 never descends into `maintenance/append-only/`
or `intelligence/applied/`. A repro test
(`tests/test_atomicity_maintenance_scope.py`) pins all four
invariants.

## Migration from the v3.1.x layout

Workspaces created on v3.1.x or earlier have `maintenance/<file>`
directly under the top level. v3.2.0 ships a one-shot migration
helper, `mind_mem.maintenance_migrate.migrate_maintenance`, that:

1. Creates `maintenance/tracked/` and `maintenance/append-only/` if
   they don't exist yet.
2. Walks every file directly under `maintenance/` (ignoring files
   already in the new subdirectories).
3. Classifies each via `classify_maintenance_file(basename)`.
4. Moves each file into the matching subdirectory. Never overwrites —
   if a destination name is taken, the file is renamed with a
   `.<n><ext>` suffix before the move.
5. Emits a one-line audit per move on stderr so operators can review.

The helper is idempotent. Second and subsequent calls are no-ops
(they detect the new layout and skip immediately). It runs
automatically on the first apply in a v3.2.0+ workspace; it can also
be invoked manually:

```bash
python3 -m mind_mem.maintenance_migrate /path/to/workspace
```

## Adding a new `maintenance/` file

When you introduce a new file kind under `maintenance/`:

1. Decide which semantic it has — behavioural state (tracked) or
   append-only signal (append-only). When in doubt, behavioural
   state is the safer default.
2. Add a matching pattern to either `_TRACKED_SUFFIXES` /
   `_APPEND_ONLY_SUFFIXES` / `_APPEND_ONLY_PREFIXES` in
   `src/mind_mem/maintenance_migrate.py`.
3. Add a test case for the new pattern in
   `tests/test_maintenance_migrate.py::test_classify`.
4. Create the file under the correct subdirectory from the start.
   Files written directly to `maintenance/<name>` will still be
   migrated automatically on the next apply, but it's cleaner to
   write to the right path from the outset.

## Deprecation timeline

The top-level-maintenance layout continues to auto-migrate throughout
v3.2.x. v4.0 removes the migration helper and will treat a file
directly under `maintenance/` as a configuration error.

## References

- `src/mind_mem/corpus_registry.py` — `SNAPSHOT_DIRS`,
  `SNAPSHOT_EXCLUDE_DIRS` definitions.
- `src/mind_mem/apply_engine.py` — snapshot / restore /
  orphan-cleanup walk honors the exclusions.
- `src/mind_mem/maintenance_migrate.py` — classifier + migration
  helper.
- `tests/test_maintenance_migrate.py` — classifier + migration tests.
- `tests/test_atomicity_maintenance_scope.py` — atomicity invariants.
- `docs/v3.2.0-atomicity-scope-plan.md` — design discussion.
- `SPEC.md` §"Atomicity Rules", clause 7.
