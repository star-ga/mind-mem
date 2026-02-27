# Migration Guide

## Migrating from mem-os to mind-mem

mind-mem is the successor to mem-os. This guide covers the migration path.

### Key Differences

| Feature | mem-os | mind-mem |
|---------|--------|----------|
| Dependencies | Multiple | Zero (core) |
| Search | Basic BM25 | Hybrid BM25 + vector + RRF |
| MCP Tools | 8 | 18 |
| Scoring | Fixed | MIND kernel-based |
| Audit | Basic | Full contradiction/drift scan |

### Migration Steps

1. **Export mem-os data:**
   ```bash
   python3 -m mem_os export --format jsonl > mem-os-export.jsonl
   ```

2. **Initialize mind-mem workspace:**
   ```bash
   python3 -c "from mind_mem.init_workspace import init; init('/path/to/workspace')"
   ```

3. **Import data:**
   Copy exported blocks to the appropriate workspace directories:
   - Decisions → `workspace/decisions/`
   - Tasks → `workspace/tasks/`
   - Entities → `workspace/entities/`

4. **Rebuild indexes:**
   ```python
   from mind_mem.sqlite_index import rebuild_fts_index
   rebuild_fts_index('/path/to/workspace')
   ```

5. **Verify:**
   ```python
   from mind_mem.block_parser import parse_file
   blocks = parse_file('/path/to/workspace/decisions/imported.md')
   print(f"Imported {len(blocks)} blocks")
   ```

### MCP Tool Mapping

| mem-os Tool | mind-mem Tool | Notes |
|-------------|---------------|-------|
| search | recall | Enhanced with graph boost |
| add_memory | propose_update | Now requires approval |
| delete | delete_memory_item | Same behavior |
| list | category_summary | Grouped by category |
| — | hybrid_search | New: vector + BM25 fusion |
| — | scan | New: integrity checking |
| — | prefetch | New: proactive context |
