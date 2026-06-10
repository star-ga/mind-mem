# mind-mem-4b — v3.9.0 retrain plan

> **Status:** drafted 2026-05-04 after v3.9.0 PyPI release. Schema delta from v3.0.0-era fine-tune is large enough to warrant a full retrain rather than continued LoRA on the existing checkpoint.

## Why retrain now

The previous fine-tune (`star-ga/mind-mem-4b`, Q4_K_M) was trained against the v3.0.0 surface: **57 MCP tools**, no `TransformHash` field, single Markdown / sqlite-vec backends, no HTTP transport, no inbox, no daemon, no walkthrough, no personas.

v3.9.0 surface (today):

| Axis | v3.0.0 | v3.9.0 | Delta |
|------|--------|--------|-------|
| MCP tools | 57 | **81** | +24 (incl. `compile_truth_walkthrough`, `recall_with_persona`, `pipeline_status`, `reindex_dirty`, MIC/MAP, model audit/sign, audit-pinned, governance hooks, kernels, etc.) |
| Block fields | base | base + `TransformHash` | +1 schema field |
| Transports | MCP only | MCP + HTTP + inbox + daemon | +3 surfaces |
| Backends | Markdown, sqlite-vec | Markdown, sqlite-vec, **replicated Postgres** | +1 routing layer |
| Personas | none | brief / detailed / technical | +3 projection modes |

**Symptom of drift if we don't retrain:** the model invents tool names that don't exist (it remembers `add_memory` from v2.x, doesn't know `compile_truth_walkthrough`), recommends `transform_hash` lowercase (Postgres-only) instead of `TransformHash` (Markdown-canonical), can't reason about HTTP/daemon transports.

## Pipeline (already on disk)

The training scaffolding shipped in `train/` (last touched 2026-04-18):

```
build_corpus.py → train_qlora.py → eval_harness.py → export_gguf.py → upload_to_hf.py
```

Output landing pad: `/data/checkpoints/mm-workspace/train-output/`.

## Blocker — build_corpus.py is stale

The current corpus harvester walks `src/mind_mem/mcp_server.py` only. As of v3.4 the tools were extracted into `src/mind_mem/mcp/tools/*.py` and registered through per-module `register()` callbacks. **Result: the existing builder sees 1 tool, not 81.** Any retrain run today would produce a corpus that teaches the model the v3.0 surface again.

### Fix (15-min change)

1. In `train/build_corpus.py`, replace the single-file AST walk with:

   ```python
   tools_dir = REPO / "src" / "mind_mem" / "mcp" / "tools"
   sources = [REPO / "src" / "mind_mem" / "mcp_server.py"]
   sources.extend(p for p in tools_dir.glob("*.py") if p.name not in ("__init__.py", "_helpers.py"))
   for path in sources:
       tree = ast.parse(path.read_text(encoding="utf-8"))
       ...
   ```

2. Update the decorator detector to match `@mcp_tool_observe` (the new wrapper) in addition to `@mcp.tool` / `@tool`:

   ```python
   def _is_tool_decorator(d):
       if isinstance(d, ast.Attribute) and d.attr == "tool": return True
       if isinstance(d, ast.Name) and d.id in ("tool", "mcp_tool_observe"): return True
       return False
   ```

3. Update `SYSTEM_PROMPT` count: `"57 MCP tools"` → `"81 MCP tools"`.

4. Add a v3.9.0 `TransformHash` schema example to the block-grammar source.

5. Bump `CHANGELOG_LIMIT` (or remove the cap) so v3.4–v3.9 entries get harvested.

## Hardware + budget

- **GPU:** RTX 3080 10 GB VRAM (local). QLoRA on Qwen3.5-4B fits.
- **Wall time:** 2–4 hours for ~600 examples × 3 epochs (current corpus is ~393; +200 from new tools).
- **Disk:** ~20 GB in `~/.cache/huggingface/hub` (already populated) + `/data/checkpoints/mm-workspace/train-output/` (currently 0 free of 250 GB).
- **Cost:** $0 — local box.

If wall-time matters, the same script runs on Runpod H200 (`benchmarks/train_mind_mem_4b.py` is the H200 full-fine-tune variant). H200 takes ~30 min, ~$3 spot.

## Eval gates (already in `eval_harness.py`)

Exit non-zero if any of:
- tool-call accuracy < 95%
- block-schema validity < 98%
- end-to-end governance workflow < 90%

For v3.9 we also want:
- v3.9-specific tool name recall (the 24 new tools) ≥ 90%
- `TransformHash` field cited in block-write examples ≥ 95%
- HTTP/inbox/daemon transport mentions don't hallucinate endpoints

I'll add three new probes to `eval_harness.py` to cover these — separate small commit.

## Rollout

1. Train → eval gate → export GGUF (Q4_K_M, ~2.7 GB).
2. Push to HF as `star-ga/mind-mem-4b` revision `v3.9.0` (overwrite default branch only after eval green; previous v3.0 fine-tune stays accessible via `revision="v3.0.0"`).
3. Update `Modelfile.mind-mem-4b` to point at the new GGUF.
4. `ollama create mind-mem:4b-v3.9 -f Modelfile.mind-mem-4b-v3.9` → smoke test.
5. Update `mind-mem.json` default `extraction.model` from `mind-mem:4b` to `mind-mem:4b-v3.9` (with a brief grace window where both aliases resolve).

## Non-goals

- No architecture change (still Qwen3.5-4B base, full FT).
- No pretraining-data change.
- No new prompt template — the SYSTEM_PROMPT only grows the tool-count field; everything else stays.

## Order of operations

1. Patch `build_corpus.py` (3 small edits above).
2. Add v3.9 probes to `eval_harness.py`.
3. Run `build_corpus.py` → diff against the v3.0 corpus to confirm the 24 new tools surface.
4. `train_qlora.py` (overnight on the 3080, or 30 min on H200).
5. `eval_harness.py` — gate.
6. `export_gguf.py` → `upload_to_hf.py` → smoke test in Ollama.

## What I'm not committing here

The corpus-builder patch is the only pre-condition that needs a code change. Everything else is execution. **I'm not kicking off training until you confirm** — RTX 3080 is the only local GPU, and an overnight training run blocks other GPU work (other local testing, llama-server, etc.). Cheaper to defer to the next idle window or send to Runpod H200.
