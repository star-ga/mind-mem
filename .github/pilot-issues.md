# Pilot Week Issues (Feb 19-25)

Wire three existing but disconnected modules into the main recall pipeline.

---

## Issue 1: Wire A-MEM block metadata into recall pipeline

**Labels:** `pilot-week`, `integration`, `recall-pipeline`

### Background

`scripts/block_metadata.py` implements `BlockMetadataManager` -- an A-MEM-style module that tracks per-block access patterns, evolves keywords via query feedback, computes importance scores (clamped to `[0.8, 1.5]`), and records co-occurrence connections. The module is fully implemented and tested in isolation but has **zero references** inside `scripts/recall.py`. None of its capabilities currently influence recall scoring or learn from retrieval activity.

### Current State

- `BlockMetadataManager` is initialized with a `db_path` (SQLite) and creates a `block_meta` table.
- Key API surface:
  - `get_importance_boost(block_id) -> float` -- returns a `[0.8, 1.5]` multiplier from stored importance.
  - `update_importance(block_id, decay_days=30) -> float` -- recalculates importance from access frequency + recency + connection degree, stores it, returns the value.
  - `record_access(block_ids, query="")` -- increments `access_count`, updates `last_accessed`, records co-occurrence pairs between all returned block IDs.
  - `evolve_keywords(block_id, query_tokens, block_content="", max_keywords=20)` -- adds query tokens found in block content to the block's keyword set.
- `recall.py` has no import of `block_metadata` and no usage of any metadata manager.

### Goal

After BM25 scoring produces per-block scores (around line 1860 in `recall()`, after the boost factors section and before results are appended to the list), multiply each result's `score` by its A-MEM importance factor. After the final top-K results are selected (after line 2297, before `context_pack`), call `record_access()` with the returned block IDs and the original query string.

### Implementation Steps

1. **Import and instantiate `BlockMetadataManager`** at the top of `recall()`.
   - Derive `db_path` from workspace: `os.path.join(workspace, ".mind-mem", "block_meta.db")`.
   - Wrap in try/except for graceful degradation if the directory does not exist.

2. **Apply importance boost** after BM25 + boost factors scoring (after the `priority` boost around line 1911, before building the result dict at line 1917):
   ```python
   if meta_mgr:
       importance = meta_mgr.get_importance_boost(block.get("_id", ""))
       score *= importance
   ```

3. **Record access on final results** after the `top = deduped[:limit]` line (line 2297), before `context_pack`:
   ```python
   if meta_mgr:
       returned_ids = [r["_id"] for r in top]
       meta_mgr.record_access(returned_ids, query=query)
   ```

4. **Optionally call `evolve_keywords()`** on each returned block to grow its keyword set over time:
   ```python
   if meta_mgr:
       for r in top:
           meta_mgr.evolve_keywords(r["_id"], query_tokens, r.get("excerpt", ""))
   ```

5. **Add tests** in `tests/test_recall_metadata.py`:
   - Test that importance boost modifies scores (mock `BlockMetadataManager`).
   - Test that `record_access` is called with correct block IDs.
   - Test graceful degradation when db_path directory is missing.
   - Test that a block with importance 1.5 outranks an otherwise-equal block with importance 0.8.

### Acceptance Criteria

- [ ] `BlockMetadataManager` is imported and initialized in `recall()` with graceful fallback.
- [ ] Each BM25 result's score is multiplied by `get_importance_boost()` (range `[0.8, 1.5]`).
- [ ] `record_access()` is called on the final returned block IDs with the query string.
- [ ] `evolve_keywords()` is called on returned blocks to grow keyword sets.
- [ ] New tests cover the integration (importance boost, access tracking, fallback).
- [ ] All existing 696 tests still pass.

### Files to Modify

- `scripts/recall.py` -- import `BlockMetadataManager`, apply boost, call `record_access` + `evolve_keywords`
- `tests/test_recall_metadata.py` -- new test file for integration tests

---

## Issue 2: Wire intent router into recall pipeline

**Labels:** `pilot-week`, `integration`, `recall-pipeline`

### Background

`scripts/intent_router.py` implements `IntentRouter` -- a regex-based query classifier that maps queries into 9 intent types (`WHY`, `WHEN`, `ENTITY`, `WHAT`, `HOW`, `LIST`, `VERIFY`, `COMPARE`, `TRACE`), each with retrieval parameter overrides (`expansion` mode, `graph_depth`, `rerank` strategy). Meanwhile, `recall.py` has its own `detect_query_type()` function (line 795) that classifies into only 5 coarser categories (`temporal`, `adversarial`, `multi-hop`, `single-hop`, `open-domain`) using separate compiled regex patterns. The `IntentRouter` is more granular and returns structured `IntentResult` objects with confidence scores and sub-intents, but is never imported or called from `recall.py`.

### Current State

- `detect_query_type(query)` (line 795) returns a string from `{'temporal', 'adversarial', 'multi-hop', 'single-hop', 'open-domain'}`.
- The return value is used to look up `_QUERY_TYPE_PARAMS` (line 1715) which controls `recency_weight`, `date_boost`, `expand_query`, `extra_limit_factor`, and `graph_boost_override`.
- `query_type` is referenced throughout `recall()` at lines 1714, 1722, 1944, 1997, 2102, 2128, 2183, and 2302.
- `IntentRouter.classify(query)` returns `IntentResult(intent, confidence, sub_intents, params)` where `params` contains `expansion`, `graph_depth`, and `rerank` keys.
- `IntentRouter` also provides `classify_with_fallback()` which falls back to `detect_query_type()` when confidence < 0.3.
- `intent_router.py` has a singleton accessor: `get_router()`.

### Goal

Replace the call to `detect_query_type()` (line 1714) with `IntentRouter`, mapping IntentRouter's 9-type output back to the existing 5 `_QUERY_TYPE_PARAMS` categories for backward compatibility, while also passing through the richer `params` dict for downstream use. The existing `detect_query_type()` function should be preserved (not deleted) since `IntentRouter.classify_with_fallback()` depends on it.

### Implementation Steps

1. **Add import** at top of `recall.py`:
   ```python
   from intent_router import IntentRouter, get_router
   ```

2. **Create an intent-to-query-type mapping** to preserve backward compatibility with `_QUERY_TYPE_PARAMS`:
   ```python
   _INTENT_TO_QUERY_TYPE = {
       "WHY": "multi-hop",
       "WHEN": "temporal",
       "ENTITY": "single-hop",
       "WHAT": "single-hop",
       "HOW": "single-hop",
       "LIST": "open-domain",
       "VERIFY": "adversarial",
       "COMPARE": "multi-hop",
       "TRACE": "multi-hop",
   }
   ```

3. **Replace the `detect_query_type` call** at line 1714:
   ```python
   # Old:
   # query_type = detect_query_type(query)

   # New:
   try:
       intent_result = get_router().classify(query)
       query_type = _INTENT_TO_QUERY_TYPE.get(intent_result.intent, "single-hop")
       intent_params = intent_result.params  # expansion, graph_depth, rerank
   except Exception:
       query_type = detect_query_type(query)
       intent_params = {}
   ```

4. **Apply `intent_params` overrides** where appropriate:
   - If `intent_params.get("graph_depth", 0) >= 2`, force `graph_boost = True` (analogous to the existing `graph_boost_override`).
   - Log the intent classification for observability: `_log.info("intent_classified", intent=intent_result.intent, confidence=intent_result.confidence)`.

5. **Preserve `detect_query_type()`** -- do not delete it. It serves as a fallback inside `classify_with_fallback()` and may be needed for benchmark reproducibility.

6. **Add tests** in `tests/test_recall_intent_router.py`:
   - Test that different query shapes route to expected `query_type` values via the mapping.
   - Test graceful fallback to `detect_query_type()` on `IntentRouter` import failure.
   - Test backward compatibility: temporal queries still get `recency_weight=0.6`, adversarial queries still suppress semantic expansion.
   - Test that `graph_depth >= 2` triggers graph boost.

### Acceptance Criteria

- [ ] `IntentRouter` is imported and called in `recall()` instead of `detect_query_type()`.
- [ ] 9 intent types are mapped back to the 5 `_QUERY_TYPE_PARAMS` categories for backward compatibility.
- [ ] `detect_query_type()` is preserved (not deleted) for fallback and benchmark use.
- [ ] `intent_params` (expansion mode, graph_depth, rerank strategy) are passed through and logged.
- [ ] `graph_depth >= 2` from IntentRouter triggers graph boost.
- [ ] Graceful fallback to `detect_query_type()` on any exception.
- [ ] New tests cover routing, mapping, fallback, and backward compatibility.
- [ ] All existing 696 tests still pass.

### Files to Modify

- `scripts/recall.py` -- import `IntentRouter`, add `_INTENT_TO_QUERY_TYPE` mapping, replace classification call
- `tests/test_recall_intent_router.py` -- new test file for integration tests

---

## Issue 3: Wire cross-encoder reranker into recall pipeline

**Labels:** `pilot-week`, `integration`, `recall-pipeline`

### Background

`scripts/cross_encoder_reranker.py` implements `CrossEncoderReranker` -- an optional neural reranking stage using `cross-encoder/ms-marco-MiniLM-L-6-v2` (80MB, CPU-friendly). It blends cross-encoder scores with original BM25 scores using a configurable `blend_weight` (default 0.6). The module handles its own model singleton, normalizes both score distributions to `[0, 1]` before blending, and provides `is_available()` to check for the `sentence-transformers` dependency. Despite being fully implemented, it is never imported or called from `recall.py`.

### Current State

- `CrossEncoderReranker` API:
  - `CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")` -- raises `ImportError` if `sentence-transformers` not installed.
  - `rerank(query, candidates, top_k=10, blend_weight=0.6) -> list[dict]` -- expects candidates with `content` or `text` key and `score` key. Returns reranked list with blended `score` and added `ce_score` field.
  - `CrossEncoderReranker.is_available() -> bool` -- static method, checks if `sentence-transformers` is importable.
- `recall.py` has a two-stage pipeline (line 2264):
  1. Wide BM25 retrieve (200 candidates) -> dedup
  2. Deterministic `rerank_hits()` (line 2294) -> top-K
  3. `context_pack()` (line 2300) -> final output
- The cross-encoder should slot in as an optional stage between the deterministic rerank and final output.
- Config file (`mind-mem.example.json`) currently has a `recall` section but no `cross_encoder` key.

### Goal

Add an optional cross-encoder reranking stage gated by `recall.cross_encoder.enabled` in `mind-mem.json` (default `false`). When enabled and `sentence-transformers` is available, apply cross-encoder reranking after the deterministic `rerank_hits()` stage (line 2294) and before final `context_pack()` (line 2300). Gracefully degrade when the dependency is missing.

### Implementation Steps

1. **Add config loading** -- extend the existing config loading section in `recall()` (around line 2000 where `rm3_config` is loaded) to also read cross-encoder config:
   ```python
   ce_config = _cfg.get("recall", {}).get("cross_encoder", {})
   ```
   Extract: `enabled` (bool, default False), `blend_weight` (float, default 0.6), `top_k` (int, default same as `limit`).

2. **Add conditional cross-encoder stage** after deterministic rerank (line 2294), before `top = deduped[:limit]` (line 2297):
   ```python
   if ce_config.get("enabled", False):
       try:
           from cross_encoder_reranker import CrossEncoderReranker
           if CrossEncoderReranker.is_available():
               ce = CrossEncoderReranker()
               # Prepare candidates: need "content" key for cross-encoder
               for r in deduped:
                   if "content" not in r:
                       r["content"] = r.get("excerpt", "")
               deduped = ce.rerank(
                   query, deduped,
                   top_k=ce_config.get("top_k", limit),
                   blend_weight=ce_config.get("blend_weight", 0.6),
               )
               _log.info("cross_encoder_rerank", candidates=len(deduped),
                         blend_weight=ce_config.get("blend_weight", 0.6))
       except (ImportError, Exception) as e:
           _log.warning("cross_encoder_unavailable", error=str(e))
   ```

3. **Update `mind-mem.example.json`** to document the new config key:
   ```json
   {
     "recall": {
       "cross_encoder": {
         "enabled": false,
         "blend_weight": 0.6,
         "top_k": 10
       }
     }
   }
   ```

4. **Add tests** in `tests/test_recall_cross_encoder.py`:
   - Test that cross-encoder is NOT called when config `enabled=false` (default).
   - Test that cross-encoder IS called when config `enabled=true` and `sentence-transformers` is available (mock `CrossEncoderReranker`).
   - Test graceful fallback when `sentence-transformers` is not installed (`is_available()` returns False).
   - Test graceful fallback when `CrossEncoderReranker.rerank()` raises an exception.
   - Test that `blend_weight` and `top_k` config values are passed through correctly.
   - Test that the `content` key is populated from `excerpt` for candidates missing it.

### Acceptance Criteria

- [ ] Cross-encoder reranking is gated by `recall.cross_encoder.enabled` config key (default `false`).
- [ ] When enabled, `CrossEncoderReranker.rerank()` is called after deterministic reranking, before `context_pack()`.
- [ ] `blend_weight` (default 0.6) and `top_k` (default = limit) are configurable via `mind-mem.json`.
- [ ] Graceful fallback when `sentence-transformers` is not installed -- recall pipeline continues without error.
- [ ] Graceful fallback on any cross-encoder exception -- logged as warning, pipeline continues.
- [ ] `mind-mem.example.json` updated with `cross_encoder` config section.
- [ ] New tests cover enabled/disabled, available/unavailable, exception handling, and config passthrough.
- [ ] All existing 696 tests still pass.

### Files to Modify

- `scripts/recall.py` -- load cross-encoder config, add conditional reranking stage after deterministic rerank
- `mind-mem.example.json` -- add `cross_encoder` config section under `recall`
- `tests/test_recall_cross_encoder.py` -- new test file for integration tests
