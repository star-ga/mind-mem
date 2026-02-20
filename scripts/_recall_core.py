"""Recall engine core — RecallBackend, main BM25 pipeline, backend loading, prefetch, CLI."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _recall_constants import (
    _STOPWORDS,
    _VALID_RECALL_KEYS,
    BM25_B,
    BM25_K1,
    CORPUS_FILES,
    FIELD_WEIGHTS,
    GRAPH_BOOST_FACTOR,
    MAX_BLOCKS_PER_QUERY,
    MAX_GRAPH_NEIGHBORS_PER_HOP,
    MAX_RERANK_CANDIDATES,
)
from _recall_context import context_pack
from _recall_detection import (
    _INTENT_TO_QUERY_TYPE,
    _QUERY_TYPE_PARAMS,
    _parse_speaker_from_tags,
    chunk_text,
    detect_query_type,
    extract_field_tokens,
    get_bigrams,
    get_block_type,
    get_excerpt,
    is_skeptical_query,
)
from _recall_expansion import expand_months, expand_query, rm3_expand
from _recall_reranking import rerank_hits
from _recall_scoring import build_xref_graph, date_score
from _recall_tokenization import tokenize
from block_parser import get_active, parse_file
from observability import get_logger, metrics

# A-MEM block metadata (optional — graceful degradation if unavailable)
try:
    from block_metadata import BlockMetadataManager
    _HAS_BLOCK_META = True
except ImportError:
    _HAS_BLOCK_META = False

# Intent Router (optional — falls back to detect_query_type)
try:
    from intent_router import get_router as _get_intent_router
    _HAS_INTENT_ROUTER = True
except ImportError:
    _HAS_INTENT_ROUTER = False

# LLM Extractor (optional — config-gated, zero deps by default)
try:
    from llm_extractor import enrich_results as _llm_enrich_results
    _HAS_LLM_EXTRACTOR = True
except ImportError:
    _HAS_LLM_EXTRACTOR = False

_log = get_logger("recall")

# Log optional subsystem availability at import time (#5: hidden coupling)
if not _HAS_BLOCK_META:
    _log.info("optional_subsystem_unavailable", subsystem="block_metadata",
              impact="A-MEM importance boost disabled")
if not _HAS_INTENT_ROUTER:
    _log.info("optional_subsystem_unavailable", subsystem="intent_router",
              impact="falling back to detect_query_type()")
if not _HAS_LLM_EXTRACTOR:
    _log.info("optional_subsystem_unavailable", subsystem="llm_extractor",
              impact="LLM enrichment disabled")


__all__ = [
    "RecallBackend",
    "recall", "_load_backend", "prefetch_context", "main",
]


# ---------------------------------------------------------------------------
# RecallBackend interface — plug in vector/semantic backends here
# ---------------------------------------------------------------------------

class RecallBackend(ABC):
    """Interface for recall backends. Default: BM25Backend (below).

    To add a vector backend:
    1. Implement this interface in recall_vector.py
    2. Set recall.backend = "vector" in mind-mem.json
    3. recall.py will load it dynamically, falling back to BM25 on error.
    """

    @abstractmethod
    def search(self, workspace, query, limit=10, active_only=False):
        """Return list of {_id, type, score, excerpt, file, line, status}."""
        ...

    @abstractmethod
    def index(self, workspace):
        """(Re)build index from workspace files."""
        ...


def recall(
    workspace: str, query: str, limit: int = 10, active_only: bool = False,
    graph_boost: bool = False, agent_id: str | None = None,
    retrieve_wide_k: int = 200, rerank: bool = True,
    rerank_debug: bool = False,
) -> list[dict]:
    """Search across all memory files using BM25 scoring. Returns ranked results.

    Args:
        workspace: Workspace root path.
        query: Search query.
        limit: Max results to return (final top-k after reranking).
        active_only: Only return blocks with active status.
        graph_boost: Enable cross-reference neighbor boosting.
        agent_id: Optional agent ID for namespace ACL filtering.
        retrieve_wide_k: Number of candidates to retrieve before reranking.
        rerank: Enable deterministic reranking (v7).
        rerank_debug: Log reranker feature breakdowns.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # Load .mind kernel overrides if available
    _kernel_bm25_k1 = BM25_K1
    _kernel_bm25_b = BM25_B
    _kernel_field_weights = None
    try:
        from mind_ffi import get_kernel_param, get_mind_dir, load_kernel_config
        mind_dir = get_mind_dir(workspace)
        recall_kernel = load_kernel_config(os.path.join(mind_dir, "recall.mind"))
        if recall_kernel:
            _kernel_bm25_k1 = get_kernel_param(recall_kernel, "bm25", "k1", BM25_K1)
            _kernel_bm25_b = get_kernel_param(recall_kernel, "bm25", "b", BM25_B)
            fields_section = recall_kernel.get("fields", {})
            if fields_section:
                _kernel_field_weights = fields_section
    except ImportError:
        _log.debug("mind_ffi_unavailable", hint="MIND kernels not installed")
    except Exception as e:
        _log.warning("mind_kernel_load_failed", error=str(e))

    # --- A-MEM block metadata manager (optional) ---
    meta_mgr = None
    if _HAS_BLOCK_META:
        try:
            meta_db = os.path.join(workspace, ".mind-mem", "block_meta.db")
            meta_dir = os.path.dirname(meta_db)
            if os.path.isdir(meta_dir):
                meta_mgr = BlockMetadataManager(meta_db)
        except Exception as e:
            _log.warning("block_metadata_init_failed", error=str(e))

    # --- Intent classification ---
    intent_params = {}
    if _HAS_INTENT_ROUTER:
        try:
            intent_result = _get_intent_router().classify(query)
            query_type = _INTENT_TO_QUERY_TYPE.get(intent_result.intent, "single-hop")
            intent_params = intent_result.params
            _log.info("intent_classified", intent=intent_result.intent,
                      confidence=intent_result.confidence, query_type=query_type)
        except Exception as e:
            _log.warning("intent_classification_failed", error=str(e))
            query_type = detect_query_type(query)
    else:
        query_type = detect_query_type(query)
    qparams = _QUERY_TYPE_PARAMS.get(query_type, _QUERY_TYPE_PARAMS["single-hop"])

    # Month normalization: inject numeric month tokens for date matching
    query_tokens = expand_months(query, query_tokens)

    # Skeptical mode: for distractor-prone queries, keep morph-only tokens
    # separate to penalize expansion-only matches later.
    skeptical = is_skeptical_query(query) and query_type in ("adversarial", "single-hop")

    # Query expansion: add domain synonyms
    # adversarial/verification queries use morph_only (no semantic synonyms)
    expand_mode = qparams.get("expand_query", True)
    if expand_mode:
        mode = expand_mode if isinstance(expand_mode, str) else "full"
        # In skeptical mode, force morph_only to suppress semantic drift
        if skeptical:
            mode = "morph_only"
        query_tokens = expand_query(query_tokens, mode=mode)

    # Force graph boost for multi-hop queries or high intent graph_depth
    if qparams.get("graph_boost_override", False) or intent_params.get("graph_depth", 0) >= 2:
        graph_boost = True

    # Adjust effective limit for retrieval (retrieve more candidates, trim later)
    limit = int(limit * qparams.get("extra_limit_factor", 1.0))

    # Namespace ACL: resolve accessible paths if agent_id is provided
    ns_manager = None
    if agent_id:
        try:
            from namespaces import NamespaceManager
            ns_manager = NamespaceManager(workspace, agent_id=agent_id)
        except ImportError:
            _log.debug("namespaces_unavailable", agent_id=agent_id)

    # Load all blocks with source file tracking
    all_blocks = []
    for label, rel_path in CORPUS_FILES.items():
        # ACL check: skip files the agent cannot read
        if ns_manager and not ns_manager.can_read(rel_path):
            continue

        path = os.path.join(workspace, rel_path)
        if not os.path.isfile(path):
            continue
        try:
            blocks = parse_file(path)
        except (OSError, UnicodeDecodeError, ValueError) as e:
            _log.debug("corpus_parse_failed", file=rel_path, error=str(e))
            continue
        if active_only:
            blocks = get_active(blocks)
        for b in blocks:
            b["_source_file"] = rel_path
            b["_source_label"] = label
            all_blocks.append(b)

    # If agent has namespace, also search agent-private corpus files
    if ns_manager and agent_id:
        agent_ns = f"agents/{agent_id}"
        for label, rel_path in CORPUS_FILES.items():
            ns_path = os.path.join(agent_ns, rel_path)
            full_path = os.path.join(workspace, ns_path)
            if not os.path.isfile(full_path):
                continue
            if not ns_manager.can_read(ns_path):
                continue
            try:
                blocks = parse_file(full_path)
            except (OSError, UnicodeDecodeError, ValueError) as e:
                _log.debug("corpus_parse_failed", file=ns_path, error=str(e))
                continue
            if active_only:
                blocks = get_active(blocks)
            for b in blocks:
                b["_source_file"] = ns_path
                b["_source_label"] = f"{label}@{agent_id}"
                all_blocks.append(b)

    if not all_blocks:
        return []

    # Cap blocks to prevent memory/latency blowup on huge workspaces (#15)
    if len(all_blocks) > MAX_BLOCKS_PER_QUERY:
        _log.warning("blocks_capped", total=len(all_blocks),
                     cap=MAX_BLOCKS_PER_QUERY,
                     hint="consider using FTS5 index for large workspaces")
        all_blocks = all_blocks[:MAX_BLOCKS_PER_QUERY]

    # --- Kernel overrides: local aliases for BM25 params ---
    _fw = _kernel_field_weights if _kernel_field_weights else FIELD_WEIGHTS
    _k1 = float(_kernel_bm25_k1)
    _b = float(_kernel_bm25_b)

    # --- BM25F: per-field tokenization + flat token list for IDF ---
    doc_field_tokens = []   # [{field: [tokens]}] per block
    doc_flat_tokens = []    # [[all_tokens]] per block (for IDF + bigrams)
    for block in all_blocks:
        ft = extract_field_tokens(block)
        doc_field_tokens.append(ft)
        flat = []
        for tokens in ft.values():
            flat.extend(tokens)
        doc_flat_tokens.append(flat)

    # Document frequency + average weighted doc length
    df = Counter()
    total_wdl = 0.0
    for i, ft in enumerate(doc_field_tokens):
        seen = set()
        wdl = 0.0
        for field, tokens in ft.items():
            w = _fw.get(field, 1.0)
            wdl += len(tokens) * w
            for t in tokens:
                seen.add(t)
        for t in seen:
            df[t] += 1
        total_wdl += wdl

    N = len(all_blocks)
    avg_wdl = total_wdl / N if N > 0 else 1.0

    # Pre-compute query bigrams for phrase matching
    query_bigrams = get_bigrams(query_tokens)

    # Pre-compute IDF per query token (constant across all docs)
    _idf_cache = {}
    for qt in query_tokens:
        _df_qt = df.get(qt, 0)
        _idf_cache[qt] = math.log((N - _df_qt + 0.5) / (_df_qt + 0.5) + 1)

    results = []

    for i, block in enumerate(all_blocks):
        ft = doc_field_tokens[i]
        flat = doc_flat_tokens[i]
        if not flat:
            continue

        # --- BM25F: field-weighted term frequency ---
        # Compute weighted TF across all fields
        weighted_tf = Counter()
        wdl = 0.0
        for field, tokens in ft.items():
            w = _fw.get(field, 1.0)
            wdl += len(tokens) * w
            for t in tokens:
                weighted_tf[t] += w

        score = 0.0
        for qt in query_tokens:
            if qt in weighted_tf:
                wtf = weighted_tf[qt]
                idf = _idf_cache[qt]
                numerator = wtf * (_k1 + 1)
                denominator = wtf + _k1 * (1 - _b + _b * wdl / avg_wdl)
                score += idf * numerator / denominator

        if score <= 0:
            continue

        # --- Bigram phrase matching boost ---
        if query_bigrams:
            doc_bigrams = get_bigrams(flat)
            phrase_matches = len(query_bigrams & doc_bigrams)
            if phrase_matches > 0:
                score *= (1.0 + 0.25 * phrase_matches)

        # --- Chunking boost: score best chunk separately, blend ---
        # For long blocks, check if a chunk scores much higher
        statement = block.get("Statement", "") or block.get("Title", "") or ""
        if len(statement) > 200:
            chunks = chunk_text(statement)
            if len(chunks) > 1:
                best_chunk_score = 0.0
                for chunk in chunks:
                    ctokens = tokenize(chunk)
                    ctf = Counter(ctokens)
                    cdl = len(ctokens)
                    cs = 0.0
                    for qt in query_tokens:
                        if qt in ctf:
                            freq = ctf[qt]
                            denom = freq + _k1 * (1 - _b + _b * cdl / max(avg_wdl, 1))
                            cs += _idf_cache[qt] * freq * (_k1 + 1) / denom
                    best_chunk_score = max(best_chunk_score, cs)
                # Blend: take the better of full-block or best-chunk score
                if best_chunk_score > score:
                    score = 0.6 * best_chunk_score + 0.4 * score

        # --- Boost factors (query-type-aware) ---
        recency = date_score(block)
        rw = qparams.get("recency_weight", 0.3)
        score *= (1.0 - rw + rw * recency)

        # Temporal queries: boost blocks that contain dates
        date_boost_val = qparams.get("date_boost", 1.0)
        if date_boost_val > 1.0 and block.get("Date", ""):
            score *= date_boost_val

        status = block.get("Status", "")
        if status == "active":
            score *= 1.2
        elif status in ("todo", "doing"):
            score *= 1.1

        priority = block.get("Priority", "")
        if priority in ("P0", "P1"):
            score *= 1.1

        # --- A-MEM importance boost ---
        if meta_mgr:
            try:
                importance = meta_mgr.get_importance_boost(block.get("_id", ""))
                score *= importance
            except Exception as e:
                _log.warning("amem_importance_boost_failed", block_id=block.get("_id", ""), error=str(e))

        # Build rich result payload with speaker + display text
        raw_excerpt = get_excerpt(block)
        tags_str = block.get("Tags", "")
        speaker = _parse_speaker_from_tags(tags_str)

        result = {
            "_id": block.get("_id", "?"),
            "type": get_block_type(block.get("_id", "")),
            "score": round(score, 4),
            "excerpt": raw_excerpt,
            "speaker": speaker,
            "tags": tags_str,
            "file": block.get("_source_file", "?"),
            "line": block.get("_line", 0),
            "status": status,
        }
        # Pass through DiaID for benchmark evidence matching
        if block.get("DiaID"):
            result["DiaID"] = block["DiaID"]
        results.append(result)

    # Graph-based neighbor boosting: 2-hop traversal for multi-hop recall
    if graph_boost and results:
        xref_graph = build_xref_graph(all_blocks)
        score_by_id = {r["_id"]: r["score"] for r in results}
        block_by_id = {b.get("_id"): b for b in all_blocks if b.get("_id")}

        neighbor_scores = {}

        # Multi-hop traversal with progressive decay.
        # For multi-hop queries, extend to 3-hop with stronger propagation.
        hop_decays = [GRAPH_BOOST_FACTOR, GRAPH_BOOST_FACTOR * 0.5]
        if query_type == "multi-hop":
            hop_decays.append(GRAPH_BOOST_FACTOR * 0.25)  # 3rd hop at 0.1

        for hop, decay in enumerate(hop_decays):
            # On first hop, seed from BM25 results; on later hops, seed from
            # newly discovered neighbors
            seeds = results if hop == 0 else [
                {"_id": nid, "score": ns}
                for nid, ns in neighbor_scores.items()
                if nid not in score_by_id
            ]
            hop_added = 0
            for r in seeds:
                rid = r["_id"]
                neighbors = xref_graph.get(rid, set())
                for neighbor_id in neighbors:
                    if hop_added >= MAX_GRAPH_NEIGHBORS_PER_HOP:
                        break
                    boost = r["score"] * decay
                    if neighbor_id not in score_by_id:
                        neighbor_scores[neighbor_id] = (
                            neighbor_scores.get(neighbor_id, 0) + boost
                        )
                        hop_added += 1
                    else:
                        neighbor_scores[neighbor_id] = (
                            neighbor_scores.get(neighbor_id, 0) + boost * 0.5
                        )

        # Apply boosts to existing results
        for r in results:
            if r["_id"] in neighbor_scores:
                r["score"] = round(r["score"] + neighbor_scores[r["_id"]], 4)
                r["via_graph"] = True

        # Add new neighbors discovered via graph
        for nid, nscore in neighbor_scores.items():
            if nid not in score_by_id and nid in block_by_id:
                nb = block_by_id[nid]
                nb_tags = nb.get("Tags", "")
                results.append({
                    "_id": nid,
                    "type": get_block_type(nid),
                    "score": round(nscore, 4),
                    "excerpt": get_excerpt(nb),
                    "speaker": _parse_speaker_from_tags(nb_tags),
                    "tags": nb_tags,
                    "file": nb.get("_source_file", "?"),
                    "line": nb.get("_line", 0),
                    "status": nb.get("Status", ""),
                    "via_graph": True,
                })

    # --- RM3 Dynamic Query Expansion ---
    # When enabled via config, use RM3 (Relevance Model 3) instead of the
    # simpler PRF heuristic below.  RM3 estimates a relevance language model
    # from top-K feedback docs, then interpolates expansion terms with the
    # original query.  Skipped for adversarial queries (static expansions only).
    is_adversarial = query_type == "adversarial"
    _rm3_used = False

    config_path = os.path.join(workspace, "mind-mem.json")
    rm3_config = {}
    ce_config = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path) as _f:
                _cfg = json.load(_f)
            rm3_config = _cfg.get("recall", {}).get("rm3", {})
            ce_config = _cfg.get("recall", {}).get("cross_encoder", {})
        except (OSError, json.JSONDecodeError, KeyError) as e:
            _log.warning("rm3_config_load_failed", path=config_path, error=str(e))

    if rm3_config.get("enabled", False) and not is_adversarial and results:
        results.sort(key=lambda r: (r["score"], r.get("_id", "")), reverse=True)

        # Build collection frequency from all flat doc tokens
        collection_freq = Counter()
        total_collection_tokens = 0
        for flat_toks in doc_flat_tokens:
            for t in flat_toks:
                collection_freq[t] += 1
                total_collection_tokens += 1

        # Prepare top docs as (doc_tokens, score) tuples
        result_id_to_idx = {}
        for i, block in enumerate(all_blocks):
            result_id_to_idx[block.get("_id", "")] = i

        top_doc_tokens = []
        for r in results[:rm3_config.get("fb_docs", 5)]:
            idx = result_id_to_idx.get(r["_id"])
            if idx is not None:
                top_doc_tokens.append((doc_flat_tokens[idx], r["score"]))

        expanded_weights = rm3_expand(
            query_tokens, top_doc_tokens,
            collection_freq, total_collection_tokens,
            alpha=rm3_config.get("alpha", 0.6),
            fb_terms=rm3_config.get("fb_terms", 10),
            fb_docs=rm3_config.get("fb_docs", 5),
            min_idf=rm3_config.get("min_idf", 1.0),
            doc_freq={t: c for t, c in df.items()},
            N=N,
        )

        # Re-score all blocks using RM3-expanded weighted query
        expansion_terms_rm3 = [t for t in expanded_weights if t not in set(query_tokens)]
        if expansion_terms_rm3:
            _rm3_used = True
            rm3_weight = 0.4
            # Build O(1) lookup index to avoid O(N^2) linear scan
            result_by_id = {r["_id"]: r for r in results}
            for i, block in enumerate(all_blocks):
                ft = doc_field_tokens[i]
                flat = doc_flat_tokens[i]
                if not flat:
                    continue

                weighted_tf_rm3 = Counter()
                wdl = 0.0
                for field, tokens in ft.items():
                    w = FIELD_WEIGHTS.get(field, 1.0)
                    wdl += len(tokens) * w
                    for t in tokens:
                        weighted_tf_rm3[t] += w

                rm3_score = 0.0
                for et in expansion_terms_rm3:
                    if et in weighted_tf_rm3:
                        wtf = weighted_tf_rm3[et]
                        idf = math.log((N - df.get(et, 0) + 0.5) / (df.get(et, 0) + 0.5) + 1)
                        numerator = wtf * (BM25_K1 + 1)
                        denominator = wtf + BM25_K1 * (1 - BM25_B + BM25_B * wdl / avg_wdl)
                        rm3_score += idf * numerator / denominator * expanded_weights[et]

                if rm3_score > 0:
                    bid = block.get("_id", "?")
                    existing = result_by_id.get(bid)
                    if existing:
                        existing["score"] = round(existing["score"] + rm3_score * rm3_weight, 4)
                    else:
                        result = {
                            "_id": bid,
                            "type": get_block_type(bid),
                            "score": round(rm3_score * rm3_weight, 4),
                            "excerpt": get_excerpt(block),
                            "file": block.get("_source_file", "?"),
                            "line": block.get("_line", 0),
                            "status": block.get("Status", ""),
                        }
                        if block.get("DiaID"):
                            result["DiaID"] = block["DiaID"]
                        results.append(result)
                        result_by_id[bid] = result

            _log.info("rm3_expansion", expansion_terms=expansion_terms_rm3[:5],
                      alpha=rm3_config.get("alpha", 0.6))

    # --- Pseudo-Relevance Feedback (PRF) ---
    # For single-hop, open-domain, and multi-hop queries, bridge the lexical
    # gap by extracting expansion terms from top-5 initial results and
    # re-scoring.  Multi-hop benefits from PRF because scattered facts often
    # use different vocabulary than the query.
    # Skipped when RM3 was used (they serve the same purpose).
    if not _rm3_used and query_type in ("single-hop", "open-domain", "multi-hop") and results:
        results.sort(key=lambda r: (r["score"], r.get("_id", "")), reverse=True)
        prf_top = results[:5]

        # Extract expansion terms: high-TF tokens from top-5 statements,
        # excluding query tokens and very common terms (low IDF).
        prf_terms = Counter()
        for r in prf_top:
            # Tokenize the excerpt (which is the Statement/Description)
            prf_tokens = tokenize(r.get("excerpt", ""))
            for t in prf_tokens:
                if t not in query_tokens and len(t) > 2:
                    prf_terms[t] += 1

        # Keep terms that appear in 2+ of top-5 docs (co-occurring = relevant)
        # and have moderate IDF (not too common, not too rare)
        expansion_terms = []
        for term, count in prf_terms.most_common(15):
            if count >= 2 and df.get(term, 0) < N * 0.3:
                expansion_terms.append(term)
            if len(expansion_terms) >= 8:
                break

        if expansion_terms:
            # Re-score all blocks with expanded query (original + PRF terms).
            # Multi-hop uses lower weight to avoid drifting away from the query.
            prf_weight = 0.25 if query_type == "multi-hop" else 0.4
            for i, block in enumerate(all_blocks):
                ft = doc_field_tokens[i]
                flat = doc_flat_tokens[i]
                if not flat:
                    continue

                # Compute weighted TF
                weighted_tf_prf = Counter()
                wdl = 0.0
                for field, tokens in ft.items():
                    w = FIELD_WEIGHTS.get(field, 1.0)
                    wdl += len(tokens) * w
                    for t in tokens:
                        weighted_tf_prf[t] += w

                # Score only expansion terms
                prf_score = 0.0
                for et in expansion_terms:
                    if et in weighted_tf_prf:
                        wtf = weighted_tf_prf[et]
                        idf = math.log((N - df.get(et, 0) + 0.5) / (df.get(et, 0) + 0.5) + 1)
                        numerator = wtf * (BM25_K1 + 1)
                        denominator = wtf + BM25_K1 * (1 - BM25_B + BM25_B * wdl / avg_wdl)
                        prf_score += idf * numerator / denominator

                if prf_score > 0:
                    bid = block.get("_id", "?")
                    # Find existing result and boost, or add new result
                    found = False
                    for r in results:
                        if r["_id"] == bid:
                            r["score"] = round(r["score"] + prf_score * prf_weight, 4)
                            found = True
                            break
                    if not found:
                        result = {
                            "_id": bid,
                            "type": get_block_type(bid),
                            "score": round(prf_score * prf_weight, 4),
                            "excerpt": get_excerpt(block),
                            "file": block.get("_source_file", "?"),
                            "line": block.get("_line", 0),
                            "status": block.get("Status", ""),
                        }
                        if block.get("DiaID"):
                            result["DiaID"] = block["DiaID"]
                        results.append(result)

    # --- Chain-of-retrieval for multi-hop queries ---
    # Emulates iterative search deterministically:
    # 1. Take top-N from first pass
    # 2. Extract bridge terms (capitalized entities, rare shared tokens)
    # 3. Re-score all blocks using bridge terms
    # 4. Merge new hits into results
    if query_type == "multi-hop" and results:
        results.sort(key=lambda r: (r["score"], r.get("_id", "")), reverse=True)
        hop1_top = results[:10]

        # Extract bridge terms: capitalized entities from top-10 that aren't in query
        query_lower_set = set(re.findall(r"[a-z]+", query.lower()))
        bridge_terms = Counter()
        for r in hop1_top:
            excerpt = r.get("excerpt", "")
            # Capitalized entities
            for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", excerpt):
                term = m.group(1).lower()
                if term not in query_lower_set and term not in _STOPWORDS:
                    bridge_terms[term] += 1
            # Rare content tokens (appear in 2+ of top-10)
            for tok in tokenize(excerpt):
                if tok not in set(query_tokens) and len(tok) > 3:
                    bridge_terms[tok] += 0.5

        # Keep bridge terms that appear in 2+ top-10 results
        bridge_tokens = [t for t, c in bridge_terms.most_common(12) if c >= 2]

        if bridge_tokens:
            # Second retrieval pass using bridge terms
            existing_ids = {r["_id"] for r in results}
            for i, block in enumerate(all_blocks):
                bid = block.get("_id", "?")
                if bid in existing_ids:
                    continue  # already in results

                ft = doc_field_tokens[i]
                flat = doc_flat_tokens[i]
                if not flat:
                    continue

                weighted_tf_br = Counter()
                wdl = 0.0
                for field, tokens in ft.items():
                    w = FIELD_WEIGHTS.get(field, 1.0)
                    wdl += len(tokens) * w
                    for t in tokens:
                        weighted_tf_br[t] += w

                bridge_score = 0.0
                for bt in bridge_tokens:
                    if bt in weighted_tf_br:
                        wtf = weighted_tf_br[bt]
                        idf = math.log((N - df.get(bt, 0) + 0.5) / (df.get(bt, 0) + 0.5) + 1)
                        numerator = wtf * (BM25_K1 + 1)
                        denominator = wtf + BM25_K1 * (1 - BM25_B + BM25_B * wdl / avg_wdl)
                        bridge_score += idf * numerator / denominator

                if bridge_score > 0:
                    # Also check original query overlap — bridge-only hits with
                    # zero original query overlap are likely noise
                    orig_overlap = sum(1 for qt in query_tokens if qt in weighted_tf_br)
                    if orig_overlap > 0:
                        # Blend: 0.3 * bridge_score (second hop is supplementary)
                        tags_str = block.get("Tags", "")
                        result = {
                            "_id": bid,
                            "type": get_block_type(bid),
                            "score": round(bridge_score * 0.3, 4),
                            "excerpt": get_excerpt(block),
                            "speaker": _parse_speaker_from_tags(tags_str),
                            "tags": tags_str,
                            "file": block.get("_source_file", "?"),
                            "line": block.get("_line", 0),
                            "status": block.get("Status", ""),
                            "via_chain": True,
                        }
                        if block.get("DiaID"):
                            result["DiaID"] = block["DiaID"]
                        results.append(result)

            _log.info("chain_of_retrieval", bridge_terms=bridge_tokens[:5],
                      new_hits=sum(1 for r in results if r.get("via_chain")))

    # Sort by score descending
    results.sort(key=lambda r: (r["score"], r.get("_id", "")), reverse=True)

    # --- v7: Two-stage pipeline — wide BM25 retrieve -> dedup -> rerank -> top-k ---
    # Stage 1: Take wide candidate set (retrieve_wide_k)
    wide_k = max(retrieve_wide_k, limit)  # never retrieve fewer than final limit
    wide_candidates = results[:wide_k]

    # Deduplicate by (file, line) stable key — prevents near-duplicate slots
    seen_keys = set()
    deduped = []
    for r in wide_candidates:
        # Primary dedup: file+line
        stable_key = (r.get("file", ""), r.get("line", 0))
        if stable_key != ("", 0) and stable_key in seen_keys:
            continue
        if stable_key != ("", 0):
            seen_keys.add(stable_key)

        # Secondary dedup: DiaID — compound key (DiaID, id_prefix) so one FACT
        # and one DIA can coexist for the same dialog turn.
        dia = r.get("DiaID", "")
        if dia:
            rid = r.get("_id", "")
            prefix = "FACT" if rid.startswith("FACT-") else "DIA" if rid.startswith("DIA-") else rid[:4]
            dia_key = (dia, prefix)
            if dia_key in seen_keys:
                continue
            seen_keys.add(dia_key)

        deduped.append(r)

    # Stage 2: Deterministic rerank (v7) — cap candidates to prevent latency (#9)
    if rerank and len(deduped) > limit:
        rerank_cap = min(len(deduped), MAX_RERANK_CANDIDATES)
        deduped = rerank_hits(query, deduped[:rerank_cap], debug=rerank_debug)

    # Stage 2.5: Optional cross-encoder neural reranking — capped (#9)
    if ce_config.get("enabled", False):
        try:
            from cross_encoder_reranker import CrossEncoderReranker
            if CrossEncoderReranker.is_available():
                ce = CrossEncoderReranker()
                ce_cap = min(len(deduped), MAX_RERANK_CANDIDATES)
                ce_input = deduped[:ce_cap]
                for r in ce_input:
                    if "content" not in r:
                        r["content"] = r.get("excerpt", "")
                deduped = ce.rerank(
                    query, ce_input,
                    top_k=ce_config.get("top_k", limit),
                    blend_weight=ce_config.get("blend_weight", 0.6),
                )
                _log.info("cross_encoder_rerank", candidates=len(ce_input),
                          blend_weight=ce_config.get("blend_weight", 0.6))
        except ImportError:
            _log.debug("cross_encoder_import_failed",
                       hint="cross_encoder_reranker not installed")
        except Exception as e:
            _log.warning("cross_encoder_unavailable", error=str(e))

    top = deduped[:limit]

    # Stage 3: Context packing — augment top-K with adjacency/diversity/rescue
    top = context_pack(query, top, all_blocks, deduped, limit)

    # --- A-MEM: record access and evolve keywords for returned blocks ---
    if meta_mgr and top:
        try:
            returned_ids = [r["_id"] for r in top]
            meta_mgr.record_access(returned_ids, query=query)
            for r in top:
                meta_mgr.evolve_keywords(r["_id"], query_tokens, r.get("excerpt", ""))
        except Exception as e:
            _log.warning("amem_access_record_failed", error=str(e))

    # --- Optional LLM extraction enrichment (config-gated) ---
    if _HAS_LLM_EXTRACTOR and top:
        try:
            top = _llm_enrich_results(top, workspace=workspace)
        except Exception as e:
            _log.warning("llm_enrichment_failed", error=str(e))

    _log.info("query_complete", query=query, query_type=query_type,
              blocks_searched=N, wide_k=wide_k, reranked=rerank,
              results=len(top),
              top_score=top[0]["score"] if top else 0)
    metrics.inc("recall_queries")
    metrics.inc("recall_results", len(top))
    return top


def _load_backend(workspace: str) -> str:
    """Load recall backend from config. Falls back to BM25 scan.

    Supported backends:
        "scan" / "tfidf" — in-memory BM25 scan (default, O(corpus))
        "sqlite"          — SQLite FTS5 index (O(log N))
        "vector"          — vector embedding backend (requires recall_vector)
    """
    config_path = os.path.join(workspace, "mind-mem.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            recall_cfg = cfg.get("recall", {})
            unknown = set(recall_cfg.keys()) - _VALID_RECALL_KEYS
            if unknown:
                _log.warning("unknown_recall_config_keys", keys=sorted(unknown))
            backend = recall_cfg.get("backend", "scan")
            if backend == "sqlite":
                return "sqlite"
            if backend == "vector":
                try:
                    from recall_vector import VectorBackend
                    return VectorBackend(recall_cfg)
                except ImportError:
                    _log.warning("vector_backend_unavailable",
                                 hint="recall_vector not installed, falling back to BM25 scan")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            _log.warning("config_load_failed", path=config_path, error=str(e))
    return None  # use built-in BM25 scan


# ---------------------------------------------------------------------------
# Prefetch Context — anticipatory pre-assembly for proactive memory
# ---------------------------------------------------------------------------

def prefetch_context(
    workspace: str,
    recent_signals: list[str],
    limit: int = 5,
) -> list[dict]:
    """Given recent conversation signals (entity mentions, topic keywords),
    pre-fetch memory blocks likely to be needed next.

    Uses intent routing + category summaries to anticipate needs.
    Returns pre-ranked blocks ready for context injection.

    Args:
        workspace: Workspace root path.
        recent_signals: List of recent entity mentions, topic keywords, or
            short phrases from the conversation that hint at upcoming needs.
        limit: Maximum number of blocks to return.

    Returns:
        Deduplicated list of blocks ranked by relevance to the signals.
    """
    if not recent_signals:
        return []

    seen_ids: set[str] = set()
    results: list[dict] = []

    # 1. Direct recall for each signal
    for signal in recent_signals:
        signal = signal.strip()
        if not signal:
            continue
        try:
            hits = recall(workspace, signal, limit=limit, rerank=True)
        except Exception as e:
            _log.warning("prefetch_recall_failed", signal=signal, error=str(e))
            hits = []
        for block in hits:
            bid = block.get("_id", "")
            if bid and bid not in seen_ids:
                seen_ids.add(bid)
                results.append(block)

    # 2. Category-aware boost: if category distiller is available,
    #    pull in category context blocks that match the signals
    try:
        from category_distiller import CategoryDistiller
        distiller = CategoryDistiller()
        combined_query = " ".join(recent_signals)
        relevant_cats = distiller.get_categories_for_query(combined_query)
        if relevant_cats:
            # Recall from category keywords as supplemental signal
            cat_query = " ".join(relevant_cats[:3])
            try:
                cat_hits = recall(workspace, cat_query, limit=3, rerank=True)
                for block in cat_hits:
                    bid = block.get("_id", "")
                    if bid and bid not in seen_ids:
                        seen_ids.add(bid)
                        results.append(block)
            except Exception as e:
                _log.warning("prefetch_category_recall_failed", error=str(e))
    except ImportError:
        _log.debug("category_distiller_unavailable", hint="prefetch categories skipped")

    # 3. Trim to limit and return
    return results[:limit]


def main():
    parser = argparse.ArgumentParser(description="mind-mem Recall Engine (BM25 + Graph)")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--workspace", "-w", default=".", help="Workspace path")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")
    parser.add_argument("--active-only", action="store_true", help="Only search active blocks")
    parser.add_argument("--graph", action="store_true", help="Enable graph-based neighbor boosting")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--retrieve-wide-k", type=int, default=200,
                        help="Candidates to retrieve before reranking (default 200)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable v7 deterministic reranking (use pure BM25)")
    parser.add_argument("--rerank-debug", action="store_true",
                        help="Show reranker feature breakdowns in JSON output")
    parser.add_argument("--backend", choices=["scan", "sqlite", "auto"],
                        default="auto",
                        help="Recall backend: scan (O(corpus)), sqlite (O(log N)), auto (config)")
    args = parser.parse_args()

    # Resolve backend: CLI flag > config > default scan
    backend = args.backend
    if backend == "auto":
        cfg_backend = _load_backend(args.workspace)
        if cfg_backend == "sqlite":
            backend = "sqlite"
        elif cfg_backend is not None:
            # Vector or other custom backend
            try:
                results = cfg_backend.search(
                    args.workspace, args.query, args.limit, args.active_only
                )
            except (OSError, ValueError, TypeError) as e:
                print(f"recall: backend error ({e}), falling back to scan", file=sys.stderr)
                backend = "scan"
            else:
                backend = None  # already have results
        else:
            backend = "scan"

    if backend == "sqlite":
        from sqlite_index import query_index
        results = query_index(
            args.workspace, args.query, limit=args.limit,
            active_only=args.active_only, graph_boost=args.graph,
            retrieve_wide_k=args.retrieve_wide_k,
            rerank=not args.no_rerank, rerank_debug=args.rerank_debug,
        )
    elif backend == "scan":
        results = recall(args.workspace, args.query, args.limit, args.active_only,
                         args.graph, retrieve_wide_k=args.retrieve_wide_k,
                         rerank=not args.no_rerank, rerank_debug=args.rerank_debug)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print("No results found.")
        else:
            for r in results:
                graph_tag = " [graph]" if r.get("via_graph") else ""
                rerank_tag = ""
                if args.rerank_debug and "_rerank_features" in r:
                    feats = r["_rerank_features"]
                    rerank_tag = (
                        f" [rerank: ent={feats['entity_overlap']:.2f}"
                        f" time={feats['time_overlap']:.2f}"
                        f" bi={feats['bigram_bonus']:.2f}"
                        f" rec={feats['recency_bonus']:.2f}"
                        f" spk={feats['speaker_bonus']:.2f}]"
                    )
                print(f"[{r['score']:.3f}] {r['_id']} ({r['type']}{graph_tag}) — {r['excerpt'][:80]}{rerank_tag}")
                print(f"        {r['file']}:{r['line']}")


if __name__ == "__main__":
    main()
