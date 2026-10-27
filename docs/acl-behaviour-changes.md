# ACL behaviour changes for namespace-bound callers

Applies only to a BOUND caller (an authenticated namespace principal). An unbound caller (stdio / local operator) is unchanged everywhere.
"Whole view" = the caller's namespace ACL lets it READ every namespace source: the workspace root corpus, `shared`, every `agents/*` directory and every declared custom namespace. It is a statement about the ACL, not about which blocks are admitted today: a namespace that holds only quarantined or pending blocks still has to be readable. "Refused" = `{"error": "namespace access denied"}`.

## Refused unless the caller has a whole view
- memory_health: root + FTS + compaction counts for any bound caller -> refused.
- index_stats: workspace block count / index status -> refused.
- graph_stats, graph_query, entity_observations: knowledge-graph reads (not namespace-attributed) -> refused.
- lint: findings with file paths and block ids -> refused.
- dream_cycle: entity slugs, cited ids, source files, quoted fact text -> refused.
- retrieval_diagnostics: every caller's query text and stage counts -> refused.
- calibration_stats: per-block feedback scores -> refused.
- outcome_stats (health report, or task_id listing): per-block outcome totals -> refused.
- verify_merkle (and memory_verify mode=merkle): proof + root + hash-confirmation oracle -> refused, for every id, present or not.
- category_summary `kind_summaries` section: per-kind digests -> omitted.
- traverse_graph `graph_summary` (workspace-wide edge / node / stale totals): returned -> omitted.

## Filtered to what the caller may read
- stale_blocks: every staleness flag -> only flags on servable blocks (`source_id` blanked when unreadable); `stale_count` is the filtered count.
- pipeline_status: dirty ids and `dirty_count` for the whole root corpus -> servable blocks only.
- reindex_dirty(dry_run=true): all dirty ids -> servable ids only (limit applied after filtering). dry_run=false is admin-only, unchanged.
- traverse_graph: causal edges / chains / dependents anywhere -> only edges and nodes on servable blocks; a denied root answers like a missing id.
- block_lineage: BFS over the whole lineage graph -> BFS over the visible subgraph only (never enters or steps through a hidden block; distances and reachability are derivable from readable blocks); a denied root answers like a root with no edges.
- list_evidence: all governance evidence rows -> rows whose `target_block_id` is servable.
- memory_evolution(get): importance, co-retrieval partners, provenance for any block -> defaults for an unreadable block; co-occurring list filtered.
- outcome_stats(block_id=...): outcomes for any block -> empty for an unreadable block.
- category_summary: category sections for any block -> sections for servable blocks only; the "N blocks in this category" line is stripped; `withheld_count` counts only withheld blocks in sources the caller can read (see below).
- list_edge_proposals / list_entity_merge_proposals: see "Proposal visibility" below (supersedes the earlier cited-block rule).

## Root-file readers now ask the ACL
- compiled_truth_load / compiled_truth_contradictions / compiled_truth(action=load|contradictions): read entities/compiled/<id>.md for anyone -> "No compiled truth page found" when the caller cannot read that file.
- check_guardrails: guardrails/GUARDRAILS.md (and configured sources) for anyone -> only sources the caller may read.
- recall_with_guardrails: root guardrails force-surfaced to anyone -> force-surfaced only when the caller can read ALL configured guardrail sources; otherwise plain ranked recall.
- resume_brief, check_dead_ends: frames/FRAMES.md and frames/DEAD-ENDS.md for anyone -> only readable sources.
- similar_trajectories: trajectories/ sidecar for anyone -> empty unless trajectories/ is readable.
- validate_block: near-duplicate rule checks root intelligence/SIGNALS.md for anyone -> rule is skipped (listed in `skipped_rules`) unless that file is readable.

## Recall correctness (bound callers)
- recall / hybrid / bm25 on a default install: the first recall created recall.db (calibration/retrieval_log tables only); every later recall treated the file as an index, found no `blocks_fts`, and fell back to a scan that dropped the principal and walked the root only -> a shared-only caller got `[]` from the second recall on. Now: the FTS SCHEMA (not the file) selects the indexed leg, and query_index's fallback scan carries the caller's principal.
- Known limitation, unchanged: once a real FTS index exists (`reindex`), it covers the root corpus only, so a caller without root access sees no shared/agents content through the indexed leg.

## New data
- propose_edge / propose_entity_merge now stamp `metadata.proposer` (the authenticated principal, bound callers only), `metadata.proposer_view` (the namespace roots the proposer could read when staging; an unbound operator carries every root) and `metadata.derives_from` (the cited block for an edge; the source blocks of both entities for a merge). Existing proposals carry none of these.

# Residual read leaks closed (branch fix/acl-residual)

Everything below applies to a BOUND caller; an unbound caller (stdio / local operator) is unchanged unless stated.

## Whole-view definition (behaviour change for aggregates)
- `ReadScope.whole` is computed from the ACL, not from admitted blocks. A caller is whole only if it can read the root corpus, `shared`, every `agents/*` directory, every declared custom namespace and every corpus file registered under them. Previously a shared-only caller was whole whenever the namespaces it could not read held only quarantined / pending blocks, and was then served the workspace-wide aggregates (retrieval_diagnostics, calibration_stats, memory_health, index_stats, graph_*, lint, verify_merkle, outcome_stats, dream_cycle, entity_observations). Those are now refused for it.

## Model tools (audit_model_tool, verify_model_tool)
- A bound caller may only point these at (a) a directory listed in the new `model_dirs` config key (operator-designated, absolute paths) or (b) a path inside the workspace its ACL lets it read (for a directory: every file under it). Every other path -- an unreadable namespace, a path outside the workspace, a path that does not exist -- gets the same body: `{"ok": false, "error": "model path not found: <path as supplied>"}` (audit) / `missing_file ... not found in <path as supplied>` (verify). Filenames, per-file sha256, `file_count` and `total_bytes` of an unreadable namespace are no longer returned and missing-vs-present no longer differs.
- `verify_model_tool(pubkey_path=...)` applies the same rule to the key file: an unreadable one reads as absent.
- Replies to a bound caller echo the path as supplied, never the resolved absolute host path (`model_path`, error text).
- `sign_model_tool` is admin scope and unchanged.

## vault_scan
- A vault root that lies inside the workspace is corpus: a bound caller can scan it only if its ACL lets it read every file under it; otherwise the reply is `vault root not found: <resolved path>`, the answer for a vault that does not exist. A vault outside the workspace is governed by the existing operator allowlist (`MIND_MEM_VAULT_ALLOWLIST`) and is unchanged. `vault_sync` is admin scope and unchanged.

## mind_mem_verify
- Refused unless the caller has a whole view (it reports workspace-wide ledger lengths, per-check counts and chain heads).
- At non-admin scope the absolute workspace path is replaced by `<workspace>` in the whole reply (the `workspace` field and any quoted file paths).

## propose_entity_merge
- A restricted (not whole-view) caller may name an entity only if it exists AND derives from blocks the caller can read (an entity with no recorded source block counts as unreadable). Any other pair -- existing but unreadable, one endpoint unreadable, or absent -- gets the exact error a missing entity gets: `winner_id and loser_id must name existing entities`. The argument checks that do not depend on the graph (distinct ids, rationale length) are unchanged and run first.

## get_block (and every id-keyed tool)
- A source the caller's ACL does not grant answers exactly like an absent id: `{"found": false, "error": "Block <id> not found in any corpus file.", ...}` (or "the block store"). `{"error": "namespace access denied"}` is no longer returned by get_block. This covers an explicit `namespace` selector the caller cannot read, `agents/<other>`, and a selector for a custom namespace that is not declared (it no longer answers `invalid namespace selector` for a bound caller who could not read it anyway; an unbound caller keeps the specific selector errors).
- Timing: a denied id is also indistinguishable from a missing id by latency. get_block now reads every corpus source for every id (found, denied or missing) instead of returning as soon as the id is found, and block_lineage walks an edgeless surrogate for a root the caller cannot read. Cost: a successful get_block scans the whole corpus rather than the prefix file only.
- Unchanged by design: a block in a namespace the caller CAN read that has not passed admission still answers `{"found": false, "withheld": true}`.
- The remaining uses of `namespace access denied` are aggregates and whole-corpus tools (refused for a restricted view) and carry no id.

## list_cores / core(action=list)
- A restricted caller sees a loaded core (namespace, version, block/edge counts, content hash) only if every block id in it, and the source block of every edge in it, is one the caller may read. A core made of unreadable blocks is not listed, and neither is an empty core (its name is an arbitrary label and proves nothing about what it was built from). An unbound caller and a whole-view caller see all cores.

## category_summary `withheld_count`
- Counts only withheld (quarantined / pending / unresolved) blocks whose source the caller can read. A caller who cannot read the root corpus gets 0 for the root corpus's withheld blocks. When no section of a category is readable the reply is the same `no_categories` answer an absent category gets (the bare category title is no longer returned).

## Stdlib HTTP `GET /status`, gRPC Health, REST health
- `GET /status` with a namespace ACL configured answers liveness only: `ok`, workspace basename, `server_time`. `memory_count` and `last_scan_timestamp` are omitted (they are workspace-wide). Without an ACL the response is unchanged.
- gRPC `Health` returns the workspace basename, never the absolute path (ACL or not).
- REST `GET /v1/health` for an authenticated caller returns the workspace basename instead of the absolute path when a namespace ACL is configured; unchanged otherwise.

## Proposal visibility (list_edge_proposals, list_entity_merge_proposals)
- `metadata.proposer`, `metadata.proposer_view` and `metadata.derives_from` are returned only to the proposer itself and to whole-view callers; anyone else sees the proposal with those keys removed.
- A proposal staged by someone else is shown to a restricted caller only if (stamped proposals) the caller can read every namespace root the proposer could read when it staged the proposal AND every block the proposal cites / derives from is servable to the caller; (legacy proposals with no stamp) it cites blocks and every one is servable. A merge proposal with no stamp and a different proposer is not shown to a restricted caller. Free text (subject / object / rationale) can come from anything the proposer could read, which is why the proposer's view, not just the cited block, decides.
- An unbound (operator) caller can read everything, so proposals it stages are stamped with every root and are shown to restricted callers only if they can read every root (that is, whole views). Previously an operator proposal citing a readable block was shown to every reader of that block.

## REST under `--allow-unauthenticated-localhost` with an ACL
- Decision: REFUSE, matching the stdlib HTTP transport and gRPC. With the loopback opt-in and no authentication, the four REST read endpoints (`/v1/recall`, `/v1/block/{id}`, `/v1/scan`, `/v1/contradictions`) answer `403 {"detail": "namespace ACL is configured; this transport has no per-caller identity and is refused"}` while `mind-mem-acl.json` declares agents or `default_policy: deny`. The flag is not an operator override: a flag that exists to skip authentication cannot also be the thing that decides which namespaces a caller sees, and every other transport already refuses. To read under an ACL, configure a token or API key (it maps to a principal). the REST launcher logs `anonymous_local_reads_refused_under_namespace_acl` at startup in that configuration. Without an ACL file the opt-in is unchanged.

## Host paths in replies (model tools, lint, vault_scan)
- No user-scope reply carries an absolute host path. A single helper, `mind_mem.path_redact`, does the rewriting: a path the caller supplied comes back exactly as supplied (`~` unexpanded, trailing slash and `//` kept), the workspace is shown by directory name or workspace-relative, and `$HOME` becomes `~`.
- `verify_model_tool`, `audit_model_tool` and `sign_model_tool` echo `pubkey_path`, `key_file` and the model path in the caller's own spelling in every error and reply (previously the `~`-expanded absolute path, or the resolved checkpoint root, was shown). Sidecar paths in a `sign_model_tool` reply are the supplied directory plus the sidecar name. For `audit_model_tool`, `model_path` is the supplied path for every caller (an unbound caller used to get the resolved path).
- `lint` returns `workspace` as the workspace directory name, not the absolute path, and the "v4.lint is disabled" answer (which is given before any ACL check) no longer names the workspace path. `lint_autofix` likewise.
- `vault_scan` answers "vault root not found: <root as supplied>" instead of the resolved path.
- `list_evidence(block_id=...)`: an unreadable block id is looked up as an id no record carries, so a denied id costs the same work as a missing one (previously the rows were matched and then filtered).
- Regression: `tests/test_read_surface_host_paths.py` calls every tool in `USER_TOOLS` (read from the registry) and every `mind-mem://` resource, as an unbound and as a shared-only caller, and fails on any absolute workspace path or `$HOME` in a response.

## New configuration
- `model_dirs` (list of absolute paths) in `mind-mem.json`: model checkpoint directories that bound callers may audit / verify.
