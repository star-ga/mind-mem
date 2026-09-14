# RA.4 entity equivalence (5.0.3 candidate)

Governed entity merging uses a reversible `SAME_AS` edge. It never deletes an
entity, rewrites a source edge, moves an alias, or combines observations. A
proposal names two existing registry IDs, identifies the winner explicitly,
and carries at least eight non-whitespace rationale characters. Staging changes
only the proposal table; approval is an admin operation through the existing
`admit_proposal` governance scope.

The approved edge and proposal lineage record the merge.
`KnowledgeGraph.same_as_component()` computes a deterministic, sorted union
view by traversing approved `SAME_AS` edges. Read callers opt in with
`edges_of(..., include_equivalents=True)` or
`graph_query(..., resolve_same_as=True)`. Graph traversal expands equivalent
IDs at every visited depth and retains source-edge provenance and the requested
depth bound. Raw callers retain the source graph. Governed reversal removes
only the approved equivalence edge, preserving the original IDs, aliases,
observations and source-edge metadata.

Approval refuses missing IDs, self-links, already connected components,
cycles, database collisions and proposal IDs that do not match the canonical
endpoint identity. Both staging and persisted-proposal validation enforce the
rationale minimum. The proposal identity binds the endpoint pair; it does not
make historical rationale text immutable.

SQLite transactions protect graph, proposal and lineage changes together.
Repeated approval is idempotent only while its applied lineage and exact
`SAME_AS` edge remain consistent with the proposal. Missing or modified state
is refused, without silently repairing it. Reversal rereads and validates the
current proposal under the write transaction before checking lineage and
removing the edge. Generic edge-write tools cannot create `SAME_AS`, so they
cannot bypass this proposal flow.

The MCP surface consists of `propose_entity_merge`,
`list_entity_merge_proposals`, `approve_entity_merge` and
`reverse_entity_merge`. Approval and reversal require admin scope. There is
no entity-merge REST surface or automatic entity resolution in this slice.
Readers that consume persisted graphs must understand the `SAME_AS` predicate.

The candidate is integrated and has focused independent acceptance evidence.
Full release verification and publication are still pending.
