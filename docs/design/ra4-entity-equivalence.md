# RA.4 entity equivalence (candidate)

This candidate implements governed entity merging as a reversible `SAME_AS`
edge. It never deletes an entity, rewrites a source edge, moves an alias, or
combines observations. A proposal names two existing registry IDs, names the
winner explicitly, and carries a non-empty rationale. Staging changes only
the proposal table; approval is an admin operation through the existing
`admit_proposal` governance scope.

The approved edge is the lineage record. `KnowledgeGraph.same_as_component()`
computes a deterministic, sorted union-find view by traversing approved
`SAME_AS` edges. Read callers that opt into this view (`edges_of(...,
include_equivalents=True)` or `graph_query(..., resolve_same_as=True)`) see
the preserved neighborhoods from both IDs while raw callers retain the
source graph. Reversal is a governed deletion of that exact edge, so the
original IDs, aliases, observations, endpoint values, and edge metadata remain
available after reversal.

Approval refuses missing IDs, self-links, an existing pairwise equivalence, and
any database collision. SQLite transaction rollback protects the edge and
proposal state together; a repeated approval is idempotent. `SAME_AS` cannot
be written through the generic edge doors, which keeps the proposal flow as
the only merge authority. There is no REST surface in this slice and no
automatic entity resolution.

This is an implemented candidate pending integration and full release gates.
