#!/usr/bin/env python3
# Copyright 2026 STARGA, Inc.
"""Build a SYNTHETIC workspace and measure Class A integrity on it.

The committed Class A scorecard was produced against the operator's private
corpus, so it carried that workspace's path and its blocks' counts. Neither
belongs in a public artifact. This script generates a workspace from nothing,
using only synthetic content, and measures the same dimensions on it.

The public number this produces is not the private one, and must never be
presented as it. They answer different questions:

* the private corpus measures HISTORICAL coverage -- a corpus that predates
  the write-path guarantee, where most blocks have no evidence record;
* this workspace measures the GUARANTEE itself -- every block written through
  the governed path, so evidence coverage is what the admission path actually
  delivers on writes it mediated.

Both are honest; conflating them is not.

Usage:
    python3 benchmarks/make_public_integrity_workspace.py --out <dir> [--blocks N]
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

#: Deterministic synthetic content. No operator data, no real corpus text.
TOPICS = (
    ("deployment", "The release pipeline promotes a build only after the gate reports green."),
    ("retrieval", "Lexical and dense legs are fused with reciprocal rank fusion."),
    ("governance", "Every write passes an admission gate that mints an evidence record."),
    ("ontology", "A predicate constrains which entity types may sit on each side of an edge."),
    ("provenance", "An origin field is mutable; a hash-chained record is not."),
    ("determinism", "The same query over the same state returns byte-identical results."),
)


def build(workspace: str, n_blocks: int) -> int:
    """Write *n_blocks* synthetic blocks through the real governed path."""
    from mind_mem.enums import IngestTier
    from mind_mem.governance_gate import get_gate
    from mind_mem.pipeline_hash import stamp_transform_hash
    from mind_mem.storage import get_block_store

    os.makedirs(os.path.join(workspace, "memory"), exist_ok=True)
    store = get_block_store(workspace)
    written = 0
    for i in range(n_blocks):
        topic, sentence = TOPICS[i % len(TOPICS)]
        block = {
            # A registered corpus prefix is required for writes; "D"
            # (decisions) is a real one. Synthetic CONTENT, real plumbing.
            "_id": f"D-SYNTH{i:04d}",
            "type": "Decision",
            "Statement": f"{sentence} (synthetic record {i}, topic {topic})",
            "Category": topic,
            # The writer must stamp exactly what the tier mints. EXTERNAL_INGEST
            # mints "quarantined"; both "active" and an absent status raise
            # UngatedWriteError, because recall would otherwise serve content
            # the tier never admitted as servable.
            "Status": "quarantined",
        }
        stamped = stamp_transform_hash(workspace, block)
        with get_gate(workspace).admit_block(
            # INGEST is the truthful evidence classification for this: the
            # gate refuses an unclassified action rather than defaulting
            # it to APPLY, which would put an unchosen claim in the chain.
            action="INGEST",
            block_id=str(stamped["_id"]),
            content=str(stamped["Statement"]),
            # EXTERNAL_INGEST is the honest tier for generated fixture
            # content: it is imported material, not a curated proposal.
            tier=IngestTier.EXTERNAL_INGEST,
            actor="benchmark:public-integrity-fixture",
            metadata={"topic": topic},
        ):
            store.write_block(stamped)
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="workspace directory to create")
    ap.add_argument("--blocks", type=int, default=120)
    a = ap.parse_args()

    n = build(a.out, a.blocks)
    print(f"synthetic workspace: {a.out}")
    print(f"blocks written through the governed path: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
