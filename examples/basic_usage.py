#!/usr/bin/env python3
"""Basic mind-mem usage example.

Demonstrates workspace initialization, block creation, and recall.
"""
import os
import tempfile

from scripts.init_workspace import init as init_workspace
from scripts.block_parser import parse_file
from scripts._recall_core import recall


def main():
    # Create a temporary workspace
    ws = tempfile.mkdtemp(prefix="mind-mem-example-")
    init_workspace(ws)
    print(f"Workspace initialized at: {ws}")

    # Create some decision blocks
    decisions_file = os.path.join(ws, "decisions", "example.md")
    with open(decisions_file, "w") as f:
        f.write("""[DEC-001]
Type: Decision
Statement: Use PostgreSQL for the primary database
Rationale: Better JSON support and full-text search capabilities
Date: 2026-02-25

[DEC-002]
Type: Decision
Statement: Deploy on Kubernetes with ArgoCD
Rationale: GitOps workflow for reproducible deployments
Date: 2026-02-25

[DEC-003]
Type: Decision
Statement: Use Redis for session caching
Rationale: Sub-millisecond latency for session lookups
Date: 2026-02-25
""")

    # Parse blocks
    blocks = parse_file(decisions_file)
    print(f"Parsed {len(blocks)} blocks")

    # Search with recall
    results = recall("database", blocks, limit=5)
    print(f"\nRecall results for 'database':")
    for r in results:
        block_id = r.get("block_id", r.get("id", "unknown"))
        score = r.get("score", 0)
        text = r.get("text", r.get("content", ""))[:80]
        print(f"  [{block_id}] score={score:.2f} — {text}")


if __name__ == "__main__":
    main()
