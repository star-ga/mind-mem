"""``mind-mem-recompact`` — cluster, recompact, PROPOSE. Never apply.

ROADMAP: "a `mm recompact` / dream-cycle pass 6 that clusters via `find_similar`
and routes results through `propose_update` (engine + bench shipped; the CLI verb
and scheduler wiring are not yet built)."

The engine shipped and was unreachable: :func:`~mind_mem.recompaction.recompact_cluster`
already refuses to mutate its input and already documents its output as "a proposal
for the HITL gate", but no verb ever called it. A capability with no entry point is
the reachability defect this codebase has repeatedly paid for.

THE DEFAULT IS DRY-RUN, and that is load-bearing rather than polite. A
recompaction rewrites several blocks' text into one summary. A verb that did that
by default, on a real corpus, would be the silent-overwrite failure wearing a
helpful name -- so ``--propose`` is required to stage anything, and even then it
stages a PROPOSAL that a human approves separately.

WHAT THIS REFUSES, and why each refusal is here rather than downstream:

* a seed id that does not exist -> error, not an empty plan. "Nothing to
  recompact" and "you typed the id wrong" must not look the same.
* a cluster of one -> refused. Recompacting a single block is a rewrite with no
  consolidation to justify it, and the loss (original wording) is real.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

#: Minimum cluster size worth recompacting. Two blocks is the smallest case where
#: a summary can say something neither said alone.
MIN_CLUSTER = 2


def _load_blocks(workspace: str) -> list[dict[str, Any]]:
    """Every admitted block in the workspace, via the shared reader."""
    from .storage import iter_active_blocks

    return list(iter_active_blocks(workspace))


def build_plan(
    workspace: str,
    *,
    block_id: str,
    limit: int = 5,
) -> dict[str, Any]:
    """What a recompaction of *block_id*'s cluster WOULD consolidate.

    Returns plain data and writes nothing -- the corpus is untouched, which a test
    asserts by comparing the file before and after.

    Keys: ``seed``, ``cluster`` (ids), ``refused`` (reason or ``""``).
    """
    blocks = _load_blocks(workspace)
    by_id = {str(b.get("_id") or b.get("id") or ""): b for b in blocks}
    if block_id not in by_id:
        return {
            "seed": block_id,
            "cluster": [],
            "refused": (
                f"no block {block_id!r} in this workspace. Refusing rather than "
                f"reporting an empty cluster: a typo and 'nothing to consolidate' "
                f"must not look the same."
            ),
        }

    from .mcp.infra.workspace import use_workspace
    from .mcp.tools.recall import find_similar

    with use_workspace(workspace):
        raw = json.loads(find_similar(block_id, limit=limit))
    similar = raw.get("results") or raw.get("similar") or raw.get("included") or []
    ids = [block_id] + [
        str(r.get("_id") or r.get("block_id") or "")
        for r in similar
        if str(r.get("_id") or r.get("block_id") or "") not in ("", block_id)
    ]
    cluster = [i for i in ids if i in by_id]

    refused = ""
    if len(cluster) < MIN_CLUSTER:
        refused = (
            f"cluster of {len(cluster)}: recompacting a single block is a rewrite "
            f"with no consolidation to justify it, and the original wording is lost "
            f"for nothing."
        )
    return {"seed": block_id, "cluster": cluster, "refused": refused}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mind-mem-recompact",
        description=(
            "Cluster a block's neighbours and propose a consolidated summary. "
            "Dry-run by default; --propose stages a proposal for human approval "
            "and never applies it."
        ),
    )
    parser.add_argument("workspace")
    parser.add_argument("--block-id", required=True, help="seed block for the cluster")
    parser.add_argument("--limit", type=int, default=5, help="neighbours to consider")
    parser.add_argument(
        "--propose",
        action="store_true",
        help="stage a proposal (still requires separate approval to apply)",
    )
    args = parser.parse_args(argv)

    plan = build_plan(args.workspace, block_id=args.block_id, limit=args.limit)
    if plan["refused"]:
        print(f"REFUSED: {plan['refused']}", file=sys.stderr)
        return 2

    print(f"seed:    {plan['seed']}")
    print(f"cluster: {', '.join(plan['cluster'])}  ({len(plan['cluster'])} blocks)")
    if not args.propose:
        print(
            "DRY RUN — nothing staged. Re-run with --propose to stage a proposal "
            "for human approval. Dry-run is the default because a recompaction "
            "rewrites several blocks' text into one summary."
        )
        return 0

    # --propose deliberately stops short of calling the governed door from here.
    # Wiring that call is the remaining half of this roadmap item and belongs in
    # the same change as the scheduler hookup, so the review surface (what gets
    # proposed, in what shape, under whose actor id) is decided once.
    print(
        "PROPOSE requested. The governed-door call is not wired yet — see the "
        "roadmap item: the propose_update hookup lands with the scheduler so the "
        "proposal shape is decided once rather than twice.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
