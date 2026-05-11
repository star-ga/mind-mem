"""Audit canonical-answer coverage on every eval probe.

For each probe in eval_harness.py, count how many assistant messages in the
corpus contain ALL required tokens. A probe with <3 canonical answers is
under-saturated — that is the gradient-signal gap behind the whack-a-mole
failure pattern across the last three retrain attempts.

Usage:
    python3 train/audit_canonical_coverage.py [--threshold 3]

Prints per-group breakdowns and exits 0 if all probes have >= threshold
matches, 1 otherwise. Always writes audit_canonical_coverage.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CORPUS = Path("/data/checkpoints/mm-workspace/train-output/corpus.jsonl")
REPORT = ROOT / "audit_canonical_coverage.json"


def _load_eval_harness():
    """Import eval_harness without running _load_model()."""
    spec = importlib.util.spec_from_file_location("eval_harness", ROOT / "eval_harness.py")
    mod = importlib.util.module_from_spec(spec)
    # eval_harness.py does sys.path.insert; tolerate missing torch by stub
    sys.modules["eval_harness"] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        # We only need the probe lists at module top; torch import would fail
        # on this side. Fall back to text extraction.
        return None
    return mod


def _extract_probes_from_source() -> list[tuple[str, str, list[str]]]:
    """Fallback: parse probe lists out of eval_harness.py source.

    Returns list of (group_name, prompt, required_tokens).
    """
    import ast

    text = (ROOT / "eval_harness.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    probes: list[tuple[str, str, list[str]]] = []
    group_names = {
        "TOOL_CALL_QUESTIONS",
        "BLOCK_SCHEMA_QUESTIONS",
        "WORKFLOW_QUESTIONS",
        "V39_NEW_TOOLS",
        "V39_TRANSFORMHASH_PROMPTS",
        "V39_TRANSPORT_PROMPTS",
        "V311_NEW_TOOLS",
        "V311_EXPLAIN_FIELD",
        "V312_QUALITY_GATE_STRICT_MODE",
        "V312_LINEAGE_STALENESS",
        "V4_SURFACES",
    }
    def _harvest(name: str, value_node):
        if not isinstance(value_node, ast.List):
            return
        for elt in value_node.elts:
            if isinstance(elt, ast.Tuple) and len(elt.elts) >= 2:
                prompt = ast.literal_eval(elt.elts[0])
                req = ast.literal_eval(elt.elts[1])
                if isinstance(req, str):
                    req = [req]
                # Transport prompts have a 3rd `must_not` element — ignored for audit
                probes.append((name, prompt, list(req)))

    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in group_names:
                _harvest(node.target.id, node.value)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in group_names:
                    _harvest(tgt.id, node.value)
    return probes


def _load_corpus_assistant_messages() -> list[str]:
    msgs: list[str] = []
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for m in rec.get("messages", []):
                if m.get("role") == "assistant":
                    msgs.append(m.get("content", ""))
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=3,
                    help="min canonical answers per probe (default 3)")
    ap.add_argument("--top-fail", type=int, default=999,
                    help="limit failures printed (default unlimited)")
    args = ap.parse_args()

    if not CORPUS.is_file():
        sys.exit(f"corpus missing: {CORPUS}")

    probes = _extract_probes_from_source()
    print(f"extracted {len(probes)} probes across groups", flush=True)

    assistant_msgs = _load_corpus_assistant_messages()
    print(f"corpus: {len(assistant_msgs):,} assistant messages", flush=True)

    by_group: dict[str, list[dict]] = {}
    weak: list[dict] = []
    for group, prompt, required in probes:
        count = sum(1 for m in assistant_msgs if all(tok in m for tok in required))
        rec = {"prompt": prompt, "required": required, "count": count}
        by_group.setdefault(group, []).append(rec)
        if count < args.threshold:
            weak.append({"group": group, **rec})

    weak.sort(key=lambda x: (x["count"], x["group"], x["prompt"]))

    print("=" * 72)
    print(f"PER-GROUP coverage (threshold = {args.threshold})")
    print("=" * 72)
    for g, recs in by_group.items():
        n_weak = sum(1 for r in recs if r["count"] < args.threshold)
        avg = sum(r["count"] for r in recs) / len(recs) if recs else 0
        min_c = min((r["count"] for r in recs), default=0)
        print(f"  {g:<36} probes={len(recs):3d}  weak={n_weak:2d}  avg={avg:.1f}  min={min_c}")

    print()
    print("=" * 72)
    print(f"WEAK probes (canonical_answers < {args.threshold}):  {len(weak)} total")
    print("=" * 72)
    for i, r in enumerate(weak[: args.top_fail]):
        toks = ", ".join(r["required"])
        print(f"  [{r['count']}x] [{r['group']}]  {r['prompt'][:70]}")
        print(f"           required: [{toks}]")

    REPORT.write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "total_probes": len(probes),
                "total_weak": len(weak),
                "by_group": by_group,
                "weak": weak,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport → {REPORT}")
    return 0 if not weak else 1


if __name__ == "__main__":
    sys.exit(main())
