"""Cross-check every 'file X ships Y' claim in the corpus against src/.

retry-2c → 2d → broken because the audit only counted token presence;
it did not verify the corpus tuples were semantically correct.
This auditor walks the corpus assistant messages, extracts every
`src/mind_mem/<file>.py` reference paired with a function/symbol/table
mention, and verifies the symbol actually lives in that file (via grep).

A finding == a corpus tuple that claims `foo.py` ships `bar` but `bar`
is actually defined in some other file. Such tuples poison training.

Usage:
    python3 train/audit_semantic_correctness.py [--corpus CORPUS]

Exits 0 if no pollutions found, 1 otherwise. Always writes the
findings JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

CORPUS_DEFAULT = "/data/checkpoints/mm-workspace/train-output/corpus.jsonl"
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "mind_mem"
REPORT = Path(__file__).parent / "audit_semantic_correctness.json"

# pattern: matches `src/mind_mem/foo.py` or `mind_mem/foo.py` or backtick-wrapped foo.py
_FILE_RE = re.compile(r"`?src/mind_mem/(?:v\d+/)?([a-z_]+\.py)`?", re.IGNORECASE)
_FILE_BARE_RE = re.compile(r"`([a-z_]+\.py)`", re.IGNORECASE)

# Symbol patterns we care about: `function_name(`, `function_name`, `TABLE_NAME`, `Class.method`
_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]+(?:\.[A-Za-z_][A-Za-z0-9_]+)?)\b`")


def _find_symbol_files(symbol: str) -> set[str]:
    """Return basenames of source files that define `symbol`."""
    # search src/ for `def <symbol>`, `class <symbol>`, `<symbol> = `, `'<symbol>'` in CREATE TABLE
    patterns = [
        rf"^def\s+{re.escape(symbol)}\b",
        rf"^class\s+{re.escape(symbol)}\b",
        rf"^{re.escape(symbol)}\s*[:=]",
        rf"^{re.escape(symbol)}\s*:\s*[A-Z]",
        rf"CREATE\s+TABLE.*\b{re.escape(symbol)}\b",
    ]
    files = set()
    for pat in patterns:
        try:
            out = subprocess.run(
                ["grep", "-rlE", pat, str(SRC)],
                capture_output=True, text=True, timeout=10,
            )
            for line in out.stdout.splitlines():
                p = Path(line)
                if "__pycache__" in p.parts:
                    continue
                if p.is_file():
                    files.add(p.name)
        except Exception:
            continue
    return files


# Symbols that are RE-EXPORTED across files (so a claim mentioning multiple files isn't a pollution)
_RE_EXPORTED: dict[str, set[str]] = {
    # propagate_lineage_staleness: defined in lineage_staleness.py, re-imported in lineage_staleness.py only
    # add as needed when whitelisted false positives surface
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS_DEFAULT)
    ap.add_argument("--max-print", type=int, default=20)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.is_file():
        sys.exit(f"corpus missing: {corpus}")

    findings: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Limit to a focused list of "high-value" symbols — every function/constant
    # the eval cares about plus a few infrastructure ones we know exist.
    target_symbols = [
        "propagate_lineage_staleness",
        "register_schema_validator",
        "register_kernel",
        "register_health_probe",
        "is_kernel_registered",
        "is_policy_registered",
        "validate_block",
        "block_lineage",
        "mind_recall",
        "compute_surprise",
        "plan_consolidation",
        "PQCodec",
        "CircuitBreaker",
        "BackpressureController",
        "health_check",
        "KIND_DECAY",
        "LINEAGE_DEPTH_CAP",
        "MAX_CARDINALITY",
        "EmbeddingFailureError",
        "StaleVersionError",
        "FallbackPolicy",
        "KernelKind",
        "MergeStrategy",
        "EvictionPlan",
        "block_staleness",
        "block_kind_tags",
        "block_edits",
        "block_tier_vclock",
        "tier_conflict_log",
        "block_recall_tier",
    ]

    # Pre-resolve each symbol's true home file
    canonical_homes: dict[str, set[str]] = {}
    for s in target_symbols:
        canonical_homes[s] = _find_symbol_files(s)

    with corpus.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_msg = next((x["content"] for x in rec["messages"] if x["role"] == "user"), "")
            asst_msg = next((x["content"] for x in rec["messages"] if x["role"] == "assistant"), "")
            if not asst_msg:
                continue

            # Only flag DIRECT attributions: "X in/defined in/from `path/file.py`"
            # or "`path/file.py` — specifically the X function".
            # Co-mention without an attribution verb is NOT pollution.
            for symbol, real_homes in canonical_homes.items():
                if not real_homes:
                    continue
                # patterns: "<symbol>... (in|defined in|from|lives in|module|file|ships) ... <file>.py"
                # or:       "<file>.py ... (specifically|implements|defines|contains|ships) ... <symbol>"
                # Limit to a 200-char window around the symbol mention.
                for m in re.finditer(rf"`?\b{re.escape(symbol)}\b`?", asst_msg):
                    start = max(0, m.start() - 220)
                    end = min(len(asst_msg), m.end() + 220)
                    window = asst_msg[start:end]
                    # heuristics: look for an attribution within the window
                    attrib_pat = (
                        rf"({re.escape(symbol)}[^\.]{{0,120}}?(?:in|defined in|from|lives in|module|file|ships)\s+`?(?:src/mind_mem/(?:v\d+/)?)?([a-z_]+\.py)`?"
                        rf"|`?(?:src/mind_mem/(?:v\d+/)?)?([a-z_]+\.py)`?[^\.]{{0,120}}?(?:specifically|implements|defines|contains|ships|owns)[^\.]{{0,120}}?{re.escape(symbol)})"
                    )
                    am = re.search(attrib_pat, window, re.IGNORECASE)
                    if not am:
                        continue
                    claimed_file = next((g for g in am.groups()[1:] if g), None)
                    if not claimed_file:
                        continue
                    if claimed_file in real_homes:
                        continue  # correct attribution
                    key = (symbol, claimed_file, user_msg)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    findings.append({
                        "line": line_no,
                        "symbol": symbol,
                        "claimed_file": claimed_file,
                        "actual_homes": sorted(real_homes),
                        "user": user_msg[:120],
                        "assistant_excerpt": asst_msg[:300],
                    })
                    break  # one finding per symbol per tuple

    print("=" * 72)
    print(f"SEMANTIC-CORRECTNESS AUDIT — {corpus}")
    print("=" * 72)
    print(f"Polluted tuples (symbol attributed to wrong file): {len(findings)}")
    for f in findings[: args.max_print]:
        print()
        print(f"  line {f['line']}  symbol=`{f['symbol']}`")
        print(f"    claims files: {f['claimed_files']}")
        print(f"    actual homes: {f['actual_homes']}")
        print(f"    Q: {f['user']}")
        print(f"    A: {f['assistant_excerpt'][:200]}")

    REPORT.write_text(json.dumps({"findings": findings}, indent=2), encoding="utf-8")
    print(f"\nreport → {REPORT}")
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
