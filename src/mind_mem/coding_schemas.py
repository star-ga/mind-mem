#!/usr/bin/env python3
"""mind-mem Coding-Native Memory Schemas.

Block types optimized for software development workflows:
- ADR: Architecture Decision Records
- CODE: Function/class/API metadata
- PERF: Performance baselines and regressions
- ALGO: Algorithm descriptions and complexity analysis
- BUG: Bug reports with reproduction/fix tracking

Auto-classification of coding content from unstructured text.

Usage:
    from .coding_schemas import classify_coding_block, ADR_TEMPLATE
    block_type = classify_coding_block(text)
    template = get_template("ADR")

Zero external deps — re, json (all stdlib).
"""

from __future__ import annotations

import re
from datetime import datetime

from .observability import get_logger

_log = get_logger("coding_schemas")

# Block type identifiers
TYPE_ADR = "ADR"
TYPE_CODE = "CODE"
TYPE_PERF = "PERF"
TYPE_ALGO = "ALGO"
TYPE_BUG = "BUG"

CODING_BLOCK_TYPES = frozenset({TYPE_ADR, TYPE_CODE, TYPE_PERF, TYPE_ALGO, TYPE_BUG})

# Classification patterns
_ADR_PATTERNS = [
    re.compile(r"\b(architecture|architectural)\s+(decision|choice)", re.I),
    re.compile(r"\b(trade-?off|alternative|consequence|rationale)\b", re.I),
    re.compile(r"\b(ADR|RFC)\s*[-#]?\d+", re.I),
    re.compile(r"\b(decided|adopted|superseded|deprecated)\s+(to|that)\b", re.I),
]

_CODE_PATTERNS = [
    re.compile(r"\b(function|method|class|interface|struct|enum|module|endpoint)\b", re.I),
    re.compile(r"\b(API|REST|gRPC|GraphQL)\s+(endpoint|route|method)", re.I),
    re.compile(r"\b(signature|return\s+type|parameter|argument)\b", re.I),
    re.compile(r"\b(import|require|include|use)\s+\w+", re.I),
]

_PERF_PATTERNS = [
    re.compile(r"\b(benchmark|latency|throughput|p\d{2}|percentile)\b", re.I),
    re.compile(r"\b(regression|baseline|profil|flame\s*graph)\b", re.I),
    re.compile(r"\b(\d+\s*(ms|µs|ns|s)|fps|qps|rps|ops/s)\b", re.I),
    re.compile(r"\b(memory|cpu|gpu)\s+(usage|consumption|allocation)\b", re.I),
]

_ALGO_PATTERNS = [
    re.compile(r"\b(algorithm|heuristic|approach|technique)\b", re.I),
    re.compile(r"\bO\([^)]+\)\b"),  # Big-O notation
    re.compile(r"\b(time|space)\s+complexity\b", re.I),
    re.compile(r"\b(sort|search|traverse|hash|encrypt|compress)\b", re.I),
]

_BUG_PATTERNS = [
    re.compile(r"\b(bug|defect|issue|crash|error|exception)\b", re.I),
    re.compile(r"\b(reproduce|reproduction|repro\s+steps)\b", re.I),
    re.compile(r"\b(root\s+cause|stack\s+trace|backtrace)\b", re.I),
    re.compile(r"\b(fix|patch|hotfix|workaround)\b", re.I),
]


def classify_coding_block(text: str) -> str | None:
    """Classify unstructured text into a coding block type.

    Returns the block type with the highest pattern match count,
    or None if no coding patterns are detected.

    Args:
        text: Unstructured text to classify.

    Returns:
        One of TYPE_ADR, TYPE_CODE, TYPE_PERF, TYPE_ALGO, TYPE_BUG, or None.
    """
    scores = {
        TYPE_ADR: sum(1 for p in _ADR_PATTERNS if p.search(text)),
        TYPE_CODE: sum(1 for p in _CODE_PATTERNS if p.search(text)),
        TYPE_PERF: sum(1 for p in _PERF_PATTERNS if p.search(text)),
        TYPE_ALGO: sum(1 for p in _ALGO_PATTERNS if p.search(text)),
        TYPE_BUG: sum(1 for p in _BUG_PATTERNS if p.search(text)),
    }

    best_type = max(scores, key=lambda k: scores[k])
    if scores[best_type] >= 2:
        return best_type
    return None


def get_template(block_type: str) -> dict:
    """Get a template dict for a coding block type.

    Args:
        block_type: One of CODING_BLOCK_TYPES.

    Returns:
        Template dict with field names and empty/default values.
    """
    templates: dict[str, dict[str, object]] = {
        TYPE_ADR: {
            "Type": "ADR",
            "Title": "",
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Status": "proposed",  # proposed | accepted | deprecated | superseded
            "Context": "",
            "Decision": "",
            "Consequences": "",
            "Alternatives": [],
            "Trade-offs": "",
            "SupersededBy": "",
        },
        TYPE_CODE: {
            "Type": "CODE",
            "Name": "",
            "Kind": "",  # function | class | interface | endpoint | module
            "File": "",
            "Signature": "",
            "Description": "",
            "Dependencies": [],
            "Performance": "",
            "Constraints": "",
        },
        TYPE_PERF: {
            "Type": "PERF",
            "Name": "",
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Metric": "",  # latency | throughput | memory | cpu
            "Baseline": "",
            "Current": "",
            "Unit": "",  # ms | µs | MB | %
            "Environment": "",
            "Regression": False,
            "Notes": "",
        },
        TYPE_ALGO: {
            "Type": "ALGO",
            "Name": "",
            "Description": "",
            "TimeComplexity": "",
            "SpaceComplexity": "",
            "BestCase": "",
            "WorstCase": "",
            "WhenToUse": "",
            "WhenNotToUse": "",
            "Implementation": "",
            "References": [],
        },
        TYPE_BUG: {
            "Type": "BUG",
            "Title": "",
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Status": "open",  # open | investigating | fixed | wontfix
            "Severity": "",  # critical | high | medium | low
            "ReproSteps": [],
            "Expected": "",
            "Actual": "",
            "RootCause": "",
            "Fix": "",
            "RegressionTest": "",
            "AffectedFiles": [],
        },
    }
    if block_type not in templates:
        raise ValueError(f"Unknown block type '{block_type}'. Must be one of: {sorted(CODING_BLOCK_TYPES)}")
    return {k: v for k, v in templates[block_type].items()}


def extract_code_metadata(text: str) -> dict:
    """Extract structured metadata from coding-related text.

    Pulls out function signatures, file paths, complexity notes,
    and other structured coding information.

    Args:
        text: Text to extract from.

    Returns:
        Dict of extracted metadata fields.
    """
    metadata: dict = {}

    # Extract file paths
    file_matches = re.findall(r"(?:^|\s)([a-zA-Z_][\w/\\.-]*\.\w{1,6})(?:\s|$|:|\()", text)
    if file_matches:
        metadata["files"] = list(set(file_matches[:10]))

    # Extract function/method names (common patterns)
    fn_matches = re.findall(r"\b(?:fn|def|func|function)\s+(\w+)", text)
    if fn_matches:
        metadata["functions"] = list(set(fn_matches))

    # Extract class names
    class_matches = re.findall(r"\b(?:class|struct|interface|enum)\s+(\w+)", text)
    if class_matches:
        metadata["classes"] = list(set(class_matches))

    # Extract Big-O complexity
    complexity = re.findall(r"O\([^)]+\)", text)
    if complexity:
        metadata["complexity"] = list(set(complexity))

    # Extract performance numbers
    perf_nums = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|µs|ns|s|MB|GB|KB)\b", text)
    if perf_nums:
        metadata["performance"] = [f"{v}{u}" for v, u in perf_nums]

    # Extract error codes
    error_codes = re.findall(r"\b[A-Z]\d{4}\b", text)
    if error_codes:
        metadata["error_codes"] = list(set(error_codes))

    return metadata


def format_adr_block(
    title: str,
    context: str,
    decision: str,
    consequences: str,
    *,
    alternatives: list[str] | None = None,
    status: str = "accepted",
    block_id: str = "",
) -> str:
    """Format an Architecture Decision Record as a MIND block.

    Args:
        title: Decision title.
        context: What prompted this decision.
        decision: What was decided.
        consequences: Impact of the decision.
        alternatives: Other options considered.
        status: Decision status.
        block_id: Optional block ID.

    Returns:
        Formatted Markdown block string.
    """
    if not block_id:
        date_str = datetime.now().strftime("%Y%m%d")
        block_id = f"ADR-{date_str}-001"

    lines = [
        f"[{block_id}]",
        "Type: ADR",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Status: {status}",
        f"Title: {title}",
        f"Context: {context}",
        f"Decision: {decision}",
        f"Consequences: {consequences}",
    ]

    if alternatives:
        lines.append(f"Alternatives: {', '.join(alternatives)}")

    return "\n".join(lines) + "\n"
