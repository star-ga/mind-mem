#!/usr/bin/env python3
"""Create a deterministic, source-bound training-readiness manifest.

This is intentionally a metadata-only check.  It reads a generated JSONL
corpus and the probe definitions, but never imports an inference backend or
loads model weights.  The main eval probes are training material by design;
only the holdout probe strings must be absent from the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "train"
SOURCE_NAMES = (
    "train/build_corpus.py",
    "train/eval_harness.py",
    "train/eval_holdout.py",
    "scripts/count_mcp_tools.py",
    "train/README.md",
    "train/HF_MODEL_CARD_v4.md",
    "ROADMAP.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    missing = [name for name in SOURCE_NAMES if not (REPO / name).is_file()]
    if missing:
        raise SystemExit(f"source file(s) missing: {', '.join(missing)}")
    return {name: sha256(REPO / name) for name in SOURCE_NAMES}


def git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def registered_tools() -> list[str]:
    """Mirror the static registration discovery used by count_mcp_tools.py."""
    count_path = REPO / "scripts" / "count_mcp_tools.py"
    spec = importlib.util.spec_from_file_location("mind_mem_count_mcp_tools", count_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {count_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names: set[str] = set()
    for path in module._tool_source_files():
        names.update(module._tool_names(path))
    return sorted(names)


def _probe_prompts(values: Iterable[Iterable[Any]]) -> list[str]:
    return [tuple_value[0] for tuple_value in values]


def probe_sets() -> tuple[list[str], list[str]]:
    """Load probe constants without importing torch/transformers."""
    original = list(sys.path)
    sys.path.insert(0, str(TRAIN))
    try:
        harness = __import__("eval_harness")
        holdout = __import__("eval_holdout")
        main_names = (
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
        )
        main = [prompt for name in main_names for prompt in _probe_prompts(getattr(harness, name))]
        holdout_prompts = _probe_prompts(holdout.V4_HOLDOUT + holdout.V312_HOLDOUT)
        return main, holdout_prompts
    finally:
        sys.path[:] = original


def read_corpus(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    users: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                messages = row["messages"]
                if not isinstance(messages, list):
                    raise ValueError("messages is not a list")
                if any(not isinstance(message, dict) for message in messages):
                    raise ValueError("message is not an object")
                user_messages = [
                    message["content"]
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str)
                ]
                if not user_messages:
                    raise ValueError("no user message")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise SystemExit(f"invalid corpus row {line_number}: {exc}") from exc
            rows.append(row)
            users.update(user_messages)
    return rows, users


def _mention_count(rows: list[dict[str, Any]], name: str) -> int:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    count = 0
    for row in rows:
        if any(pattern.search(message.get("content", "")) for message in row["messages"]):
            count += 1
    return count


def _read_model_facts() -> dict[str, Any]:
    readme = (REPO / "train" / "README.md").read_text(encoding="utf-8")
    model_card = (REPO / "train" / "HF_MODEL_CARD_v4.md").read_text(encoding="utf-8")
    default_match = re.search(r"current default base is `([^`]+)`", readme)
    trained_match = re.search(r"trained against an \*\*(\d+)-tool\*\* surface", model_card)
    if default_match is None or trained_match is None:
        raise SystemExit("could not bind base-model or trained-tool facts to source docs")
    return {
        "base_model_default": default_match.group(1),
        "base_model_requested": os.environ.get("MM_BASE_MODEL", default_match.group(1)),
        "qwen3_8_availability": "UNVERIFIED",
        "historical_model_card_trained_tool_count": int(trained_match.group(1)),
        "historical_model_card_path": "train/HF_MODEL_CARD_v4.md",
    }


def build_manifest(corpus: Path) -> dict[str, Any]:
    rows, users = read_corpus(corpus)
    tools = registered_tools()
    main, holdout = probe_sets()
    main_set = set(main)
    holdout_set = set(holdout)
    main_overlap = sorted(main_set & users)
    holdout_overlap = sorted(holdout_set & users)
    tool_counts = {name: _mention_count(rows, name) for name in tools}
    return {
        "schema": "mind-mem/training-readiness-manifest@1",
        "source_commit": git_commit(),
        "source_hashes": source_hashes(),
        "regeneration": {
            "command_template": "MM_CORPUS_OUT=<output>/corpus.jsonl python3 train/build_corpus.py",
            "manifest_command_template": "python3 train/training_readiness_manifest.py --corpus <output>/corpus.jsonl --output <output>/training-readiness-manifest.json",
            "network_calls": False,
            "model_loads": False,
        },
        "corpus": {
            "path": str(corpus),
            "sha256": sha256(corpus),
            "rows": len(rows),
            "unique_user_prompts": len(users),
            "invalid_rows": 0,
        },
        "live_mcp_surface": {
            "registered_tool_count": len(tools),
            "tool_names": tools,
            "coverage_basis": "textual tool-name mention in a corpus message; not model competence",
            "tools_missing_from_corpus": [name for name, count in tool_counts.items() if count == 0],
            "examples_mentioning_tool": tool_counts,
            "surface_coverage_status": "PASS" if all(tool_counts.values()) else "FAIL",
        },
        "evaluation_probe_binding": {
            "main_probe_count": len(main_set),
            "main_exact_overlap_count": len(main_overlap),
            "main_unseen_prompts": sorted(main_set - users),
            "main_eval_status": "TRAINING_OVERLAP",
            "main_eval_is_training_overlap": True,
            "holdout_probe_count": len(holdout_set),
            "holdout_exact_overlap_count": len(holdout_overlap),
            "holdout_exact_overlaps": holdout_overlap,
            "holdout_is_exact_string_guard": True,
            "training_eval_separation_status": "PASS" if not holdout_overlap else "FAIL",
        },
        "model_facts": _read_model_facts(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="generated corpus JSONL")
    parser.add_argument("--output", type=Path, required=True, help="manifest JSON destination")
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    if not corpus.is_file():
        raise SystemExit(f"corpus file not found: {corpus}")
    manifest = build_manifest(corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "corpus_sha256": manifest["corpus"]["sha256"],
                "rows": manifest["corpus"]["rows"],
                "registered_tool_count": manifest["live_mcp_surface"]["registered_tool_count"],
                "tools_missing_from_corpus": manifest["live_mcp_surface"]["tools_missing_from_corpus"],
                "main_exact_overlap_count": manifest["evaluation_probe_binding"]["main_exact_overlap_count"],
                "holdout_exact_overlap_count": manifest["evaluation_probe_binding"]["holdout_exact_overlap_count"],
                "training_eval_separation_status": manifest["evaluation_probe_binding"]["training_eval_separation_status"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return (
        0
        if (
            manifest["evaluation_probe_binding"]["training_eval_separation_status"] == "PASS"
            and manifest["live_mcp_surface"]["surface_coverage_status"] == "PASS"
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
