#!/usr/bin/env python3
"""Public benchmark entry point: verify saved evidence or run a pinned evaluation.

Copyright (c) STARGA Inc. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.release_scoring import CorrectedKey, aggregate, read_rows, sha256, verify_bundle, verify_correction_evidence  # noqa: E402

BUNDLE = ROOT / "benchmarks" / "public_release"


def download(spec: dict, cache: Path) -> Path:
    """Only atomic, hash-verified downloads enter the dataset cache."""
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / spec["filename"]
    if target.exists():
        if sha256(target) != spec["sha256"]:
            raise ValueError("Cached dataset checksum mismatch; remove it explicitly to download again")
        return target
    temp = target.with_suffix(target.suffix + ".partial")
    try:
        with urllib.request.urlopen(spec["url"], timeout=120) as response, temp.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha256(temp) != spec["sha256"]:
            raise ValueError("Downloaded dataset checksum mismatch")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def verify_partitions(rows: list[dict], split: dict) -> None:
    for row in rows:
        if row["benchmark"] not in {"locomo", "longmemeval_s"}:
            raise ValueError("Unknown dataset in full evaluation")
        if row["benchmark"] == "locomo":
            held = split["locomo_heldout_conversations"]
            dev = split["locomo_dev_conversations"]
            if row["group"] not in held + dev:
                raise ValueError("Conversation absent from frozen split")
        else:
            held = split["lme_heldout_question_ids"]
        expected = "held_out" if row["group"] in held else "dev"
        if row["partition"] != expected:
            raise ValueError("Saved partition disagrees with frozen split")
    for benchmark, n, h in (("locomo", 1986, 944), ("longmemeval_s", 500, 243)):
        selected = [r for r in rows if r["benchmark"] == benchmark]
        if len(selected) != n or sum(r["partition"] == "held_out" for r in selected) != h:
            raise ValueError("Incomplete benchmark population")


def rescore(benchmark: str, bundle: Path, historical_audit: bool = False, cache: Path | None = None) -> dict:
    manifest, rows = verify_bundle(bundle)
    from benchmarks.release_inputs import corrected_dataset, reconstruct

    config = json.loads((bundle / "protocol.json").read_text(encoding="utf-8"))
    cache = cache or ROOT / "benchmarks" / ".cache" / "release"
    datasets = {}
    for name in ("locomo", "longmemeval_s"):
        spec = config["datasets"][name]
        path = cache / spec["filename"]
        if not path.exists() or sha256(path) != spec["sha256"]:
            raise ValueError("Pinned dataset cache missing or corrupt; run --prepare first")
        datasets[name] = json.loads(path.read_text(encoding="utf-8"))
    rows = reconstruct(bundle, datasets, rows)
    split = json.loads((bundle / "HELDOUT_SPLIT_20261029.json").read_text(encoding="utf-8"))
    verify_partitions(rows, split)
    corrections = read_rows(bundle / "corrections.jsonl")
    if any(r["source"] != "independent audit" for r in corrections):
        raise ValueError("Ineligible correction source")
    verify_correction_evidence(bundle, corrections, datasets["locomo"], corrected_dataset(datasets["locomo"], corrections))
    adjudications = read_rows(bundle / "adjudications.jsonl")
    key = CorrectedKey(corrections, adjudications)
    known = {r["id"] for r in rows}
    if not set(key.rows) <= known:
        raise ValueError("Correction outside evaluated population")
    selected = [r for r in rows if r["benchmark"] == benchmark]
    groups = {"all": selected}
    if benchmark == "locomo":
        groups.update(categories_1_4=[r for r in selected if r["category"] != 5], category_5=[r for r in selected if r["category"] == 5])
    result = {
        "benchmark": benchmark,
        "status": manifest["status"],
        "configuration_a": {
            "official_key": {g: aggregate(v) for g, v in groups.items()},
            "corrected_key_verification_pending": {g: aggregate(v, key) for g, v in groups.items()},
        },
        "configuration_b": {
            "status": "not measured; reference answer-model and immutable judge revision unresolved",
            "full": None,
            "dev": None,
            "held_out": None,
        },
        "correction_selection": {"included": len(corrections), "excluded": manifest["excluded_correction_candidates"]},
        "judge_replay": "saved majority votes; semantic correctness is not independently rejudged",
        "exact_match_and_token_f1": "computed from raw answers without a model",
    }
    return result


def niah_rescore() -> dict:
    from benchmarks.repro_metrics import niah_metrics
    from benchmarks.repro_verify import verify_package

    path = ROOT / "benchmarks" / "repro" / "niah"
    report = verify_package(str(path))
    if not report.passed:
        raise ValueError(report.render())
    rows = read_rows(path / "raw.ndjson")
    result = {
        "benchmark": "NIAH",
        "configuration": "retrieval only; no answer model or judge",
        "split_note": "No untouched held-out NIAH set exists; every saved case was previously exposed.",
    }
    result["full"] = niah_metrics(rows)["headline"]
    result["dev"] = result["full"]
    result["held_out"] = {"passed": None, "total": 0, "value": None}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("locomo", "longmemeval_s", "niah"))
    parser.add_argument("--prepare", action="store_true", help="download pinned datasets and verify rebuilt prompts")
    parser.add_argument("--rescore", action="store_true", help="offline only; no API, model or dataset download")
    parser.add_argument("--historical-baseline", action="store_true", help="replay the earlier public graded baseline")
    parser.add_argument("--historical-audit", action="store_true", help="also show excluded legacy correction selection")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--cache", type=Path, default=ROOT / "benchmarks" / ".cache" / "release")
    parser.add_argument("--config", type=Path, default=BUNDLE / "protocol.json")
    parser.add_argument("--configuration", choices=("a", "b"), default="a")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--answer-model-id")
    parser.add_argument("--answer-revision")
    parser.add_argument("--judge-model-id")
    parser.add_argument("--judge-revision")
    parser.add_argument("--judge-url", default=os.environ.get("BENCH_JUDGE_URL"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.historical_baseline or args.historical_audit:
            raise ValueError("Excluded historical artifacts are not distributed in this kit")
        if args.prepare:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            for name in ("locomo", "longmemeval_s"):
                download(config["datasets"][name], args.cache)
            rescore(args.benchmark, args.bundle, cache=args.cache)
            print("Verified 2486 reconstructed prompt hashes and 96 corrections")
            return 0
        if args.historical_baseline and (not args.rescore or args.benchmark != "locomo"):
            raise ValueError("Historical baseline requires locomo --rescore")
        if args.rescore:
            result = niah_rescore() if args.benchmark == "niah" else rescore(args.benchmark, args.bundle, args.historical_audit, args.cache)
        else:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            if args.benchmark == "niah":
                if args.download_only:
                    path = download(config["datasets"]["niah"], args.cache)
                    result = {"dataset": str(path), "sha256": sha256(path)}
                else:
                    descriptor = download(config["datasets"]["niah"], args.cache)
                    if descriptor.read_bytes() != (ROOT / "benchmarks/repro/niah/dataset.json").read_bytes():
                        raise ValueError("Pinned synthetic dataset differs from the local generator descriptor")
                    if json.loads(descriptor.read_text(encoding="utf-8"))["generator_sha256"] != sha256(ROOT / "tests/test_niah.py"):
                        raise ValueError("Synthetic generator source differs from the pinned descriptor")
                    os.environ["PYTHONHASHSEED"] = str(config["seed"])
                    from benchmarks.repro_niah import main as run_niah

                    return run_niah(["--out", str(args.out or args.cache / "niah-run"), "--headline-claim"])
            else:
                path = download(config["datasets"][args.benchmark], args.cache)
                if args.download_only:
                    result = {"dataset": str(path), "sha256": sha256(path)}
                else:
                    from benchmarks.release_runtime import run

                    result = run(args, config, path)
        if args.rescore:
            expected_path = args.bundle / "expected_scores.json"
            if expected_path.exists():
                expected_name = (
                    "historical_baseline" if args.historical_baseline else {"longmemeval_s": "lme"}.get(args.benchmark, args.benchmark)
                )
                expected = json.loads(expected_path.read_text(encoding="utf-8"))[expected_name]
                if not args.historical_audit:
                    expected.pop("legacy_selection_audit_not_publishable", None)
                if result != expected:
                    raise ValueError("Recomputed aggregates differ from the committed expected outputs")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
