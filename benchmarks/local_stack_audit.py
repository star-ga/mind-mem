#!/usr/bin/env python3
"""Single-shot audit of the local mind-mem stack before a bench run.

Prints a pass/fail table for every optional feature mind-mem can
exploit at bench time. Fails loudly (exit code 1) on any critical gap;
soft-warns (exit code 0) on missing nice-to-haves.

Usage::

    python3 benchmarks/local_stack_audit.py
    python3 benchmarks/local_stack_audit.py --strict   # fail on warnings
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Check:
    name: str
    run: Callable[[], tuple[str, str]]  # (status, detail)
    critical: bool = True


def _ok(msg: str) -> tuple[str, str]:
    return ("OK", msg)


def _warn(msg: str) -> tuple[str, str]:
    return ("WARN", msg)


def _fail(msg: str) -> tuple[str, str]:
    return ("FAIL", msg)


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def check_mind_mem_install() -> tuple[str, str]:
    try:
        import mind_mem

        return _ok(f"v{mind_mem.__version__} at {os.path.dirname(mind_mem.__file__)}")
    except Exception as e:  # noqa: BLE001
        return _fail(str(e))


def check_integrity() -> tuple[str, str]:
    try:
        import mind_mem

        r = mind_mem.verify_integrity()
        return _ok(f"mode={r.mode} manifest={r.manifest_present} checked={r.checked}")
    except Exception as e:  # noqa: BLE001
        return _fail(str(e))


def check_mind_kernels() -> tuple[str, str]:
    try:
        from mind_mem.mind_ffi import get_mind_dir, load_all_kernel_configs

        ws = os.environ.get("MIND_MEM_WORKSPACE", "/home/n/.openclaw/workspace")
        cfgs = load_all_kernel_configs(get_mind_dir(ws))
        required = {
            "query_plan", "graph", "session", "truth",
            "answer", "evidence", "ensemble",
            "hybrid", "rrf", "bm25", "temporal", "intent", "prefetch",
        }
        missing = required - cfgs.keys()
        if missing:
            return _fail(f"missing kernels: {sorted(missing)}")
        return _ok(f"{len(cfgs)} kernels loaded ({len(required)}/{len(required)} required)")
    except Exception as e:  # noqa: BLE001
        return _fail(str(e))


def check_ollama() -> tuple[str, str]:
    if not _port_open("127.0.0.1", 11434):
        return _fail("ollama not running on :11434")
    out = subprocess.run(
        ["curl", "-s", "http://127.0.0.1:11434/api/tags"],
        capture_output=True, text=True, timeout=5,
    ).stdout
    models = [m.get("name", "") for m in json.loads(out).get("models", [])]
    mm = [m for m in models if m.startswith("mind-mem")]
    if not mm:
        return _warn(f"mind-mem:4b not pulled (models={models})")
    return _ok(f"{', '.join(mm)} available")


def check_redis() -> tuple[str, str]:
    if not _port_open("127.0.0.1", 6379):
        return _warn("redis not running on :6379 (L2 cache disabled)")
    try:
        out = subprocess.run(
            ["redis-cli", "ping"], capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return _ok(out) if out == "PONG" else _fail(f"unexpected ping: {out}")
    except FileNotFoundError:
        return _warn("redis-cli not on PATH")


def check_claude_proxy() -> tuple[str, str]:
    if not _port_open("127.0.0.1", 8766):
        return _warn("claude-proxy not running on :8766 (Opus answerer falls back to Mistral)")
    return _ok("claude-proxy (Opus OAuth) responding")


def check_cross_encoder() -> tuple[str, str]:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    ce = hub / "models--cross-encoder--ms-marco-MiniLM-L-6-v2"
    if not ce.is_dir():
        return _fail("cross-encoder not cached")
    return _ok(f"cached at {ce}")


def check_bge_reranker() -> tuple[str, str]:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    bge = hub / "models--BAAI--bge-reranker-v2-m3"
    if not bge.is_dir():
        return _fail("bge-reranker-v2-m3 not cached (rerank ensemble will disable BGE member)")
    return _ok(f"cached at {bge}")


def check_sqlite_vec() -> tuple[str, str]:
    try:
        import sqlite_vec  # type: ignore[import-not-found]

        return _ok(f"v{getattr(sqlite_vec, '__version__', '?')}")
    except ImportError:
        return _warn("sqlite-vec not installed (vector recall disabled)")


def check_v3_3_modules() -> tuple[str, str]:
    modules = [
        "query_planner", "graph_recall", "entity_prefetch", "session_boost",
        "evidence_bundle", "rerank_ensemble", "truth_score", "answer_quality",
        "streaming", "consensus_vote", "retrieval_trace", "feature_gate",
    ]
    missing = []
    for m in modules:
        try:
            __import__(f"mind_mem.{m}")
        except ImportError as e:
            missing.append(f"{m}({e})")
    if missing:
        return _fail(f"missing: {missing}")
    return _ok(f"{len(modules)}/{len(modules)} v3.3.0 modules importable")


def check_v4_prep_modules() -> tuple[str, str]:
    modules = [
        "event_fanout", "tenant_audit", "tenant_kms", "governance_raft",
        "api.grpc_server", "storage.sharded_pg",
    ]
    missing = []
    for m in modules:
        try:
            __import__(f"mind_mem.{m}")
        except ImportError as e:
            missing.append(f"{m}({e})")
    if missing:
        return _fail(f"missing: {missing}")
    return _ok(f"{len(modules)}/{len(modules)} v4.0-prep modules importable")


CHECKS: list[Check] = [
    Check("mind_mem install", check_mind_mem_install),
    Check("integrity", check_integrity),
    Check("v3.3.0 modules", check_v3_3_modules),
    Check("v4.0-prep modules", check_v4_prep_modules),
    Check(".mind kernels", check_mind_kernels),
    Check("ollama mind-mem:4b", check_ollama, critical=False),
    Check("redis L2 cache", check_redis, critical=False),
    Check("claude-proxy", check_claude_proxy, critical=False),
    Check("cross-encoder HF cache", check_cross_encoder, critical=False),
    Check("bge-reranker-v2-m3 cache", check_bge_reranker, critical=False),
    Check("sqlite-vec", check_sqlite_vec, critical=False),
]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="Exit 1 on warnings too")
    args = parser.parse_args(argv)

    worst = "OK"
    print(f"{'CHECK':<28} {'STATUS':<6} {'DETAIL'}")
    print("-" * 80)
    for c in CHECKS:
        status, detail = c.run()
        print(f"{c.name:<28} {status:<6} {detail}")
        if status == "FAIL" and c.critical:
            worst = "FAIL"
        elif status == "WARN" and worst == "OK":
            worst = "WARN"
    print("-" * 80)
    print(f"overall: {worst}")

    if worst == "FAIL":
        return 1
    if worst == "WARN" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
