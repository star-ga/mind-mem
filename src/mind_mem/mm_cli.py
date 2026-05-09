# Copyright 2026 STARGA, Inc.
"""``mm`` — unified mind-mem CLI for non-MCP agents (v2.7.0).

Usage::

    mm recall "<query>"             # search memory
    mm context "<query>"            # generate token-budgeted snippet
    mm inject --agent <name> "<q>"  # render snippet for a specific agent
    mm vault scan <vault_root>      # list parsed vault blocks (JSON)
    mm vault write <vault_root> <id> --type <t> --body <b>
    mm status                       # workspace summary

Filesystem-watcher integration and the per-agent hook installer
remain deferred (need ``watchdog`` and per-agent setup scripts).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Workspace resolution (mirrors mcp_server._workspace)
# ---------------------------------------------------------------------------


def _workspace() -> str:
    ws = os.environ.get("MIND_MEM_WORKSPACE", "").strip()
    if not ws:
        ws = os.getcwd()
    return os.path.realpath(ws)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_recall(args: argparse.Namespace) -> int:
    from mind_mem.recall import recall

    results = recall(_workspace(), args.query, limit=args.limit, active_only=args.active_only)
    print(json.dumps(results, indent=2, default=str))
    return 0


def _cmd_context(args: argparse.Namespace) -> int:
    from mind_mem.cognitive_forget import pack_to_budget
    from mind_mem.recall import recall

    results = recall(_workspace(), args.query, limit=args.limit, active_only=False)
    if not isinstance(results, list):
        results = [results] if isinstance(results, dict) else []
    packed = pack_to_budget(results, max_tokens=args.max_tokens)
    payload = {
        "query": args.query,
        "included": packed.included,
        "dropped": packed.dropped,
        **packed.as_dict(),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _cmd_inject(args: argparse.Namespace) -> int:
    from mind_mem.agent_bridge import AgentFormatter
    from mind_mem.recall import recall

    fmt = AgentFormatter(max_blocks=args.limit)
    results = recall(_workspace(), args.query, limit=args.limit, active_only=False)
    if not isinstance(results, list):
        results = [results] if isinstance(results, dict) else []
    print(fmt.inject(args.agent, args.query, results), end="")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    ws = _workspace()
    info: dict[str, Any] = {"workspace": ws, "exists": os.path.isdir(ws)}
    cfg_path = os.path.join(ws, "mind-mem.json")
    info["config_exists"] = os.path.isfile(cfg_path)
    for sub in ("decisions", "tasks", "entities", "intelligence", "memory"):
        info[f"{sub}_dir_exists"] = os.path.isdir(os.path.join(ws, sub))
    print(json.dumps(info, indent=2))
    return 0 if info["exists"] else 1


def _cmd_vault_scan(args: argparse.Namespace) -> int:
    from mind_mem.agent_bridge import VaultBridge

    bridge = VaultBridge(vault_root=args.vault_root)
    sync_dirs = args.sync_dirs or None
    blocks = bridge.scan(sync_dirs=sync_dirs)
    print(json.dumps([b.as_dict() for b in blocks], indent=2, default=str))
    return 0


def _extract_embed_text(block: dict) -> str:
    """Pick the best free-text content from a block for embedding.

    Priority: ``Statement`` → ``content`` → ``Subject`` + ``Excerpt`` →
    every non-private string field concatenated. Returns "" when no text
    is available (caller skips the block).
    """
    for key in ("Statement", "content"):
        v = block.get(key)
        if isinstance(v, str) and v.strip():
            return v
    parts: list[str] = []
    subj = block.get("Subject")
    if isinstance(subj, str) and subj.strip():
        parts.append(subj)
    excerpt = block.get("Excerpt")
    if isinstance(excerpt, str) and excerpt.strip():
        parts.append(excerpt)
    if parts:
        return ". ".join(parts)
    # Last-resort: glue every non-private string-ish field together.
    fallback: list[str] = []
    for k, v in block.items():
        if k.startswith("_") or k in ("_id", "_source_file"):
            continue
        if isinstance(v, str) and v.strip():
            fallback.append(f"{k}: {v}")
    return " | ".join(fallback)


def _redact_dsn(dsn: str) -> str:
    """Redact password fields from a Postgres DSN before logging it.

    Handles BOTH formats psycopg accepts:
      * URL form:        postgresql://user:secret@host:5432/db
      * Keyword form:    host=localhost password=secret dbname=db

    The URL parser misses the keyword form entirely (urlparse returns
    no scheme + no .password), so v3.8.13's receipt JSON leaked the
    password verbatim for any operator using the keyword DSN style.
    Both formats now route through the same redaction.
    """
    import re

    if not dsn:
        return dsn
    # URL form first.
    try:
        from urllib.parse import urlparse, urlunparse

        u = urlparse(dsn)
        if u.scheme and u.password:
            new_netloc = f"{u.username or ''}:***@{u.hostname or ''}"
            if u.port:
                new_netloc += f":{u.port}"
            return urlunparse(u._replace(netloc=new_netloc))
    except Exception:
        pass
    # Keyword form (postgresql://… without password OR plain key=value).
    return re.sub(r"(?i)\bpassword\s*=\s*\S+", "password=***", dsn)


def _embed_via_ollama(texts: list[str], model: str = "mxbai-embed-large") -> list[list[float]]:
    """Minimal Ollama embedder for migrate-store --with-embeddings.

    Kept inside mm_cli to avoid pulling the recall_vector module's heavy
    deps just for migration. Mirrors recall_vector.embed_ollama batching
    behaviour. URL is hardcoded to localhost loopback only.
    """
    import urllib.request as _req

    MAX_CHARS = 1500
    BATCH = 16
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = [t[:MAX_CHARS] for t in texts[i : i + BATCH]]
        payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
        req = _req.Request(
            "http://localhost:11434/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with _req.urlopen(req, timeout=180) as resp:  # nosec B310 — loopback only.
            data = json.loads(resp.read().decode("utf-8"))
        out.extend(data.get("embeddings", []))
    return out


def _cmd_migrate_store(args: argparse.Namespace) -> int:
    """Migrate the workspace block corpus between backends.

    v3.8.12 — implements the documented ``mm migrate-store`` flow
    (`docs/storage-migration.md`). Today's only supported direction is
    markdown -> postgres because that's the one operators actually
    need; markdown <- postgres is a follow-up.

    Steps when --execute:
      1. Load all blocks from the source MarkdownBlockStore.
      2. Ensure the Postgres schema exists (idempotent).
      3. Insert each block via PostgresBlockStore.write_block (which
         is itself an INSERT ... ON CONFLICT DO UPDATE).
      4. Verify SELECT COUNT(*) matches the migrated count.
      5. Write a JSON receipt to memory/migrations/<ts>-<from>-to-<to>.json.

    --dry-run prints the projected counts without writing anything.

    v3.8.13 — ``--with-embeddings`` adds an embedding-backfill pass after
    the row insert. Uses Ollama by default with mxbai-embed-large; the
    embedder model name comes from ``--embed-model`` (default
    ``mxbai-embed-large``). Skipped silently when pgvector is missing
    on the target.
    """
    import datetime
    import time

    from mind_mem.block_store import MarkdownBlockStore

    if args.from_backend != "markdown" or args.to_backend != "postgres":
        print(
            json.dumps(
                {
                    "error": f"unsupported direction: {args.from_backend} -> {args.to_backend}",
                    "supported": ["markdown -> postgres"],
                },
                indent=2,
            )
        )
        return 2
    if not args.dsn:
        print(json.dumps({"error": "--dsn is required for postgres target"}, indent=2))
        return 2
    if not args.execute and not args.dry_run:
        print(json.dumps({"error": "specify --dry-run or --execute"}, indent=2))
        return 2

    try:
        from mind_mem.block_store_postgres import PostgresBlockStore
    except ImportError as exc:
        print(
            json.dumps(
                {
                    "error": "postgres backend requires the [postgres] extra",
                    "install": "pip install 'mind-mem[postgres]'",
                    "detail": str(exc),
                },
                indent=2,
            )
        )
        return 2

    ws = _workspace()
    redacted_dsn = _redact_dsn(args.dsn)

    src = MarkdownBlockStore(ws)
    blocks = src.get_all()
    with_id = [b for b in blocks if b.get("_id")]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "from_backend": args.from_backend,
                    "to_backend": args.to_backend,
                    "workspace": ws,
                    "dsn": redacted_dsn,
                    "schema": args.schema,
                    "block_count_total": len(blocks),
                    "block_count_migratable": len(with_id),
                    "blocks_skipped_no_id": len(blocks) - len(with_id),
                },
                indent=2,
            )
        )
        return 0

    dst = PostgresBlockStore(args.dsn, schema=args.schema, workspace=ws)
    dst._ensure_schema()

    t0 = time.monotonic()
    written = 0
    errors: list[dict[str, str]] = []
    for b in with_id:
        try:
            dst.write_block(b)
            written += 1
        except Exception as exc:
            errors.append({"id": str(b.get("_id")), "error": str(exc)[:200]})
    duration = round(time.monotonic() - t0, 2)

    embedded = 0
    embed_errors: list[dict[str, str]] = []
    if args.with_embeddings:
        if not dst._has_vector:
            embed_errors.append({"id": "*", "error": "pgvector not available on target — skipped"})
        else:
            embed_t0 = time.monotonic()
            for b in with_id:
                bid = str(b.get("_id"))
                content = _extract_embed_text(b)
                if not content:
                    continue
                try:
                    vecs = _embed_via_ollama([content], model=args.embed_model)
                    if not vecs:
                        raise RuntimeError("embedder returned empty list")
                    dst.backfill_embedding(bid, vecs[0])
                    embedded += 1
                except Exception as exc:
                    embed_errors.append({"id": bid, "error": str(exc)[:200]})
            duration += round(time.monotonic() - embed_t0, 2)

    # Verify via the existing pool — keeps connection counts honest and
    # avoids a second psycopg.connect that would bypass the pool's
    # health-check + replay logic.
    from psycopg import sql as _pgsql

    with dst._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_pgsql.SQL("SELECT COUNT(*) FROM {s}.blocks").format(s=_pgsql.Identifier(args.schema)))
            target_count = cur.fetchone()[0]

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt = {
        "timestamp": ts,
        "from_backend": args.from_backend,
        "to_backend": args.to_backend,
        "workspace": ws,
        "dsn": redacted_dsn,
        "schema": args.schema,
        "source_block_count": len(blocks),
        "source_blocks_migratable": len(with_id),
        "target_row_count": target_count,
        "rows_written": written,
        "errors": len(errors),
        "duration_seconds": duration,
        "blocks_per_second": round(written / duration, 1) if duration > 0 else None,
        "verified": target_count >= written,
        "embeddings": {
            "requested": bool(args.with_embeddings),
            "available": bool(dst._has_vector) if args.with_embeddings else None,
            "embedder": args.embed_model if args.with_embeddings else None,
            "rows_embedded": embedded,
            "errors": len(embed_errors),
        },
    }
    if errors:
        receipt["error_sample"] = errors[:10]
    if embed_errors:
        receipt["embeddings"]["error_sample"] = embed_errors[:10]

    rec_dir = os.path.join(ws, "memory", "migrations")
    os.makedirs(rec_dir, exist_ok=True)
    rec_path = os.path.join(rec_dir, f"{ts}-{args.from_backend}-to-{args.to_backend}.json")
    with open(rec_path, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2)
    receipt["receipt_path"] = rec_path
    print(json.dumps(receipt, indent=2))
    return 0 if not errors and target_count >= written else 1


def _cmd_install(args: argparse.Namespace) -> int:
    """Install mind-mem config for a single named agent."""
    from mind_mem.hook_installer import install_config

    ws = _workspace()
    try:
        result = install_config(args.agent, ws, dry_run=args.dry_run, force=args.force)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


def _cmd_install_model(args: argparse.Namespace) -> int:
    """Download mind-mem-4b GGUF from HF and import into Ollama.

    Idempotent — safe to re-run. Skips download if file already
    present with matching size; skips Ollama import if tag already
    present (unless --force on the parent command).
    """
    import shutil
    import re
    import subprocess  # nosec B404 — used with absolute paths from shutil.which + list args, never shell=True
    import urllib.parse
    import urllib.request

    hf_repo = "star-ga/mind-mem-4b"

    # Validate args.model: filename only, no path traversal, no URL injection.
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.gguf", args.model):
        print(json.dumps({"error": f"invalid --model {args.model!r}; must match [A-Za-z0-9._-]+\\.gguf"}, indent=2))
        return 1
    # Validate args.name: alphanumerics + : _ . - / — Ollama tag charset.
    if not re.fullmatch(r"[A-Za-z0-9._:/\-]+", args.name):
        print(json.dumps({"error": f"invalid --name {args.name!r}; must match [A-Za-z0-9._:/-]+"}, indent=2))
        return 1
    # Validate args.keep-alive: -1 | <number><unit> (e.g. 30m, 1h, 24h)
    if not re.fullmatch(r"(-1|\d+(s|m|h|d)?)", str(args.keep_alive)):
        print(json.dumps({"error": f"invalid --keep-alive {args.keep_alive!r}"}, indent=2))
        return 1

    gguf_url = f"https://huggingface.co/{hf_repo}/resolve/main/{args.model}"
    # Defense in depth: confirm the URL we built is HTTPS + huggingface.co.
    parsed = urllib.parse.urlparse(gguf_url)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        print(json.dumps({"error": "internal: refusing to fetch from non-HF URL"}, indent=2))
        return 1
    dest = os.path.realpath(os.path.expanduser(args.dest))
    # Refuse writes to canonical system paths. We allow $HOME-symlinked-to-/data
    # (common on workstations with a separate SSD for ~/.cache), so we deny by
    # blacklist instead of confining by allowlist.
    _SYSTEM_PREFIXES = ("/etc/", "/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/",
                        "/var/", "/sys/", "/proc/", "/dev/", "/root/", "/boot/")
    if any(dest.startswith(p) for p in _SYSTEM_PREFIXES):
        print(json.dumps({"error": f"refusing to write to system path: {dest}"}, indent=2))
        return 1

    output: dict[str, Any] = {
        "model_file": args.model,
        "ollama_tag": args.name,
        "dest": dest,
        "keep_alive": args.keep_alive,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        output["plan"] = [
            f"would download {gguf_url}  ->  {dest}",
            f"would write Modelfile with FROM {dest}",
            f"would run `ollama create {args.name} -f Modelfile`",
            f"would set OLLAMA_KEEP_ALIVE={args.keep_alive}",
        ]
        print(json.dumps(output, indent=2))
        return 0

    # 1. Check ollama on PATH (graceful skip if absent)
    if not shutil.which("ollama"):
        output["error"] = "ollama not found on PATH"
        output["hint"] = "install ollama from https://ollama.com/download then re-run `mm install-model`"
        print(json.dumps(output, indent=2))
        return 2

    # 2. Download GGUF (skip if dest already correct size).
    # URL is constrained above to https://huggingface.co/<known repo>/<validated filename>;
    # bandit B310 doesn't see the validation but it's enforced.
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    expected_size = None
    req = urllib.request.Request(gguf_url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310 — URL is parse-validated above (https + huggingface.co only)
            expected_size = int(resp.headers.get("Content-Length") or 0)
    except Exception as exc:
        output["error"] = f"could not query HF for {args.model}: {exc}"
        print(json.dumps(output, indent=2))
        return 3

    if os.path.exists(dest) and expected_size and os.path.getsize(dest) == expected_size:
        output["downloaded"] = False
        output["reason"] = "dest already present with matching size"
    else:
        try:
            req = urllib.request.Request(gguf_url)
            with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as fh:  # nosec B310 — URL is parse-validated above (https + huggingface.co only)
                while chunk := resp.read(8 * 1024 * 1024):
                    fh.write(chunk)
            output["downloaded"] = True
            output["bytes"] = os.path.getsize(dest)
        except Exception as exc:
            output["error"] = f"download failed: {exc}"
            print(json.dumps(output, indent=2))
            return 4

    # 3. Build Modelfile next to the GGUF (idempotent)
    modelfile = os.path.join(os.path.dirname(dest), "Modelfile")
    modelfile_body = (
        f"FROM {dest}\n"
        "PARAMETER temperature 0.6\n"
        "PARAMETER top_p 0.95\n"
        "PARAMETER num_ctx 8192\n"
        f'PARAMETER stop "<|im_end|>"\n'
    )
    with open(modelfile, "w", encoding="utf-8") as fh:
        fh.write(modelfile_body)
    output["modelfile"] = modelfile

    # 4. Ollama import — args.name and modelfile are validated above;
    # we resolve `ollama` to its absolute path and pass argv as a list
    # (never shell=True) so B603/B607 do not apply.
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        output["error"] = "ollama disappeared from PATH between checks"
        print(json.dumps(output, indent=2))
        return 2
    try:
        result = subprocess.run(  # nosec B603 B607 — argv list w/ absolute path from shutil.which, no shell, validated args
            [ollama_bin, "create", args.name, "-f", modelfile],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        output["ollama_create_returncode"] = result.returncode
        if result.returncode != 0:
            output["ollama_stderr"] = result.stderr[-500:]
            print(json.dumps(output, indent=2))
            return 5
    except subprocess.TimeoutExpired:
        output["error"] = "`ollama create` timed out after 180s"
        print(json.dumps(output, indent=2))
        return 6

    # 5. Smoke test (warm the model + keep-alive). Same safety profile
    # as step 4: absolute path + argv list + validated args, no shell.
    try:
        smoke = subprocess.run(  # nosec B603 B607 — argv list w/ absolute path from shutil.which, no shell, validated args
            [ollama_bin, "run", args.name, "test"],
            input="hi\n",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={**os.environ, "OLLAMA_KEEP_ALIVE": args.keep_alive},
        )
        output["smoke_returncode"] = smoke.returncode
        if smoke.returncode == 0:
            output["smoke_response_first_60"] = smoke.stdout.strip()[:60]
    except subprocess.TimeoutExpired:
        output["smoke_response"] = "(timeout — model likely importing in background; run `ollama list` to verify)"

    output["status"] = "ok"
    output["next_steps"] = [
        f"ollama run {args.name}  # test the model",
        "mm status                # confirm mind-mem.json is configured",
    ]
    print(json.dumps(output, indent=2))
    return 0


def _cmd_install_all(args: argparse.Namespace) -> int:
    """Auto-detect installed AI clients and configure all of them."""
    from mind_mem.hook_installer import detect_installed_agents, install_all

    ws = _workspace()
    agents = args.agent or None  # None → auto-detect
    results = install_all(
        ws,
        dry_run=args.dry_run,
        force=args.force,
        agents=agents,
        include_mcp=not getattr(args, "no_mcp", False),
    )
    errored = sum(1 for r in results if "error" in r)
    summary: dict[str, Any] = {
        "workspace": ws,
        "agents": agents if agents is not None else detect_installed_agents(ws),
        "results": results,
        "summary": {
            "written": sum(1 for r in results if r.get("written")),
            "merged": sum(1 for r in results if r.get("merged")),
            "skipped": sum(1 for r in results if r.get("skipped")),
            "errored": errored,
            "total": len(results),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0 if errored == 0 else 1


def _cmd_detect(args: argparse.Namespace) -> int:
    """List AI clients detected on the current machine."""
    from mind_mem.hook_installer import AGENT_REGISTRY, detect_installed_agents

    ws = _workspace()
    detected = detect_installed_agents(ws)
    out = {
        "workspace": ws,
        "detected": detected,
        "details": {
            name: {
                "description": AGENT_REGISTRY[name].description,
                "config_path": AGENT_REGISTRY[name].expand_path(ws),
            }
            for name in detected
        },
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_vault_write(args: argparse.Namespace) -> int:
    from mind_mem.agent_bridge import VaultBlock, VaultBridge

    bridge = VaultBridge(vault_root=args.vault_root)
    block = VaultBlock(
        relative_path=args.relative_path,
        block_id=args.id,
        block_type=args.type,
        title=args.title or args.id,
        body=args.body,
    )
    target = bridge.write(block, overwrite=args.overwrite)
    print(json.dumps({"written": target}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Debug visualization subcommands (v3.2.0)
# ---------------------------------------------------------------------------

# ANSI colour codes — used only when stdout is a TTY
_C_RESET = "\033[0m"
_C_BOLD = "\033[1m"
_C_DIM = "\033[2m"
_C_GREEN = "\033[32m"
_C_YELLOW = "\033[33m"
_C_CYAN = "\033[36m"
_C_RED = "\033[31m"

_INSPECT_FIXED_FIELDS = ("_id", "Statement", "Status", "Date", "Tags")
_INSPECT_SKIP = frozenset({"_source_file", "_line_number", "_raw"})


def _use_color() -> bool:
    return sys.stdout.isatty()


def _c(text: str, *codes: str) -> str:
    if not _use_color():
        return text
    return "".join(codes) + text + _C_RESET


def _build_provenance(workspace: str, block_id: str) -> dict[str, Any]:
    """Gather edges + causal chains for a block from the CausalGraph."""
    from mind_mem.causal_graph import CausalGraph

    graph = CausalGraph(workspace)
    deps = [e.to_dict() for e in graph.dependencies(block_id)]
    chains = graph.causal_chain(block_id, max_depth=3)
    contradiction_types = {"contradicts", "supersedes"}
    all_edges = deps + [e.to_dict() for e in graph.dependents(block_id)]
    contradictions = [e for e in all_edges if e.get("edge_type") in contradiction_types]
    return {
        "block_id": block_id,
        "dependencies": deps,
        "causal_chains": chains,
        "contradictions": contradictions,
    }


def _render_inspect_text(block: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines: list[str] = []
    block_id = block.get("_id", "?")
    lines.append(_c(f"Block: {block_id}", _C_BOLD, _C_CYAN))
    lines.append(_c("─" * 60, _C_DIM))
    for key in _INSPECT_FIXED_FIELDS:
        if key in block and key not in _INSPECT_SKIP:
            lines.append(f"{_c(f'  {key}:', _C_BOLD)} {block[key]}")
    remaining = sorted(k for k in block if k not in _INSPECT_FIXED_FIELDS and k not in _INSPECT_SKIP)
    for key in remaining:
        lines.append(f"{_c(f'  {key}:', _C_DIM)} {block[key]}")

    lines.append("")
    lines.append(_c("Provenance", _C_BOLD, _C_YELLOW))
    lines.append(_c("─" * 60, _C_DIM))
    deps = provenance.get("dependencies", [])
    if deps:
        lines.append(_c("  Direct dependencies:", _C_BOLD))
        for edge in deps:
            tid = edge["target_id"]
            etype = edge["edge_type"]
            w = edge.get("weight", 1.0)
            lines.append(f"  {_c(f'→ {tid}', _C_GREEN)}  [{etype}]  weight={w:.2f}")
    else:
        lines.append(_c("  No direct dependencies.", _C_DIM))

    chains = provenance.get("causal_chains", [])
    if chains:
        lines.append(_c("  Causal chains (depth ≤ 3):", _C_BOLD))
        for chain in chains:
            lines.append("    " + _c(" → ", _C_DIM).join(chain))

    contradictions = provenance.get("contradictions", [])
    if contradictions:
        lines.append(_c("  Contradictions / supersessions:", _C_BOLD))
        for edge in contradictions:
            lines.append(f"  {_c(edge['source_id'], _C_RED)} [{edge['edge_type']}] {_c(edge['target_id'], _C_RED)}")
    return "\n".join(lines)


def _cmd_inspect(args: argparse.Namespace) -> int:
    ws = _workspace()
    from mind_mem.block_store import MarkdownBlockStore

    store = MarkdownBlockStore(ws)
    block = store.get_by_id(args.block_id)
    if block is None:
        print(json.dumps({"error": f"Block not found: {args.block_id}"}, indent=2), file=sys.stderr)
        return 1

    provenance = _build_provenance(ws, args.block_id)
    fmt = getattr(args, "format", "text") or "text"
    if fmt == "json":
        print(json.dumps({"block": block, "provenance": provenance}, indent=2, default=str))
    else:
        print(_render_inspect_text(block, provenance))
    return 0


# ── mm explain ────────────────────────────────────────────────────────────────


def _stage_bar(stages_hit: list[bool]) -> str:
    if _use_color():
        symbols = [_c("◉", _C_GREEN) if s else _c("◯", _C_DIM) for s in stages_hit]
    else:
        symbols = [f"[{'x' if s else ' '}]" for s in stages_hit]
    return " ".join(symbols)


def _cmd_explain(args: argparse.Namespace) -> int:
    ws = _workspace()
    limit = getattr(args, "limit", 10) or 10
    backend = getattr(args, "backend", "auto") or "auto"

    if backend == "hybrid":
        try:
            from mind_mem.hybrid_recall import HybridBackend

            cfg: dict[str, Any] = {}
            cfg_path = os.path.join(ws, "mind-mem.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, encoding="utf-8") as fh:
                    cfg = json.load(fh).get("recall", {})
            results = HybridBackend(cfg).search(ws, args.query, limit=limit)
        except Exception:
            from mind_mem.recall import recall as _recall

            results = _recall(ws, args.query, limit=limit)
    else:
        from mind_mem.recall import recall as _recall

        results = _recall(ws, args.query, limit=limit)

    trace_rows: list[dict[str, Any]] = []
    for rank, r in enumerate(results, 1):
        bm25 = r.get("score", r.get("bm25_score", 0.0))
        vec = r.get("vector_score", r.get("embed_score"))
        rrf = r.get("rrf_score")
        rerank = r.get("rerank_score", r.get("ce_score"))
        trace_rows.append(
            {
                "rank": rank,
                "block_id": r.get("_id", "?"),
                "bm25": round(float(bm25), 4),
                "vector": round(float(vec), 4) if vec is not None else None,
                "rrf": round(float(rrf), 4) if rrf is not None else None,
                "rerank": round(float(rerank), 4) if rerank is not None else None,
                "stages_hit": [True, vec is not None, rrf is not None, rerank is not None],
            }
        )

    from mind_mem.retrieval_graph import retrieval_diagnostics

    diag = retrieval_diagnostics(ws, last_n=20)

    fmt = getattr(args, "format", "text") or "text"
    if fmt == "json":
        print(json.dumps({"query": args.query, "results": trace_rows, "diagnostics": diag}, indent=2, default=str))
        return 0

    print(_c(f"Retrieval trace: {args.query!r}", _C_BOLD, _C_CYAN))
    print(_c("─" * 70, _C_DIM))
    header = f"  {'#':>4}  {'BLOCK':20}  {'BM25':>7}  {'VEC':>7}  {'RRF':>7}  {'RERANK':>7}  STAGES"
    print(_c(header, _C_BOLD))
    print(_c("─" * 70, _C_DIM))
    for row in trace_rows:
        bm25_s = f"{row['bm25']:7.4f}"
        vec_s = f"{row['vector']:7.4f}" if row["vector"] is not None else "      -"
        rrf_s = f"{row['rrf']:7.4f}" if row["rrf"] is not None else "      -"
        rer_s = f"{row['rerank']:7.4f}" if row["rerank"] is not None else "      -"
        bid = str(row["block_id"])[:20]
        print(f"  {row['rank']:>4}  {bid:20}  {bm25_s}  {vec_s}  {rrf_s}  {rer_s}  {_stage_bar(row['stages_hit'])}")

    print()
    print(_c("Diagnostics summary", _C_BOLD, _C_YELLOW))
    intent_dist = diag.get("intent_distribution", {})
    if intent_dist:
        top_intent = max(intent_dist, key=lambda k: intent_dist[k])
        print(f"  Intent: {_c(top_intent, _C_CYAN)} ({intent_dist[top_intent]} recent queries)")
    for stage, rate in sorted(diag.get("stage_rejection_rates", {}).items()):
        print(f"  {stage}: {rate:.1%} rejected")
    return 0


# ── mm trace ─────────────────────────────────────────────────────────────────

_TRACE_HEADER = "TIME              TOOL                           DURATION  STATUS  SIZE"


def _parse_log_lines(lines: list[str], *, tool_filter: Optional[str] = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry.get("event") != "mcp_tool_call":
            continue
        data = entry.get("data") or {}
        tool = data.get("tool") or entry.get("tool") or "?"
        if tool_filter and tool != tool_filter:
            continue
        rows.append(
            {
                "time": entry.get("ts", ""),
                "tool": str(tool),
                "duration_ms": data.get("duration_ms", data.get("latency_ms")),
                "success": data.get("success", True),
                "result_size": data.get("result_size"),
            }
        )
    return rows


def _render_trace_rows(rows: list[dict[str, Any]]) -> None:
    print(_c(_TRACE_HEADER, _C_BOLD))
    print(_c("─" * 70, _C_DIM))
    for r in rows:
        ts = str(r["time"])[:16]
        tool = str(r["tool"])[:29]
        dur = f"{int(r['duration_ms']):>6}ms" if r["duration_ms"] is not None else "      ?"
        size = str(r["result_size"]) if r["result_size"] is not None else "-"
        ok = r.get("success", True)
        status_str = _c("OK   ", _C_GREEN) if ok else _c("ERROR", _C_RED)
        print(f"{ts:16}  {tool:29}  {dur}  {status_str}  {size}")


def _cmd_trace(args: argparse.Namespace) -> int:
    last_n = getattr(args, "last", 20) or 20
    tool_filter: Optional[str] = getattr(args, "tool", None)
    live = getattr(args, "live", False)

    if live:
        log_file = os.environ.get("MIND_MEM_LOG_FILE", "")
        if log_file and os.path.isfile(log_file):
            import subprocess  # nosec B404 — subprocess is used with a fixed argument list

            proc = subprocess.Popen(["tail", "-f", "-n", str(last_n), log_file], stdout=subprocess.PIPE, text=True)  # nosec B603 B607 — fixed arg list; log_file is from MIND_MEM_LOG_FILE env var (operator-controlled), validated isfile above; shell=False (default)
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    rows = _parse_log_lines([line], tool_filter=tool_filter)
                    if rows:
                        _render_trace_rows(rows)
                        sys.stdout.flush()
            except KeyboardInterrupt:
                proc.terminate()
        else:
            import select

            while True:
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if ready:
                        line = sys.stdin.readline()
                        if not line:
                            break
                        rows = _parse_log_lines([line], tool_filter=tool_filter)
                        if rows:
                            _render_trace_rows(rows)
                            sys.stdout.flush()
                except KeyboardInterrupt:
                    break
        return 0

    log_file = os.environ.get("MIND_MEM_LOG_FILE", "")
    if log_file and os.path.isfile(log_file):
        with open(log_file, encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    elif not sys.stdin.isatty():
        all_lines = sys.stdin.readlines()
    else:
        all_lines = []

    rows = _parse_log_lines(all_lines, tool_filter=tool_filter)
    rows = rows[-last_n:]
    if not rows:
        print(_c("No mcp_tool_call log entries found.", _C_DIM))
        return 0
    _render_trace_rows(rows)
    return 0


# ---------------------------------------------------------------------------
# skill subcommands
# ---------------------------------------------------------------------------


def _cmd_skill_list(args: argparse.Namespace) -> int:
    from mind_mem.skill_opt.adapters import discover_all
    from mind_mem.skill_opt.config import load_config

    cfg = load_config(_workspace())
    specs = discover_all(cfg.resolve_sources())
    rows = [s.as_dict() for s in specs]
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_skill_test(args: argparse.Namespace) -> int:
    import asyncio

    from mind_mem.skill_opt.adapters import discover_all
    from mind_mem.skill_opt.config import load_config
    from mind_mem.skill_opt.fleet_bridge import FleetBridge
    from mind_mem.skill_opt.test_runner import generate_test_cases, run_tests

    cfg = load_config(_workspace())
    specs = discover_all(cfg.resolve_sources())
    spec = next((s for s in specs if s.skill_id == args.skill_id), None)
    if spec is None:
        print(json.dumps({"error": f"Skill not found: {args.skill_id}"}))
        return 1
    fleet = FleetBridge(models=cfg.fleet_models.get("test_execution"))

    async def _run() -> list[dict[str, Any]]:
        cases = await generate_test_cases(spec, fleet, count=cfg.test_cases_per_skill)
        results = await run_tests(spec, cases, fleet)
        return [r.as_dict() for r in results]

    out = asyncio.run(_run())
    print(json.dumps(out, indent=2))
    return 0


def _cmd_skill_analyze(args: argparse.Namespace) -> int:
    import asyncio

    from mind_mem.skill_opt.adapters import discover_all
    from mind_mem.skill_opt.analyzer import aggregate_analysis, analyze_skill
    from mind_mem.skill_opt.config import load_config
    from mind_mem.skill_opt.fleet_bridge import FleetBridge
    from mind_mem.skill_opt.test_runner import generate_test_cases, run_tests

    cfg = load_config(_workspace())
    specs = discover_all(cfg.resolve_sources())
    spec = next((s for s in specs if s.skill_id == args.skill_id), None)
    if spec is None:
        print(json.dumps({"error": f"Skill not found: {args.skill_id}"}))
        return 1
    fleet = FleetBridge()

    async def _run() -> dict[str, Any]:
        cases = await generate_test_cases(spec, fleet, count=cfg.test_cases_per_skill)
        results = await run_tests(spec, cases, fleet)
        critiques = await analyze_skill(spec, results, fleet, min_critics=cfg.min_critics)
        return aggregate_analysis(critiques)

    out = asyncio.run(_run())
    print(json.dumps(out, indent=2))
    return 0


def _cmd_skill_optimize(args: argparse.Namespace) -> int:
    import asyncio
    import uuid

    from mind_mem.skill_opt.adapters import discover_all
    from mind_mem.skill_opt.analyzer import aggregate_analysis, analyze_skill
    from mind_mem.skill_opt.config import load_config
    from mind_mem.skill_opt.fleet_bridge import FleetBridge
    from mind_mem.skill_opt.history import HistoryStore
    from mind_mem.skill_opt.mutator import propose_mutations
    from mind_mem.skill_opt.test_runner import generate_test_cases, run_tests
    from mind_mem.skill_opt.validator import submit_to_governance, validate_mutation

    cfg = load_config(_workspace())
    specs = discover_all(cfg.resolve_sources())
    spec = next((s for s in specs if s.skill_id == args.skill_id), None)
    if spec is None:
        print(json.dumps({"error": f"Skill not found: {args.skill_id}"}))
        return 1
    fleet = FleetBridge()
    ws = cfg.governance_workspace or _workspace()
    db_path = os.path.join(ws, cfg.history_db_path)
    store = HistoryStore(db_path)
    run_id = f"R-{uuid.uuid4().hex[:12]}"

    async def _run() -> dict[str, Any]:
        store.start_run(run_id, spec.skill_id, spec.content_hash, cfg.as_dict())
        cases = await generate_test_cases(spec, fleet, count=cfg.test_cases_per_skill)
        results = await run_tests(spec, cases, fleet)
        critiques = await analyze_skill(spec, results, fleet, min_critics=cfg.min_critics)
        analysis = aggregate_analysis(critiques)
        mutations = await propose_mutations(spec, analysis, fleet, max_mutations=cfg.max_mutations_per_run)

        best_mutation = None
        best_validation = None
        for m in mutations:
            v = await validate_mutation(spec, m, cases, fleet, cfg)
            store.store_mutation(
                run_id,
                m.mutation_id,
                spec.skill_id,
                m.proposed_content,
                m.rationale,
                v.score_before,
                v.score_after,
            )
            if v.accepted and (best_validation is None or v.score_after > best_validation.score_after):
                best_mutation = m
                best_validation = v

        result: dict[str, Any] = {
            "run_id": run_id,
            "skill_id": spec.skill_id,
            "score_before": analysis.get("consensus_score", 0.0),
            "mutations_proposed": len(mutations),
            "best_accepted": None,
        }

        if best_mutation and best_validation:
            signal_id = submit_to_governance(best_mutation, best_validation, ws)
            store.update_mutation_status(best_mutation.mutation_id, "validated", signal_id)
            store.complete_run(run_id, "completed", best_validation.score_before, best_validation.score_after, True)
            result["best_accepted"] = {
                "mutation_id": best_mutation.mutation_id,
                "score_after": best_validation.score_after,
                "governance_signal": signal_id,
            }
        else:
            store.complete_run(run_id, "completed", analysis.get("consensus_score", 0.0), analysis.get("consensus_score", 0.0), False)
            result["best_accepted"] = None

        return result

    out = asyncio.run(_run())
    store.close()
    print(json.dumps(out, indent=2))
    return 0


def _cmd_skill_history(args: argparse.Namespace) -> int:
    from mind_mem.skill_opt.config import load_config
    from mind_mem.skill_opt.history import HistoryStore

    cfg = load_config(_workspace())
    ws = cfg.governance_workspace or _workspace()
    db_path = os.path.join(ws, cfg.history_db_path)
    store = HistoryStore(db_path)
    rows = store.get_run_history(args.skill_id, limit=args.limit)
    store.close()
    print(json.dumps(rows, indent=2, default=str))
    return 0


def _cmd_skill_score(args: argparse.Namespace) -> int:
    from mind_mem.skill_opt.config import load_config
    from mind_mem.skill_opt.history import HistoryStore

    cfg = load_config(_workspace())
    ws = cfg.governance_workspace or _workspace()
    db_path = os.path.join(ws, cfg.history_db_path)
    store = HistoryStore(db_path)
    score = store.get_latest_score(args.skill_id)
    store.close()
    print(json.dumps({"skill_id": args.skill_id, "latest_score": score}))
    return 0


# ---------------------------------------------------------------------------
# serve subcommand
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    from mind_mem.api.rest import run

    run(
        host=args.host,
        port=args.port,
        workspace=_workspace(),
        allow_unauthenticated_localhost=args.allow_unauthenticated_localhost,
    )
    return 0


def _cmd_http_serve(args: argparse.Namespace) -> int:
    """Launch the v3.9 stdlib HTTP transport. Blocks until SIGINT."""
    from mind_mem.http_transport import serve_http

    ws = _workspace()
    print(f"mind-mem http-serve: workspace={ws} bind={args.host}:{args.port}")
    thread, stop = serve_http(
        workspace=ws,
        host=args.host,
        port=args.port,
        allow_unauthenticated_localhost=args.allow_unauthenticated_localhost,
    )
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nshutting down ...")
        stop()
    return 0


def _cmd_daemon(args: argparse.Namespace) -> int:
    """Launch the v3.9 background daemon (auto-scheduled jobs). Blocks."""
    from mind_mem.daemon import run_daemon

    return run_daemon(_workspace(), dry_run=args.dry_run, once=args.once)


def _cmd_pipeline_status(args: argparse.Namespace) -> int:
    """Show the current pipeline hash + dirty-block count (v3.9)."""
    from mind_mem.pipeline_hash import current_pipeline_hash, pipeline_dirty_blocks

    ws = _workspace()
    digest, inputs = current_pipeline_hash(ws, return_inputs=True)  # type: ignore[misc]
    print(f"workspace            : {ws}")
    print(f"current pipeline hash: {digest}")
    print(f"  package_version    : {inputs.package_version}")
    print(f"  backend            : {inputs.backend}")
    print(f"  model              : {inputs.model}")
    print(f"  extractor sha256   : {inputs.extractor_source_sha256[:16]}...")
    print(f"  prompt template    : {inputs.prompt_template_sha256[:16] if inputs.prompt_template_sha256 else '(none)'}")

    if not args.list_dirty:
        return 0

    dirty = pipeline_dirty_blocks(ws)
    print(f"\ndirty blocks (transform_hash != current): {len(dirty)}")
    if args.json:
        import json as _json

        print(_json.dumps({"current_hash": digest, "dirty_blocks": dirty}, sort_keys=True))
        return 0
    for bid in dirty[:50]:
        print(f"  {bid}")
    if len(dirty) > 50:
        print(f"  ... and {len(dirty) - 50} more")
    return 0


def _cmd_inbox_watch(args: argparse.Namespace) -> int:
    """Watch an inbox directory and route files into the workspace (v3.9)."""
    import time as _time

    from mind_mem.inbox import InboxWatcher

    ws = _workspace()
    watcher = InboxWatcher(workspace=ws, inbox=args.directory, interval=args.interval)
    if args.once:
        results = watcher.process_once()
        ok = sum(1 for r in results if r.ok)
        bad = len(results) - ok
        print(f"mind-mem inbox-watch: processed {ok} ok, {bad} failed (of {len(results)})")
        for r in results:
            marker = "+" if r.ok else "!"
            extra = r.block_id if r.ok else (r.error or "")
            print(f"  [{marker}] {r.handler:6s} {r.path}  {extra}")
        return 0 if bad == 0 else 1

    print(f"mind-mem inbox-watch: workspace={ws} inbox={args.directory} interval={args.interval}s")
    watcher.start()
    try:
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        print("\nshutting down ...")
        watcher.stop()
    return 0


def _cmd_audit_model(args: argparse.Namespace) -> int:
    from mind_mem.model_audit import audit_model, format_report_text

    extra = tuple(args.allow_publisher) if args.allow_publisher else None
    try:
        report = audit_model(args.path, allow_extra_publishers=extra)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        out = report.to_dict()
        if not args.include_manifest:
            out.pop("manifest", None)
        print(json.dumps(out, indent=2))
    else:
        print(format_report_text(report, color=_use_color()))
        if args.include_manifest:
            print("\nmanifest (sha256):")
            for name, digest in sorted(report.manifest.items()):
                print(f"  {digest}  {name}")

    if args.manifest_out:
        out_path = os.path.expanduser(args.manifest_out)
        with open(out_path, "w") as f:
            for name, digest in sorted(report.manifest.items()):
                f.write(f"{digest}  {name}\n")
        print(f"\nmanifest written to {out_path}", file=sys.stderr)

    return 0 if report.passed else 1


def _read_keyfile(path: str, expected_len: int, kind: str) -> bytes:
    """Read a raw Ed25519 key file and validate its length."""
    p = Path(os.path.expanduser(path))
    if not p.is_file():
        raise FileNotFoundError(f"{kind} file not found: {p}")
    data = p.read_bytes()
    if len(data) != expected_len:
        raise ValueError(f"{kind} file must be {expected_len} raw bytes, got {len(data)} ({p})")
    return data


def _cmd_sign_model(args: argparse.Namespace) -> int:
    from mind_mem.model_signing import (
        ED25519_PRIVATE_KEY_BYTES,
        ED25519_PUBLIC_KEY_BYTES,
        generate_keypair,
        sign_model,
    )

    # Resolve the private key. Three modes:
    #   --key-file <path>            (raw 32-byte Ed25519 secret key)
    #   --generate-key <out-prefix>  (write <out-prefix>.sk + .pub, then sign)
    #   default                      (error — refuse to sign with an
    #                                 ephemeral key the operator can't
    #                                 verify against later)
    sk: bytes
    if args.generate_key:
        sk, pk = generate_keypair()
        sk_path = Path(os.path.expanduser(args.generate_key + ".sk"))
        pk_path = Path(os.path.expanduser(args.generate_key + ".pub"))
        sk_path.write_bytes(sk)
        try:
            os.chmod(sk_path, 0o600)
        except OSError:
            pass  # Windows / non-POSIX FS — best effort
        pk_path.write_bytes(pk)
        print(f"generated keypair: {sk_path} (private, 0600), {pk_path} (public)", file=sys.stderr)
    elif args.key_file:
        try:
            sk = _read_keyfile(args.key_file, ED25519_PRIVATE_KEY_BYTES, "private key")
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        print(
            "error: pass --key-file <sk> or --generate-key <prefix>; refusing to sign with an unrecorded ephemeral key.",
            file=sys.stderr,
        )
        return 2

    try:
        result = sign_model(args.path, sk, write_sidecars=not args.no_sidecars)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        out = {
            "manifest_sha256": result.manifest_sha256,
            "signature_hex": result.signature.hex(),
            "public_key_hex": result.public_key.hex(),
            "manifest_path": str(result.manifest_path) if result.manifest_path else None,
            "signature_path": str(result.signature_path) if result.signature_path else None,
            "pubkey_path": str(result.pubkey_path) if result.pubkey_path else None,
            "files_signed": len(result.manifest_text.splitlines()),
            "public_key_bytes": ED25519_PUBLIC_KEY_BYTES,
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"manifest sha256:  {result.manifest_sha256}")
        print(f"files signed:     {len(result.manifest_text.splitlines())}")
        print(f"signature:        {result.signature.hex()}")
        print(f"public key:       {result.public_key.hex()}")
        if result.manifest_path:
            print(f"wrote {result.manifest_path}")
            print(f"wrote {result.signature_path}")
            print(f"wrote {result.pubkey_path}")
    return 0


def _cmd_gate_check(args: argparse.Namespace) -> int:
    from mind_mem.model_gate import gate_check

    extra = tuple(args.allow_publisher) if args.allow_publisher else None
    decision = gate_check(
        args.path,
        trust_without_audit=args.trust_without_audit,
        allow_extra_publishers=extra,
    )
    out = {
        "passed": decision.passed,
        "reason": decision.reason,
        "path": decision.path,
        "manifest_sha256": decision.manifest_sha256,
        "audit_passed": decision.audit_passed,
        "audit_summary": decision.audit_summary,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        verdict = "ALLOW" if decision.passed else "BLOCK"
        print(f"{verdict}  reason={decision.reason}  path={decision.path}")
        print(f"        manifest_sha256={decision.manifest_sha256}")
        if decision.audit_summary:
            print(f"        audit_summary={decision.audit_summary}")
    return 0 if decision.passed else 1


def _cmd_gate_list(args: argparse.Namespace) -> int:
    from mind_mem.model_gate import gate_list

    rows = gate_list()
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        if not rows:
            print("(no entries — registry is empty)")
            return 0
        for row in rows:
            audit_passed = row.get("audit_passed")
            tag = "PASS " if audit_passed is True else "FAIL " if audit_passed is False else "??   "
            override = " [override]" if row.get("trust_without_audit") else ""
            print(f"{tag} {row['audited_at']}  {row['path']}{override}")
            summary = row.get("audit_report_summary", {})
            if summary.get("checks_failed"):
                print(f"       failed: {', '.join(summary['checks_failed'])}")
    return 0


def _cmd_gate_remove(args: argparse.Namespace) -> int:
    from mind_mem.model_gate import gate_remove

    removed = gate_remove(args.path)
    if args.json:
        print(json.dumps({"removed": removed, "path": args.path}, indent=2))
    else:
        print(f"{'removed' if removed else 'not present'}: {args.path}")
    return 0 if removed else 1


def _read_mic_input(path: str) -> tuple[bytes | str, str]:
    """Read a mic file; auto-detect mic@2 (text) vs mic-b (binary).
    Returns ``(payload, fmt)`` where ``fmt`` is ``"mic2"`` or ``"micb"``.
    """
    with open(os.path.expanduser(path), "rb") as f:
        raw = f.read()
    if raw.startswith(b"MICB"):
        return raw, "micb"
    # mic@2 starts with the literal header "mic@2"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"could not decode {path} as text or recognise as mic-b: {exc}") from exc
    if not text.lstrip().startswith("mic@2"):
        raise ValueError(f"{path} is not a recognised mic@2 / mic-b payload")
    return text, "mic2"


def _cmd_mic_convert(args: argparse.Namespace) -> int:
    """Convert a MIND IR graph between mic@2 and mic-b. Pure-Python,
    zero-dep — uses only :mod:`mind_mem.mic_map`."""
    from mind_mem.mic_map import (
        Mic2ParseError,
        MicbParseError,
        emit_mic2,
        emit_micb,
        parse_mic2,
        parse_micb,
    )

    try:
        payload, fmt = _read_mic_input(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if fmt == "mic2":
            assert isinstance(payload, str)
            graph = parse_mic2(payload)
        else:
            assert isinstance(payload, bytes)
            graph = parse_micb(payload)
    except (Mic2ParseError, MicbParseError) as exc:
        print(f"error: parse failed: {exc}", file=sys.stderr)
        return 1

    if args.mic_to == "mic2":
        out_text = emit_mic2(graph)
        if args.mic_out == "-":
            sys.stdout.write(out_text)
        else:
            with open(os.path.expanduser(args.mic_out), "w", encoding="utf-8") as f:
                f.write(out_text)
            print(f"wrote {len(out_text.encode('utf-8'))} bytes to {args.mic_out}", file=sys.stderr)
    else:
        out_bin = emit_micb(graph)
        if args.mic_out == "-":
            sys.stdout.buffer.write(out_bin)
        else:
            with open(os.path.expanduser(args.mic_out), "wb") as f:
                f.write(out_bin)
            print(f"wrote {len(out_bin)} bytes to {args.mic_out}", file=sys.stderr)
    return 0


def _cmd_mic_inspect(args: argparse.Namespace) -> int:
    """Print a structural summary of a MIC payload — type count, value
    count, output index, per-value tag (Arg/Param/Node + opcode)."""
    from mind_mem.mic_map import (
        Arg,
        Mic2ParseError,
        MicbParseError,
        Node,
        Param,
        parse_mic2,
        parse_micb,
    )

    try:
        payload, fmt = _read_mic_input(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if fmt == "mic2":
            assert isinstance(payload, str)
            graph = parse_mic2(payload)
        else:
            assert isinstance(payload, bytes)
            graph = parse_micb(payload)
    except (Mic2ParseError, MicbParseError) as exc:
        print(f"error: parse failed: {exc}", file=sys.stderr)
        return 1

    if args.mic_json:
        out: dict = {
            "format": fmt,
            "type_count": len(graph.types),
            "value_count": len(graph.values),
            "output_idx": graph.output,
            "types": [{"index": i, "dtype": t.dtype, "dims": list(t.dims)} for i, t in enumerate(graph.types)],
            "values": [],
        }
        for i, v in enumerate(graph.values):
            if isinstance(v, Arg):
                out["values"].append({"index": i, "kind": "arg", "name": v.name, "type_idx": v.type_idx})
            elif isinstance(v, Param):
                out["values"].append({"index": i, "kind": "param", "name": v.name, "type_idx": v.type_idx})
            elif isinstance(v, Node):
                out["values"].append({"index": i, "kind": "node", "opcode": v.opcode, "inputs": list(v.inputs)})
        print(json.dumps(out, indent=2))
    else:
        print(f"format:        {fmt}")
        print(f"types:         {len(graph.types)}")
        print(f"values:        {len(graph.values)}")
        print(f"output:        #{graph.output}")
        print()
        print("Types:")
        for i, t in enumerate(graph.types):
            print(f"  T{i}: {t.dtype}({', '.join(t.dims)})")
        print()
        print("Values:")
        for i, v in enumerate(graph.values):
            if isinstance(v, Arg):
                print(f"  #{i:3} arg     {v.name} : T{v.type_idx}")
            elif isinstance(v, Param):
                print(f"  #{i:3} param   {v.name} : T{v.type_idx}")
            elif isinstance(v, Node):
                inputs = ", ".join(f"#{j}" for j in v.inputs)
                print(f"  #{i:3} node    {v.opcode}({inputs})")
    return 0


def _cmd_audit_pinned(args: argparse.Namespace) -> int:
    """Run the seven-check audit (and optional Ed25519 verify) on every
    entry in ``audit_pinned_models`` of the chosen config. Designed for
    release CI — exits non-zero on any HIGH finding or verify failure.
    """
    from mind_mem.audit_pinned import (
        PinnedConfigError,
        audit_pinned,
        format_pinned_report_text,
    )

    try:
        report = audit_pinned(
            config_path=args.config,
            workspace=os.path.dirname(os.path.abspath(args.config)) or ".",
        )
    except PinnedConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_pinned_report_text(report))

    # Honour --fail-on-missing — by default a missing pinned path is a
    # warning so a freshly-cloned repo without checkpoints checked in
    # still passes.
    if args.fail_on_missing:
        for f in report.findings:
            if not f.exists:
                print(
                    f"error: pinned path missing and --fail-on-missing set: {f.path}",
                    file=sys.stderr,
                )
                return 2

    return 0 if report.passed else 1


def _cmd_verify_model(args: argparse.Namespace) -> int:
    from mind_mem.model_signing import ED25519_PUBLIC_KEY_BYTES, verify_model

    pk: bytes | None = None
    if args.pubkey:
        try:
            pk = _read_keyfile(args.pubkey, ED25519_PUBLIC_KEY_BYTES, "public key")
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    try:
        result = verify_model(args.path, public_key=pk)
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        out = {
            "passed": result.passed,
            "manifest_sha256": result.manifest_sha256,
            "error_kind": result.error_kind,
            "error_detail": result.error_detail,
        }
        print(json.dumps(out, indent=2))
    else:
        if result.passed:
            print(f"OK  manifest sha256: {result.manifest_sha256}")
        else:
            print(f"FAIL [{result.error_kind}] {result.error_detail}", file=sys.stderr)
    return 0 if result.passed else 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mm",
        description="Unified mind-mem CLI for non-MCP coding agents.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # recall
    p_recall = sub.add_parser("recall", help="Search memory and print JSON results.")
    p_recall.add_argument("query")
    p_recall.add_argument("--limit", type=int, default=10)
    p_recall.add_argument("--active-only", action="store_true")
    p_recall.set_defaults(func=_cmd_recall)

    # context
    p_ctx = sub.add_parser(
        "context",
        help="Recall + token-budget pack into a context snippet (JSON).",
    )
    p_ctx.add_argument("query")
    p_ctx.add_argument("--limit", type=int, default=20)
    p_ctx.add_argument("--max-tokens", type=int, default=2000)
    p_ctx.set_defaults(func=_cmd_context)

    # inject
    p_inj = sub.add_parser(
        "inject",
        help="Render a context snippet in the format a specific agent expects.",
    )
    p_inj.add_argument("query")
    p_inj.add_argument(
        "--agent",
        default="generic",
        help="Target agent (claude-code, codex, gemini, cursor, windsurf, aider, generic)",
    )
    p_inj.add_argument("--limit", type=int, default=10)
    p_inj.set_defaults(func=_cmd_inject)

    # status
    p_status = sub.add_parser("status", help="Print workspace status as JSON.")
    p_status.set_defaults(func=_cmd_status)

    # migrate-store — markdown <-> postgres backend migration
    p_migrate = sub.add_parser(
        "migrate-store",
        help="Migrate the workspace block corpus between storage backends.",
    )
    p_migrate.add_argument("--from", dest="from_backend", choices=["markdown"], required=True)
    p_migrate.add_argument("--to", dest="to_backend", choices=["postgres"], required=True)
    p_migrate.add_argument(
        "--dsn",
        help='Postgres DSN, e.g. "postgresql://mindmem:***@127.0.0.1:5432/mindmem".',
    )
    p_migrate.add_argument(
        "--schema",
        default="mind_mem",
        help='Postgres schema name (default: "mind_mem").',
    )
    p_migrate.add_argument("--dry-run", action="store_true", dest="dry_run")
    p_migrate.add_argument("--execute", action="store_true")
    p_migrate.add_argument(
        "--with-embeddings",
        action="store_true",
        dest="with_embeddings",
        help="Backfill embedding vectors after the row insert (requires pgvector + Ollama).",
    )
    p_migrate.add_argument(
        "--embed-model",
        default="mxbai-embed-large",
        dest="embed_model",
        help="Ollama embedding model name (default: mxbai-embed-large, dim 1024).",
    )
    p_migrate.set_defaults(func=_cmd_migrate_store)

    # detect — list AI coding clients present on the machine
    p_detect = sub.add_parser(
        "detect",
        help="Auto-detect installed AI coding clients for this workspace.",
    )
    p_detect.set_defaults(func=_cmd_detect)

    # install — configure a single agent
    p_install = sub.add_parser(
        "install",
        help="Configure mind-mem for one named AI coding client.",
    )
    p_install.add_argument(
        "agent",
        help=(
            "Agent key: claude-code, codex, gemini, cursor, windsurf, "
            "aider, openclaw, nanoclaw, nemoclaw, continue, cline, roo, "
            "zed, copilot, cody, qodo."
        ),
    )
    p_install.add_argument("--dry-run", action="store_true")
    p_install.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config instead of non-destructive merge.",
    )
    p_install.set_defaults(func=_cmd_install)

    # install-all — auto-detect + configure everything
    p_install_all = sub.add_parser(
        "install-all",
        help="Auto-detect + configure every installed AI coding client.",
    )
    p_install_all.add_argument(
        "--agent",
        action="append",
        help=("Restrict installation to these named agents. Repeat flag for multiple. Default = auto-detect every installed client."),
    )
    p_install_all.add_argument("--dry-run", action="store_true")
    p_install_all.add_argument("--force", action="store_true")
    p_install_all.add_argument(
        "--no-mcp",
        action="store_true",
        help=(
            "Skip native MCP server registration. Default: write both "
            "the text hook AND the MCP registration for every agent "
            "that supports MCP (Codex, Gemini, Cursor, Windsurf, "
            "Continue, Cline, Roo, Zed)."
        ),
    )
    p_install_all.set_defaults(func=_cmd_install_all)

    # install-model — pull mind-mem-4b GGUF from HF + import to Ollama
    p_install_model = sub.add_parser(
        "install-model",
        help=(
            "Download `mind-mem-4b` GGUF (~2.5GB) from HuggingFace and "
            "import into Ollama as `mind-mem:4b`. Idempotent."
        ),
    )
    p_install_model.add_argument(
        "--model",
        default="mind-mem-4b-Q4_K_M.gguf",
        help="GGUF filename on HF. Default: mind-mem-4b-Q4_K_M.gguf",
    )
    p_install_model.add_argument(
        "--name",
        default="mind-mem:4b",
        help="Ollama tag to register. Default: mind-mem:4b",
    )
    p_install_model.add_argument(
        "--dest",
        default=os.path.expanduser("~/.cache/mind-mem/mind-mem-4b-Q4_K_M.gguf"),
        help="Local path to download into. Default: ~/.cache/mind-mem/",
    )
    p_install_model.add_argument(
        "--keep-alive",
        default="-1",
        help="Ollama keep-alive value. -1 = forever (default), 30m, etc.",
    )
    p_install_model.add_argument("--dry-run", action="store_true")
    p_install_model.set_defaults(func=_cmd_install_model)

    # vault namespace
    p_vault = sub.add_parser("vault", help="Vault sync subcommands.")
    vsub = p_vault.add_subparsers(dest="vault_cmd", required=True)

    v_scan = vsub.add_parser("scan", help="Walk a vault and print parsed blocks.")
    v_scan.add_argument("vault_root")
    v_scan.add_argument(
        "--sync-dirs",
        nargs="*",
        help="Restrict scan to these subdirectories (relative).",
    )
    v_scan.set_defaults(func=_cmd_vault_scan)

    v_write = vsub.add_parser("write", help="Write a block to a vault.")
    v_write.add_argument("vault_root")
    v_write.add_argument("relative_path")
    v_write.add_argument("--id", required=True)
    v_write.add_argument("--type", default="note")
    v_write.add_argument("--title", default="")
    v_write.add_argument("--body", default="")
    v_write.add_argument("--overwrite", action="store_true")
    v_write.set_defaults(func=_cmd_vault_write)

    # skill namespace
    p_skill = sub.add_parser("skill", help="Self-improving skill optimization subcommands.")
    ssub = p_skill.add_subparsers(dest="skill_cmd", required=True)

    s_list = ssub.add_parser("list", help="List all discovered skills across systems.")
    s_list.set_defaults(func=_cmd_skill_list)

    s_test = ssub.add_parser("test", help="Generate + run synthetic tests for a skill.")
    s_test.add_argument("skill_id", help="Skill ID (e.g. openclaw:coding-agent, claude:code-reviewer)")
    s_test.set_defaults(func=_cmd_skill_test)

    s_analyze = ssub.add_parser("analyze", help="Cross-model critique of a skill.")
    s_analyze.add_argument("skill_id")
    s_analyze.set_defaults(func=_cmd_skill_analyze)

    s_optimize = ssub.add_parser("optimize", help="Full optimization loop: test → analyze → mutate → validate.")
    s_optimize.add_argument("skill_id")
    s_optimize.set_defaults(func=_cmd_skill_optimize)

    s_history = ssub.add_parser("history", help="Show optimization run history for a skill.")
    s_history.add_argument("skill_id")
    s_history.add_argument("--limit", type=int, default=10)
    s_history.set_defaults(func=_cmd_skill_history)

    s_score = ssub.add_parser("score", help="Show current score for a skill.")
    s_score.add_argument("skill_id")
    s_score.set_defaults(func=_cmd_skill_score)

    # serve — launch the REST API
    p_serve = sub.add_parser(
        "serve",
        help="Launch the mind-mem REST API server (requires mind-mem[api]).",
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8080, help="TCP port (default: 8080)")
    p_serve.add_argument(
        "--allow-unauthenticated-localhost",
        action="store_true",
        help=(
            "v3.7.0 H4: explicit operator opt-in to start the REST API without "
            "authentication. Requires a loopback bind (127.0.0.1 / localhost / ::1)."
        ),
    )
    p_serve.set_defaults(func=_cmd_serve)

    # http-serve — stdlib-only HTTP transport (v3.9 candidate)
    p_http = sub.add_parser(
        "http-serve",
        help="Launch the v3.9 stdlib HTTP transport (zero dependencies; minimal endpoint surface).",
    )
    p_http.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_http.add_argument("--port", type=int, default=8765, help="TCP port (default: 8765)")
    p_http.add_argument(
        "--allow-unauthenticated-localhost",
        action="store_true",
        help=(
            "Bypass token auth when the bind host is loopback. Required for "
            "non-token operators; refuses to start without it when no MIND_MEM_TOKEN is set."
        ),
    )
    p_http.set_defaults(func=_cmd_http_serve)

    # daemon — v3.9 background scheduler (set-and-forget mode)
    p_daemon = sub.add_parser(
        "daemon",
        help="Launch the v3.9 background daemon — runs configured jobs on internal intervals.",
    )
    p_daemon.add_argument("--dry-run", action="store_true", help="Log what would run, do not execute.")
    p_daemon.add_argument("--once", action="store_true", help="Run every enabled task once and exit.")
    p_daemon.set_defaults(func=_cmd_daemon)

    # inbox-watch — v3.9 inbox folder ingestion
    p_inbox = sub.add_parser(
        "inbox-watch",
        help="Watch an inbox directory; route files by extension into the workspace.",
    )
    p_inbox.add_argument("directory", help="Path to the inbox directory (created if absent).")
    p_inbox.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds (>=0.5).")
    p_inbox.add_argument("--once", action="store_true", help="Drain inbox once and exit.")
    p_inbox.set_defaults(func=_cmd_inbox_watch)

    # pipeline-status — v3.9 hash-of-code invalidation inspection
    p_pipeline = sub.add_parser(
        "pipeline-status",
        help="Show current extractor pipeline hash + count of dirty (re-extract) blocks.",
    )
    p_pipeline.add_argument("--list-dirty", action="store_true", help="List the block ids whose transform_hash is stale.")
    p_pipeline.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p_pipeline.set_defaults(func=_cmd_pipeline_status)

    # audit-model — static security scan of any local model checkpoint
    p_audit = sub.add_parser(
        "audit-model",
        help=(
            "Static security scan of a local model checkpoint. Flags remote-code "
            "hooks, unsafe pickle, tokenizer injection. Outputs SHA-256 manifest."
        ),
    )
    p_audit.add_argument(
        "path",
        help="Path to a local model directory (HF checkpoint layout).",
    )
    p_audit.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text report.")
    p_audit.add_argument(
        "--include-manifest",
        action="store_true",
        help="Include SHA-256 per-file manifest in output.",
    )
    p_audit.add_argument(
        "--manifest-out",
        default="",
        help="Write SHA-256 manifest to this path (shasum-compatible format).",
    )
    p_audit.add_argument(
        "--allow-publisher",
        action="append",
        default=[],
        metavar="HF_ORG_SLUG",
        help=(
            "Augment the default provenance allowlist with this HF org slug "
            "(repeatable). Use for internal fine-tune orgs that aren't in "
            "the canonical publisher list."
        ),
    )
    p_audit.set_defaults(func=_cmd_audit_model)

    # sign-model — Ed25519 manifest signing on a local checkpoint
    p_sign = sub.add_parser(
        "sign-model",
        help=("Sign every file in a local model checkpoint with Ed25519. Writes MODEL_MANIFEST.txt + .sig + MODEL_PUBKEY.pub sidecars."),
    )
    p_sign.add_argument("path", help="Path to a local model directory.")
    sign_key_group = p_sign.add_mutually_exclusive_group()
    sign_key_group.add_argument(
        "--key-file",
        default="",
        help="Path to a raw 32-byte Ed25519 private key file.",
    )
    sign_key_group.add_argument(
        "--generate-key",
        default="",
        metavar="PREFIX",
        help="Generate a new keypair, write PREFIX.sk (0600) + PREFIX.pub, then sign.",
    )
    p_sign.add_argument(
        "--no-sidecars",
        action="store_true",
        help="Do not write MANIFEST/.sig/.pub files; print result only.",
    )
    p_sign.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable lines.",
    )
    p_sign.set_defaults(func=_cmd_sign_model)

    # verify-model — Ed25519 verification of a previously-signed checkpoint
    p_verify = sub.add_parser(
        "verify-model",
        help=("Verify a previously-signed checkpoint. Returns nonzero if the manifest, signature, or public key fail."),
    )
    p_verify.add_argument("path", help="Path to a signed model directory.")
    p_verify.add_argument(
        "--pubkey",
        default="",
        help=("Path to a raw 32-byte Ed25519 public key. If omitted, the MODEL_PUBKEY.pub sidecar in the directory is used."),
    )
    p_verify.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable lines.",
    )
    p_verify.set_defaults(func=_cmd_verify_model)

    # gate — load-gate registry: check / list / remove
    p_gate = sub.add_parser(
        "gate",
        help=("Load-gate registry that tracks which local checkpoints have been audited. Three sub-commands: check, list, remove."),
    )
    gsub = p_gate.add_subparsers(dest="gate_cmd", required=True)

    g_check = gsub.add_parser(
        "check",
        help=(
            "Run gate_check on PATH. Audits the checkpoint if not seen "
            "before; re-audits on file drift; refuses to load a path that "
            "fails any check."
        ),
    )
    g_check.add_argument("path")
    g_check.add_argument(
        "--trust-without-audit",
        action="store_true",
        help=(
            "Force-load even if the audit fails (or hasn't been run). "
            "Recorded in the registry as an explicit override so the "
            "decision is auditable."
        ),
    )
    g_check.add_argument(
        "--allow-publisher",
        action="append",
        default=[],
        metavar="HF_ORG_SLUG",
        help="Augment the provenance allowlist (repeatable).",
    )
    g_check.add_argument("--json", action="store_true")
    g_check.set_defaults(func=_cmd_gate_check)

    g_list = gsub.add_parser("list", help="Print the registry contents.")
    g_list.add_argument("--json", action="store_true")
    g_list.set_defaults(func=_cmd_gate_list)

    g_remove = gsub.add_parser("remove", help="Remove a path from the registry.")
    g_remove.add_argument("path")
    g_remove.add_argument("--json", action="store_true")
    g_remove.set_defaults(func=_cmd_gate_remove)

    # audit-pinned — release-CI gate that audits every pinned model
    p_pinned = sub.add_parser(
        "audit-pinned",
        help=(
            "Run the seven-check audit (and optional Ed25519 verify) on every "
            "entry in audit_pinned_models of mind-mem.json. Designed for release "
            "CI — non-zero exit on any HIGH finding or verify failure."
        ),
    )
    p_pinned.add_argument(
        "--config",
        default="mind-mem.json",
        help="Path to mind-mem.json (default: ./mind-mem.json).",
    )
    p_pinned.add_argument(
        "--fail-on-missing",
        action="store_true",
        help=("Treat a missing pinned path as a hard failure (exit 2). By default, missing paths are skipped with a SKIP marker."),
    )
    p_pinned.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable lines.",
    )
    p_pinned.set_defaults(func=_cmd_audit_pinned)

    # ── mic — MIND IR graph serialization (mic@2 / mic-b) ─────────────────
    p_mic = sub.add_parser(
        "mic",
        help=("MIND IR graph serialization (mic@2 text + mic-b binary). Subcommands: convert, inspect."),
    )
    mic_sub = p_mic.add_subparsers(dest="mic_action", required=True)

    p_mic_convert = mic_sub.add_parser(
        "convert",
        help=("Convert a graph between mic@2 (text) and mic-b (binary). Round-trips byte-for-byte."),
    )
    p_mic_convert.add_argument("path", help="Input file (auto-detect format).")
    p_mic_convert.add_argument(
        "--to",
        choices=["mic2", "micb"],
        required=True,
        dest="mic_to",
        help="Output format.",
    )
    p_mic_convert.add_argument(
        "-o",
        "--output",
        default="-",
        dest="mic_out",
        help="Output file (default: stdout). Binary written raw, text as UTF-8.",
    )
    p_mic_convert.set_defaults(func=_cmd_mic_convert)

    p_mic_inspect = mic_sub.add_parser(
        "inspect",
        help=("Print a structural summary of a MIC payload: type/value count, output index, per-value tag (Arg/Param/Node + opcode)."),
    )
    p_mic_inspect.add_argument("path", help="Input file (auto-detect format).")
    p_mic_inspect.add_argument(
        "--json",
        action="store_true",
        dest="mic_json",
        help="Emit JSON instead of human-readable lines.",
    )
    p_mic_inspect.set_defaults(func=_cmd_mic_inspect)

    # ── inspect — full block fields + provenance tree ──────────────────────
    p_inspect = sub.add_parser(
        "inspect",
        help="Print full block fields and provenance tree for a block ID.",
    )
    p_inspect.add_argument("block_id", help="Block ID to inspect (e.g. D-001).")
    p_inspect.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="format",
        help="Output format: human-readable text (default) or structured JSON.",
    )
    p_inspect.set_defaults(func=_cmd_inspect)

    # ── explain — retrieval trace for a query ─────────────────────────────
    p_explain = sub.add_parser(
        "explain",
        help="Show per-stage retrieval scores (BM25 → vector → RRF → rerank) for a query.",
    )
    p_explain.add_argument("query", help="Search query to trace.")
    p_explain.add_argument("--limit", type=int, default=10, help="Number of results to show (default: 10).")
    p_explain.add_argument(
        "--backend",
        choices=["auto", "bm25", "hybrid"],
        default="auto",
        help="Retrieval backend: auto (default), bm25, or hybrid.",
    )
    p_explain.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="format",
        help="Output format: human-readable text (default) or structured JSON.",
    )
    p_explain.set_defaults(func=_cmd_explain)

    # ── trace — MCP call log tail ─────────────────────────────────────────
    p_trace = sub.add_parser(
        "trace",
        help="Display recent MCP tool calls parsed from structured JSON logs.",
    )
    p_trace.add_argument(
        "--live",
        action="store_true",
        help=("Stream new events in real time. Reads MIND_MEM_LOG_FILE if set, otherwise reads stdin."),
    )
    p_trace.add_argument(
        "--last",
        type=int,
        default=20,
        metavar="N",
        help="Show the last N MCP calls (default: 20).",
    )
    p_trace.add_argument(
        "--tool",
        default=None,
        metavar="TOOL_NAME",
        help="Filter output to a single tool name.",
    )
    p_trace.set_defaults(func=_cmd_trace)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
