#!/usr/bin/env python3
"""mind-mem Cron Runner — single entry point for all periodic jobs. Zero external deps.

Reads workspace config (mind-mem.json) for feature toggles, then dispatches
to the appropriate script(s) via subprocess.

Jobs:
  transcript_scan  — scan recent transcripts for signals
  entity_ingest    — extract entities from signals/logs
  intel_scan       — contradiction detection, drift analysis, briefings
  all              — run all enabled jobs sequentially

Usage:
    python3 scripts/cron_runner.py /path/to/workspace --job all
    python3 scripts/cron_runner.py /path/to/workspace --job transcript_scan
    python3 scripts/cron_runner.py --install-cron
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from observability import get_logger, metrics

_log = get_logger("cron_runner")

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Job definitions: name -> (script, args_fn, config_toggle)
# ---------------------------------------------------------------------------

JOB_DEFS: dict[str, tuple[str, list[str], str]] = {
    "transcript_scan": (
        "transcript_capture.py",
        ["--scan-recent", "--days", "1"],
        "transcript_scan",
    ),
    "entity_ingest": (
        "entity_ingest.py",
        [],
        "entity_ingest",
    ),
    "intel_scan": (
        "intel_scan.py",
        [],
        "intel_scan",
    ),
}

ALL_JOBS = list(JOB_DEFS.keys())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(workspace: str) -> dict:
    """Load mind-mem.json from workspace. Returns defaults if missing/unreadable."""
    config_path = os.path.join(workspace, "mind-mem.json")
    try:
        with open(config_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        _log.warning("config_load_failed", path=config_path, error=str(exc))
        return {}


def is_job_enabled(config: dict, job_name: str) -> bool:
    """Check if a job is enabled in the auto_ingest config section."""
    auto_ingest = config.get("auto_ingest", {})
    # If auto_ingest.enabled is explicitly false, nothing runs
    if not auto_ingest.get("enabled", True):
        return False
    # Check individual toggle (default: enabled)
    return auto_ingest.get(job_name, True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_job(job_name: str, workspace: str) -> dict:
    """Run a single job. Returns result dict with status and details."""
    script, extra_args, _ = JOB_DEFS[job_name]
    script_path = os.path.join(SCRIPTS_DIR, script)

    if not os.path.isfile(script_path):
        _log.error("script_not_found", job=job_name, path=script_path)
        return {"job": job_name, "status": "error", "error": f"script not found: {script_path}"}

    cmd = [sys.executable, script_path, workspace] + extra_args
    _log.info("job_start", job=job_name, cmd=" ".join(cmd))

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            timeout=120,
            capture_output=True,
            text=True,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        metrics.observe(f"job_{job_name}_ms", elapsed_ms)

        if result.returncode == 0:
            metrics.inc(f"job_{job_name}_ok")
            _log.info("job_complete", job=job_name, returncode=0, duration_ms=round(elapsed_ms, 1))
            return {"job": job_name, "status": "ok", "duration_ms": round(elapsed_ms, 1)}
        else:
            metrics.inc(f"job_{job_name}_fail")
            stderr_tail = (result.stderr or "")[-500:]
            _log.error("job_failed", job=job_name, returncode=result.returncode, stderr=stderr_tail)
            return {
                "job": job_name,
                "status": "failed",
                "returncode": result.returncode,
                "stderr": stderr_tail,
            }

    except subprocess.TimeoutExpired:
        elapsed_ms = (time.monotonic() - start) * 1000
        metrics.inc(f"job_{job_name}_timeout")
        _log.error("job_timeout", job=job_name, timeout_s=120)
        return {"job": job_name, "status": "timeout"}

    except Exception as exc:
        metrics.inc(f"job_{job_name}_error")
        _log.error("job_exception", job=job_name, error=str(exc))
        return {"job": job_name, "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Crontab installer
# ---------------------------------------------------------------------------

def print_cron_instructions(workspace: str) -> None:
    """Print crontab entries for all jobs."""
    script_path = os.path.abspath(__file__)
    ws = os.path.abspath(workspace)
    print("# mind-mem auto-ingestion (add to crontab with: crontab -e)")
    print(f"0 */6 * * * python3 {script_path} {ws} --job transcript_scan 2>&1 | logger -t mind-mem")
    print(f"0 3 * * * python3 {script_path} {ws} --job entity_ingest 2>&1 | logger -t mind-mem")
    print(f"0 3 * * * python3 {script_path} {ws} --job intel_scan 2>&1 | logger -t mind-mem")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="mind-mem periodic job runner")
    parser.add_argument("workspace", nargs="?", default=".", help="workspace path (default: cwd)")
    parser.add_argument("--job", choices=ALL_JOBS + ["all"], default="all", help="which job to run")
    parser.add_argument("--install-cron", action="store_true", help="print crontab install instructions")
    args = parser.parse_args()

    workspace = os.path.abspath(args.workspace)

    if args.install_cron:
        print_cron_instructions(workspace)
        return 0

    config = load_config(workspace)
    jobs = ALL_JOBS if args.job == "all" else [args.job]

    _log.info("cron_run_start", workspace=workspace, jobs=jobs)

    results = []
    for job_name in jobs:
        if not is_job_enabled(config, job_name):
            _log.info("job_skipped", job=job_name, reason="disabled in config")
            results.append({"job": job_name, "status": "skipped"})
            continue
        results.append(run_job(job_name, workspace))

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] in ("failed", "error", "timeout"))
    skipped = sum(1 for r in results if r["status"] == "skipped")

    _log.info("cron_run_complete", total=len(results), ok=ok, failed=failed, skipped=skipped)

    print(f"\nmind-mem cron: {ok} ok, {failed} failed, {skipped} skipped (of {len(results)} jobs)")
    for r in results:
        marker = {"ok": "+", "skipped": "-", "failed": "!", "error": "!", "timeout": "!"}
        print(f"  [{marker.get(r['status'], '?')}] {r['job']}: {r['status']}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
