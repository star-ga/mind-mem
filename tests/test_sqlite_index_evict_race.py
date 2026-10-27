"""Evicting a cached ConnectionManager must not close it under a running query.

Closing a sqlite3 handle while a statement runs on it is a use-after-free in
the C layer: it segfaults the interpreter. Each scenario runs in a subprocess
so a crash is observable as a return code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import mind_mem.sqlite_index as si

SRC = str(Path(si.__file__).resolve().parent.parent)

_PRELUDE = textwrap.dedent(
    """
    import json, os, sys, threading, time, sqlite3
    import mind_mem.sqlite_index as si
    root = sys.argv[1]
    def mk(name):
        ws = os.path.join(root, name)
        for d in ["decisions","tasks","entities","intelligence","memory"]:
            os.makedirs(os.path.join(ws, d), exist_ok=True)
        with open(os.path.join(ws,"decisions","DECISIONS.md"),"w") as f:
            row = "[D-20260101-{i:03d}]\\nStatement: Use PostgreSQL database number {i}\\nStatus: active\\nDate: 2026-01-01\\n\\n"
            f.write("".join(row.format(i=i) for i in range(1,60)))
        for fn in ["entities/people.md","entities/tools.md","entities/incidents.md","entities/projects.md",
                   "intelligence/CONTRADICTIONS.md","intelligence/DRIFT.md","intelligence/SIGNALS.md","tasks/TASKS.md"]:
            p = os.path.join(ws, fn); os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p,"w").write("# x\\n")
        si.build_index(ws, incremental=False)
        return ws
    A = mk("A")
    si._CONN_MANAGER_CACHE_MAX = 1
    errors = []; stop = time.time() + 4.0
    def evictor():
        i = 0
        while time.time() < stop:
            i += 1
            try:
                os.makedirs(os.path.join(root, f"o{i%50}"), exist_ok=True)
                si._get_conn_manager(os.path.join(root, f"o{i%50}"))
                si._get_conn_manager(A)
            except Exception as e:
                errors.append(repr(e)); return
    def run(readers, target):
        ts = [threading.Thread(target=target) for _ in range(readers)] + [threading.Thread(target=evictor)]
        [t.start() for t in ts]; [t.join() for t in ts]
        print(json.dumps({"errors": errors[:3], "n": len(errors)}))
    """
)

_QUERY_INDEX = _PRELUDE + textwrap.dedent(
    """
    def reader():
        while time.time() < stop:
            try:
                si.query_index(A, "PostgreSQL database", limit=5)
            except Exception as e:
                errors.append(repr(e)); return
    run(3, reader)
    """
)

_RAW_LEASED = _PRELUDE + textwrap.dedent(
    """
    def reader():
        while time.time() < stop:
            try:
                with si.leased_conn_manager(A) as mgr:
                    c = mgr.get_read_connection()
                    c.execute("SELECT count(*) FROM blocks").fetchall()
                    for _ in c.execute("SELECT * FROM blocks"):
                        pass
            except Exception as e:
                errors.append(repr(e)); return
    run(3, reader)
    """
)


def _run(script: str, tmp_path):
    env = {**os.environ, "PYTHONPATH": SRC}
    return subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=120,
    )


@pytest.mark.parametrize("script", [_QUERY_INDEX, _RAW_LEASED], ids=["query_index", "leased_raw_reader"])
def test_evicting_manager_mid_query_is_safe(script, tmp_path):
    for attempt in range(3):
        proc = _run(script, tmp_path / f"r{attempt}")
        assert proc.returncode == 0, f"returncode {proc.returncode}\n{proc.stderr[-1500:]}"
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert out["n"] == 0, out
