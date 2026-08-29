#!/usr/bin/env python3
"""With-memory versus without-memory, on this repository's own history.

The benchmark the product thesis rests on: *does governed memory make a
coding agent better?*  Retrieval quality is measured elsewhere in this
directory; this is the join nobody had measured -- whether the memory
actually **helped**.

Run the positive control first (it proves the grader can see a pass, which
is what licenses reading a null result), then the comparison::

    python3 benchmarks/memory_ab_bench.py selfcheck --select bucket:single_file:1
    python3 benchmarks/memory_ab_bench.py run --select bucket:single_file:1 --agent none
    python3 benchmarks/memory_ab_bench.py run --select bucket:single_file:1 \\
        --agent command --agent-env AGENT_API_KEY -- /path/to/agent-cli --flag

Identical to ``mind-mem-bench-ab``; this file exists so the benchmark is
reachable from ``benchmarks/`` the way the rest of the suite is.  All the
machinery lives in :mod:`mind_mem.bench.ab_harness` and is importable,
tested and typed -- nothing that decides a number lives in this script.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mind_mem.bench.ab_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
