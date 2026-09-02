"""Opt-in pytest plugin: find which test/product code leaks live sqlite3 handles.

Not auto-loaded. `conftest.py` is the name pytest collects automatically; this
file is deliberately named otherwise so it costs nothing on a normal run, and
is enabled explicitly for one diagnostic run:

    python3 -m pytest tests/ -p conftest_trace

It wraps ``sqlite3.connect`` to record the opening stack per connection, then at
session end walks the GC for ``sqlite3.Connection`` objects that are still alive
and reports the most common opening frames for index databases. That report is
what turns "the suite leaks descriptors" into a specific call site.
"""

import collections
import gc
import sqlite3
import traceback

_real = sqlite3.connect
_origin = {}


def _traced(*a, **kw):
    conn = _real(*a, **kw)
    try:
        _origin[id(conn)] = (
            str(a[0]) if a else "?",
            "".join(traceback.format_stack(limit=12)[:-1]),
        )
    except Exception:
        # A diagnostic must never change the outcome of the run it observes:
        # if bookkeeping fails, the connection is still returned untouched and
        # this handle is simply absent from the report.
        pass
    return conn


sqlite3.connect = _traced


def pytest_sessionfinish(session, exitstatus):
    live = [o for o in gc.get_objects() if isinstance(o, sqlite3.Connection)]
    c = collections.Counter()
    n = 0
    for o in live:
        info = _origin.get(id(o))
        if not info:
            continue
        path, stack = info
        if "index.db" not in path:
            continue
        n += 1
        for line in stack.splitlines():
            if "/tests/" in line or "mind_mem/" in line:
                c[line.strip()] += 1
    print(f"\nTRACE2 live_index_db={n}", flush=True)
    for k, v in c.most_common(10):
        print(f"   {v:>4}  {k}", flush=True)
