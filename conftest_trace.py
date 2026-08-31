import sqlite3, traceback, collections, gc
_real = sqlite3.connect
_origin = {}
def _traced(*a, **kw):
    conn = _real(*a, **kw)
    try: _origin[id(conn)] = (str(a[0]) if a else "?", "".join(traceback.format_stack(limit=12)[:-1]))
    except Exception: pass
    return conn
sqlite3.connect = _traced
def pytest_sessionfinish(session, exitstatus):
    live = [o for o in gc.get_objects() if isinstance(o, sqlite3.Connection)]
    c = collections.Counter(); n = 0
    for o in live:
        info = _origin.get(id(o))
        if not info: continue
        path, stack = info
        if "index.db" not in path: continue
        n += 1
        for line in stack.splitlines():
            if "/tests/" in line or "mind_mem/" in line: c[line.strip()] += 1
    print(f"\nTRACE2 live_index_db={n}", flush=True)
    for k, v in c.most_common(10): print(f"   {v:>4}  {k}", flush=True)
