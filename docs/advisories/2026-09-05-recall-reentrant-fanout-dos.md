# Advisory (DRAFT — not published): unbounded thread fan-out in auto-enabled recall

- **Status:** DRAFT. Not filed as a GHSA. Publication is an operator decision.
- **Affected:** `mind-mem` 3.3.0 through 5.0.1 inclusive
- **Fixed in:** 5.0.2 (commit `bff3913`)
- **Impact:** remotely-triggerable resource exhaustion (thread exhaustion) on the
  default recall path
- **Requires:** the ability to influence a recall *query string*. No credential,
  no configuration change, and no non-default flag.

## What happens

`recall` can auto-enable query expansion and query decomposition when it
classifies a query as `multi-hop` or `temporal`. Both features fan out: they
turn one query into several and search each in a thread pool.

The fan-out is designed to be depth-1, and `_skip_auto_features` exists to make
the nested searches plain. Both auto-enable branches read only their own active
flag and never consulted it:

```python
if not expansion_active and self._query_expansion_config.get("auto_enable", True):
    if _qt() in ("multi-hop", "temporal"):
        expansion_active = True     # re-enabled regardless of the guard
```

Variant 0 of every expansion is the original query. So a query that classifies
as temporal or multi-hop expands into a list *containing itself*, at every
level — and each level holds a pool thread blocked in `Future.result` on the
level beneath it. Recursion is bounded only by the point at which the host
refuses to create another thread. One process was measured at **30,935 OS
threads**; a box under test reached 76,242 and could no longer fork.

## Why it shipped unnoticed for two minor lines

The terminating `RuntimeError: can't start new thread` was caught and logged
through the same warning path as an ordinary "expansion found nothing" miss. A
system at the edge of fork exhaustion and a system with an unhelpful query
produced the same log line at the same level.

## Reachability

Reached from `_recall_impl_uncached` under the default `backend="auto"` — so the
MCP `recall` tool and the HTTP transport are on the affected path with stock
configuration. Any deployment where a query string can be influenced by content
the operator does not control (an agent recalling against attacker-supplied
text, a shared workspace, a chat surface) is exposed.

## Fix

One clause on each branch, restoring the guard that was already designed in:

```python
if not _skip_auto_features and not expansion_active and ...:
```

Every pool is kept. The off-path served ranking is byte-identical; the on-path
one becomes the documented depth-1 RRF fusion rather than a function of where
the box ran out of threads. Thread-pool exhaustion now carries `failure_kind`,
logs at error level, gets its own counter, and degrades to the single-query
answer instead of being indistinguishable from a miss.

## For operators who cannot upgrade immediately

Set `auto_enable: false` for both `query_expansion` and `query_decomposition`.
That disables the auto-enable branches entirely; explicitly-enabled expansion
is not affected, because the explicit path always honoured the guard.

## Credit

Found in internal review while auditing the recall path, not by an external
reporter. No evidence of exploitation; the measurement above came from our own
test box, whose fork exhaustion is what prompted the audit.
