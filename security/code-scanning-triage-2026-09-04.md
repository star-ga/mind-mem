# Code-scanning triage — the 17 alerts blocking 5.0.2

Alerts as recorded at commit `6cd37e5f`. Line numbers are given twice where
they moved: **`@6cd37e5f`** is the line the alert points at, **`@now`** is the
line in the tree after the fixes below.

Nothing here has been dismissed on GitHub. Dismissal is an outward-facing act
and the operator makes it; this file is the evidence for that decision.

**Result: 1 true positive (fixed), 16 false positives (rationale below).**
The true positive is *not* one of the flows CodeQL reported — it was found by
auditing the `apply_engine` cluster rather than accepting it.

---

## Summary

| # | Rule | Location `@6cd37e5f` | Verdict | Action |
|---|------|----------------------|---------|--------|
| 243 | py/path-injection | apply_engine.py:521 | FALSE POSITIVE | rationale §1 |
| 244 | py/path-injection | apply_engine.py:540 | FALSE POSITIVE | rationale §1 |
| 245 | py/path-injection | apply_engine.py:561 | FALSE POSITIVE (reported flow) | rationale §1 — **but see §2: the same line held a real, unreported defect, now fixed** |
| 246 | py/path-injection | apply_engine.py:2047 | FALSE POSITIVE | rationale §1 |
| 247 | py/path-injection | block_parser.py:574 | FALSE POSITIVE | rationale §1 |
| 248 | py/path-injection | block_store.py:1257 | FALSE POSITIVE | rationale §1 |
| 249 | py/path-injection | block_store.py:1290 | FALSE POSITIVE | rationale §1 |
| 254 | py/path-injection | block_store.py:1304 | FALSE POSITIVE | rationale §1 |
| 241 | py/clear-text-storage-sensitive-data | apply_engine.py:2151 | FALSE POSITIVE | rationale §3 |
| 242 | py/clear-text-storage-sensitive-data | apply_engine.py:2153 | FALSE POSITIVE | rationale §3 |
| 250 | py/tarslip | scripts/alignment_authorities.py:273 | FALSE POSITIVE (re-verified) | rationale §4 |
| 255 | py/insecure-protocol | tests/test_network_tls_floor.py:267 | FALSE POSITIVE (re-verified) | rationale §5 |
| 256 | py/insecure-protocol | tests/test_network_tls_floor.py:318 | FALSE POSITIVE (re-verified) | rationale §5 |
| 251 | B105 hardcoded-password-string | compliance/detectors.py:74 | FALSE POSITIVE | rationale §6 |
| 252 | B105 hardcoded-password-string | dream_cycle.py:1007 | FALSE POSITIVE | rationale §6 |
| 253 | B105 hardcoded-password-string | http_transport.py:749 | FALSE POSITIVE | rationale §6 |
| 257 | B603 subprocess-without-shell | mcp/tools/arch_mind.py:165 | FALSE POSITIVE | rationale §7 |

No alert in this set is a credential. §6 was checked value-by-value for that
specifically.

---

## §1 — The eight `py/path-injection` alerts are one flow, and two guards stop it

### They are not eight findings

Read out of the analysis SARIF, all eight alerts share an identical prefix.
There is exactly one taint source between them:

```
0: src/mind_mem/api/rest.py:1117          body           (POST /v1/rollback_proposal)
1: src/mind_mem/api/rest.py:1123          body.receipt_ts
2: src/mind_mem/mcp/tools/governance.py:1172   receipt_ts (def rollback_proposal)
3: src/mind_mem/mcp/tools/governance.py:1209   receipt_ts (-> engine_rollback)
4: src/mind_mem/apply_engine.py:2056      receipt_ts     (def rollback)
```

Every one of the eight then fans out from `apply_engine.rollback`. So the
question is not eight questions. It is one: **can a caller of
`/v1/rollback_proposal` put a path separator into `receipt_ts`?**

No. Two independent guards stop it, and the SARIF flow paths record neither.

### Guard A — the format gate (`apply_engine.py:2077 @6cd37e5f` / `:2101 @now`)

```python
if not re.match(r"^\d{8}-\d{6}\Z", receipt_ts):
    print(f"ERROR: Invalid receipt timestamp format: {receipt_ts} (expected YYYYMMDD-HHMMSS)")
    return False
```

The admitted alphabet is `[0-9]` and one `-`. It contains no `/`, no `\`, no
`.`, and no NUL. **A traversal component cannot be spelled in it at all.** The
gate returns `False` before `_safe_resolve` is reached on line 2106 (`@now`),
so no other value ever reaches the filesystem.

The same gate is applied a second time, independently, one layer out at
`mcp/tools/governance.py:1186`, which is the only door the REST route uses.

*(This gate was anchored with `\Z` as part of this triage. It previously used
`$`, which in Python also matches immediately before a trailing newline, so
`"20260101-000000\n"` was accepted. No traversal followed — the class admits no
separator — but this gate is the stated reason the whole cluster is safe, so it
now accepts exactly what it claims. Regression test:
`tests/test_restore_record_manifest_containment.py::TestReceiptTimestampGate`.)*

### Guard B — containment, applied twice

`apply_engine.py:250-261 @now` (`249-260 @6cd37e5f`):

```python
def _safe_resolve(ws, rel_path):
    ws_real = os.path.realpath(ws)
    joined = os.path.join(ws, rel_path)
    resolved = os.path.realpath(joined)
    if not resolved.startswith(ws_real + os.sep) and resolved != ws_real:
        raise ValueError(f"Path escapes workspace: {rel_path}")
    return resolved
```

`block_store.py:34-52` is the same shape for the snapshot-relative side
(`_safe_child_path`). Both call `os.path.realpath` **before** the containment
test, so a symlink escape is resolved and caught, not just a literal `..`. Both
raise rather than return a value on failure, so there is no "guard returned the
unsafe path anyway" branch.

Alerts 248, 249 and 254 sit directly on the outputs of `_safe_child_path` —
`block_store.py:1247`, `:1277`, `:1299` respectively — and their SARIF paths
confirm it (each flow's last three steps are `_safe_child_path()` -> `src` ->
the sink). Routing verified, not asserted: every path-touching line in
`BlockStore.restore` is one of `1196/1197`, `1247/1248`, `1277/1278`,
`1299/1300`, and all eight are `_safe_child_path` calls.

### The blind spot, shown from the scanner's own output

This is the checkable part. Take the SARIF for any of the eight and look at the
steps through `_safe_resolve`:

```
 6: apply_engine.py:249   rel_path        <- enters the helper
 7: apply_engine.py:256   joined          <- os.path.join
 8: apply_engine.py:257   joined
 9: apply_engine.py:257   Attribute()     <- os.path.realpath
10: apply_engine.py:257   resolved
11: apply_engine.py:260   resolved        <- the `return`
12: apply_engine.py:2082  _safe_resolve() <- back at the call site, still tainted
```

CodeQL **does** follow the value into the helper and back out. Step 11 is the
`return` statement on line 260. The guard is on lines 258-259 — *between step 10
and step 11* — and the flow path does not record it. Nor does any step land on
line 2077, which is Guard A.

The reason is CodeQL's sanitiser model for this query. `PathSanitizer.qll`
implements containment checks as **barrier guards**, which block flow only on
the branch of a check that **dominates the sink node being scored**. Here the
check dominates a `return` inside a helper; every sink is in a *different*
function, reached through that return's value. No barrier dominates the sink, so
the query never applies one. Guard A is missed for a plainer reason: it is a
`re.match` on a *format*, and this query has no notion that an alphabet without
separators cannot express a path.

**How a reviewer checks this without reading the code:** fetch the analysis
SARIF and assert that the `codeFlows` for alerts 243-249 and 254 contain no
location at `apply_engine.py:2077` and none at `apply_engine.py:258-259` or
`block_store.py:50-51`. If those locations are absent from the path, the scanner
did not see the guards.

```
gh api repos/star-ga/mind-mem/code-scanning/analyses/<id> \
   -H "Accept: application/sarif+json"
```

### What is NOT claimed here

Only that this *reported flow* is unreachable. It is not a claim that the
snapshot machinery is free of traversal — see §2, which is the opposite.

---

## §2 — TRUE POSITIVE (unreported by any scanner): the restore record trusted the manifest

**`apply_engine._block_ids_in_snapshot`, `:560 @6cd37e5f` / `:572 @now`. Fixed.**

`restore_snapshot` builds its evidence record from the snapshot's
`MANIFEST.json` file list. Two consumers read that same list:

* `BlockStore.restore` (`block_store.py:1190-1200`) routes **every** entry
  through `_safe_child_path(snap_dir, rel_path)` and `_safe_child_path(ws,
  rel_path)`, logging `restore_unsafe_manifest_entry` and skipping any that
  escape.
* `apply_engine._block_ids_in_snapshot` did **not**. It joined each entry onto
  the snapshot directory bare —

  ```python
  path = os.path.join(snap_dir, rel.replace("/", os.sep))   # no containment check
  ```

  — then `os.path.isfile(path)` and `parse_file(path)`.

`MANIFEST.json` is file content, not a validated argument, so the entries are
attacker-influenced in any scenario where a snapshot directory is not trusted.

### Why this is worse than an out-of-tree read

The parsed ids do not merely appear in the record. `restore_snapshot:686` reads:

```python
withdrawn = sorted(set(_live_block_ids(ws)) - set(reinstated))
```

`reinstated` is exactly what `_block_ids_in_snapshot` returned. So a manifest
entry pointing at a `.md` file **outside the snapshot** whose block ids collide
with live workspace ids **subtracts those ids from `withdrawn`** — and
`withdrawn` is the list the code itself documents (lines 682-685) as the half
the record has to spell out, "because a withdrawn block is recoverable from
nothing once the restore lands." A crafted manifest could therefore suppress
what a restore destroyed, from the governance record of that restore. `block_ids`
(line 687) feeds `admit_batch(block_ids=...)`, so the admission entry inherits it.

CodeQL did not report this. Its flow for alert 245 goes through the `os.walk`
branch (`_manifest_files:525`), which is inherently contained; it has no taint
source for JSON file content, so the dangerous branch (`_manifest_files:519`,
`manifest_data.get("files", [])`) was never explored.

### The fix

Route it through the helper that already exists, rather than adding a second
bespoke check:

```python
try:
    path = _safe_child_path(snap_dir, rel.replace("/", os.sep))
except ValueError as exc:
    _log.warning("restore_record_unsafe_manifest_entry", snap_dir=snap_dir, entry=rel, reason=str(exc))
    continue
```

`_safe_child_path` is imported from `block_store` alongside the `_read_manifest`
this module already shares with it, so both consumers of the manifest now apply
one guard from one definition.

### Capability lost: none

An entry that escapes the snapshot root is already refused by
`BlockStore.restore:1196-1200`, so it is never copied back. Counting its ids as
`reinstated` was therefore a record of something that did not happen. Skipping
it makes the record *more* accurate, not narrower. Escaping entries are logged
individually (`restore_record_unsafe_manifest_entry`) rather than dropped
silently, matching the existing `restore_unsafe_manifest_entry` convention on
the restore side.

### Proof

`tests/test_restore_record_manifest_containment.py` — 7 tests covering `../`
traversal, an absolute manifest entry (where `os.path.join` discards the root),
a symlink inside the snapshot pointing out of it, and agreement with the guard
`BlockStore.restore` itself applies.

Positive controls, because a "not in results" assertion proves nothing on its
own: each test first asserts the foreign file **exists**, **parses to the id
being searched for**, and **is reachable by the bare join the fix removed** —
so the fixture is proven to exercise the traversal before the absence is
asserted. A third control asserts the crafted entry survives `_manifest_files`
into the function under test, so no green can come from an empty list.

Mutation control (the test must be able to fail): restoring the bare
`os.path.join` turns all 4 containment tests red; restoring `$` for `\Z` turns
the newline test red. Both were run, then reverted and confirmed byte-identical
with `cmp -s`.

### Related, not fixed, reported instead

`apply_engine.generate_diff_text:1471` joins `files_touched` entries onto
`snap_dir` with a partial normalisation (`:1468-1470`) and no rejection. Its
input is internally generated from the proposal's ops rather than from file
content, and it is not one of the 17 alerts, so it is named here rather than
changed under a security triage.

---

## §3 — `py/clear-text-storage-sensitive-data` (#241, #242): the "secret" is a test fixture's name

**Sinks:** `apply_engine.py:2151` and `:2153` (`@6cd37e5f`) — the two `f.write`
calls that append the rollback's `Reason:` lines to `APPLY_RECEIPT.md`.

**The source, from the SARIF:**

```
0: tests/test_event_fanout_wiring.py:573   Fstring
1: tests/test_event_fanout_wiring.py:573   secret_reason
2: tests/test_event_fanout_wiring.py:574   secret_reason  -> governance.rollback_proposal(reason=...)
```

`tests/test_event_fanout_wiring.py:573` is:

```python
secret_reason = f"reverting because {CANARY} was wrong"
```

`py/clear-text-storage-sensitive-data` classifies its sources by **identifier
name**. A local variable in a test called `secret_reason` matches the query's
sensitive-name heuristic, and the taint is carried from there into product code.
The scanner is reporting the name of a test fixture, not a property of the data.

**What actually gets written.** The `reason` parameter of
`rollback_proposal` — the operator's written rationale for the rollback,
mandatory and at least 8 non-whitespace characters (`governance.py:1189`). It is
passed through `_sanitize_reason_for_markdown` (`apply_engine.py:2149`) before
the write. Recording it is the purpose of the field: it exists so that
"a rejection rationale three months ago shows up next to the receipt, not in chat
scrollback" (`governance.py:1178-1180`).

**Plaintext is the product, and this is documented.** The corpus is Markdown by
design; that is what makes it auditable and diffable. Writing a governance
rationale into the receipt in the clear is the specified behaviour of the audit
chain, not an incident.

**Positive evidence that no real secret rides this path.** The very test CodeQL
draws the source from asserts the opposite — `test_event_fanout_wiring.py:582`:

```python
assert CANARY not in json.dumps(payload, default=str), "the rollback reason leaked into the payload"
```

The suite plants a canary in the reason *specifically to prove it does not
escape into the event payload*, and only `reason_length` is fanned out
(`:581`). The alert is triggered by the canary that exists to catch leaks.

---

## §4 — `py/tarslip` (#250): re-verified, the filter is present and correct

**Adjudicated previously by the architecture seat. Re-checked, and confirmed.**

`scripts/alignment_authorities.py:273`:

```python
with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r|") as tar:
    tar.extractall(tmp, members=_safe_members(tar, Path(tmp)))  # nosec B202
```

The `members=` generator, `_safe_members` at `:208-230`, refuses on **four**
grounds and raises rather than skipping:

```python
if member.issym() or member.islnk():                    # :220  links
    raise AuthorityError(...)
if not (member.isfile() or member.isdir()):             # :222  devices, fifos
    raise AuthorityError(...)
target = Path(member.name)
if target.is_absolute() or target.drive or target.root: # :225  absolute / drive-relative
    raise AuthorityError(...)
resolved = (root / target).resolve()
if resolved != root and root not in resolved.parents:   # :227-228  `..` escape, post-resolve
    raise AuthorityError(...)
```

`root` is `dest.resolve()` (`:218`), so the containment test runs on fully
resolved paths. The check covers every extraction, on every supported Python —
which is the point of the comment at `:262-270`: the previous form passed
`filter="data"` on 3.12+ and nothing on 3.10/3.11, resting on an argument about
the *caller* rather than a property of the *code*.

**Scanner blind spot.** CodeQL's `py/tarslip` sanitiser recognises two shapes:
the `filter=` keyword on `extractall`, and a per-member check written *inline in
a loop* before an `extract` call. It does not model a **generator passed as
`members=`**: the filtering happens lazily, inside a different function, driven
by `extractall`'s own iteration. There is no guard at the call site for the query
to find, so it reports the `extractall` as unfiltered.

**Checkable:** if `_safe_members` were removed or its raises turned into
`continue`, §4's claim would change; it has not. `mode="r|"` is a stream, so
members are consumed in order and the generator sees each one before extraction
— the lazy filter is applied to every member, not a prefix.

---

## §5 — `py/insecure-protocol` (#255, #256): the test is the assertion that the floor refuses this

**Adjudicated previously by the architecture seat. Re-checked, and confirmed.**

Both alerts are in `tests/test_network_tls_floor.py`, in
`test_tls12_only_peer_cannot_be_reached_and_never_sees_a_request`.

* **#255 @:267** — `self.socket = ctx.wrap_socket(...)`, where the context in
  question is built at `:309-311`:

  ```python
  legacy = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
  legacy.load_cert_chain(str(cert), str(key))
  legacy.maximum_version = ssl.TLSVersion.TLSv1_2
  ```

* **#256 @:318** — `floorless.wrap_socket(raw, ...)`, where `floorless` is a
  plain `ssl.create_default_context()` at `:316`.

Both are **adversary fixtures**, not product configuration. The test's structure
is a negative assertion with its own positive control:

1. Stand up a listener that maxes out at TLS 1.2 (`:309-313`).
2. **Prove the listener works** by completing a handshake with a floorless
   stdlib client — `assert tls.version() == "TLSv1.2"` (`:316-319`). Without
   this, step 3 would be satisfied by a broken fixture.
3. Point the product's client at it and assert it **refuses**, and that no
   request reached the peer (`:321-325`).

The floor being tested is product code, and it is not weakened by any of this:
`src/mind_mem/v4/tls_floor.py:46` sets `TLS_FLOOR = ssl.TLSVersion.TLSv1_3`,
assigns it at `:85`, and **reads it back** at `:88-90`, raising
`TlsFloorUnavailable` if the assignment did not take. `:114` is the predicate
that asserts a context carries the floor.

**Scanner blind spot.** `py/insecure-protocol` flags a context whose allowed
version range includes TLS 1.0/1.1. It has no concept of a test's *intent*: a
deliberately sub-floor peer, constructed so that a conforming client can be
proven to reject it, is byte-for-byte the same API call as a misconfigured
server. Suppressing the fixture would delete the negative control — the test
would then assert only that the client refuses *something*, with nothing proving
the something was reachable.

**Do not "fix" these.** Raising the fixture's version floor makes
`test_tls12_only_peer_cannot_be_reached_and_never_sees_a_request` vacuous while
leaving it green.

---

## §6 — B105 (#251, #252, #253): three names, zero credentials

Bandit's `hardcoded_password_string` fires on a string literal whose
**surrounding identifier or dict key** matches a wordlist (`pass`, `secret`,
`token`, ...). It does not inspect the value. Each was checked as a value:

| # | Location | Literal | What it is |
|---|----------|---------|------------|
| 251 | `compliance/detectors.py:74` | `CATEGORY_SECRET = "secret"` | One of exactly two **category labels** a detector may claim (`_CATEGORIES = frozenset({CATEGORY_PII, CATEGORY_SECRET})`, `:75`). The docstring at `:69-72` states it is "a routing hint for an operator... never an authorisation decision." Flagged because the *constant name* contains `SECRET`. |
| 252 | `dream_cycle.py:1007` | `metadata={"pass": "auto_repair", ...}` | The name of a dream-cycle **pass** (as in a pass over the data), recorded in the admission metadata. Flagged because the *dict key* is `"pass"`. |
| 253 | `http_transport.py:749` | `HTTP_TOKEN_ACTOR_PREFIX = "http:tok:"` | A 9-character **actor-id prefix** for the audit record. The comment at `:745-748` is explicit that the suffix is `sha256(token)[:12]` — "the credential's identity without the credential — an auditor can group every act under one token, and rotating that token does not rewrite what it already did." Flagged because the *constant name* contains `TOKEN`. |

**None of the three is a credential**, and #253 is the site that specifically
implements *not storing* one. There is nothing here to rotate.

---

## §7 — B603 (#257): list argv, no shell, every element validated

`src/mind_mem/mcp/tools/arch_mind.py:165`:

```python
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, encoding="utf-8", errors="replace")
```

with `cmd = [binary, *args]` (`:163`).

* **No shell.** `shell` is not passed, so it defaults to `False`. The argv is a
  list, so no element is ever parsed for metacharacters — `;`, `|`, `$(...)`,
  backticks and globs are inert.
* **The program is not caller-controlled.** `binary` comes from
  `_resolve_binary()` (`:140-147`): the `ARCH_MIND_BIN` environment variable, or
  `shutil.which("arch-mind")`. Neither is reachable from an MCP argument. An
  actor who can set the server's environment already controls the process.
* **Every argument is validated before it reaches the list.** There is no path
  into `_run` that skips this — all seven call sites (`:204, :225, :245, :273,
  :294, :319, :347`) validate first:
  * `_validate_arch_path` (`:76-103`) — str-typed, non-empty, length-bounded,
    **rejects NUL** (defeats C-string truncation), and **rejects a leading `-`**:
    an explicit flag-injection guard, `"path may not start with '-'"`.
  * `_validate_arch_id` (`:106-117`) — allowlist `^[A-Za-z0-9_.\-]{1,128}$`, so
    no whitespace, quotes, or flag prefixes.
  * `_validate_arch_mode` (`:120-123`) and `_validate_arch_metric` (`:126-128`)
    — closed allowlists.
  * `_validate_arch_days` (`:132-137`) — int-typed (with an explicit `bool`
    rejection), range-bounded.
* **Bounded.** `timeout=60.0` default, `TimeoutExpired` raised as
  `ArchMindError`.

The worst a hostile caller achieves is passing a *valid* path or id to
`arch-mind` — which is the tool's purpose. B603 is a categorical "you called
subprocess" note; it fires on every `subprocess.run` that is not a literal
constant list and carries no claim that a specific injection exists.

---

## Verification for this triage

* `python3.12 -m ruff check` — all checks passed (whole repo)
* `python3.12 -m ruff format --check` — clean
* `python3.12 -m mypy src/ --ignore-missing-imports` — no issues, 345 files
* `pytest` over the 86 test files touching rollback / restore / apply_engine /
  MANIFEST — **1935 passed, 34 skipped, 0 failed**
* Mutation controls run and reverted (`cmp -s` byte-identical) for both fixes
