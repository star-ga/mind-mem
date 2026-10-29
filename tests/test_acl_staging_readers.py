"""A user-scope staging write is allowed only while no reader serves it to others.

``USER_STAGING_WRITES`` lets four user-scope tools append to their own store.
That is only safe while nothing reads the store back into what OTHER clients
are served. ``recall_outcome`` broke the rule: with the supported opt-in
``recall.validity_gate.outcome_attribution`` on, every client's recall reads it
and reweights its served scores, so one user-scope ``report_outcome`` could
demote a block for everyone -- and the gate checked only v4 flags.

The fix derives the gate from the READERS of each staged store
(``STAGING_STORE_READERS``). This module holds that table to the code:

* an AST scan finds every function that touches a staged store, directly or
  through another function that does, and fails on any it finds unclassified --
  so a new reader (and the config key that turns it on) cannot be forgotten;
* a live reader that serves other clients makes the write admin-scope, proved
  end to end with the positive control (admin writes) and the mutation (reader
  off -> the user write is staging again).
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from typing import Any, Iterator

import pytest

from mind_mem.mcp.infra.acl import (
    ADMIN_TOOLS,
    STAGING_STORE_READERS,
    USER_STAGING_WRITES,
    check_write_mode_acl,
    cross_client_reader_config_keys,
    staging_store_token,
)
from mind_mem.mcp.infra.workspace import use_workspace

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
#: The classifier itself names every store; scanning it would be circular.
_EXCLUDED_MODULES = frozenset({"mind_mem.mcp.infra.acl"})


# ---------------------------------------------------------------------------
# The scan: every function that touches a staged store, transitively
# ---------------------------------------------------------------------------


class _Source:
    """Parsed ``src/mind_mem`` with just enough name resolution for the scan."""

    def __init__(self, extra: dict[str, str] | None = None) -> None:
        self.text: dict[str, str] = {}
        # Per module: its UTF-8 bytes and the byte offset of each line start,
        # computed once so a node's source is one slice (see ``segment``).
        self._utf8: dict[str, bytes] = {}
        self._line_starts: dict[str, list[int]] = {}
        self.funcs: dict[str, ast.AST] = {}  # "module.Qual.name" -> node
        self.module_of: dict[str, str] = {}
        self.class_methods: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.by_leaf: dict[str, set[str]] = defaultdict(set)
        # Module- and class-level assignments: a store named only in a constant
        # (a schema, a query string, a file name) is reached through the name.
        self.consts: dict[str, ast.AST] = {}
        self.const_module: dict[str, str] = {}
        modules: list[tuple[str, str, bool]] = []
        for path in sorted(SRC.rglob("*.py")):
            parts = path.relative_to(SRC).with_suffix("").parts
            mod = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            if mod not in _EXCLUDED_MODULES:
                modules.append((mod, path.read_text(encoding="utf-8"), path.name == "__init__.py"))
        for mod, text in (extra or {}).items():
            modules = [m for m in modules if m[0] != mod] + [(mod, text, False)]
        for mod, text, is_pkg in modules:
            self.text[mod] = text
            data = text.encode("utf-8")
            self._utf8[mod] = data
            # The parser's line breaks: \r\n, \r, \n (never \f or other
            # str.splitlines() separators), the same split ast uses.
            self._line_starts[mod] = [0] + [m.end() for m in re.finditer(rb"\r\n|\r|\n", data)]
            tree = ast.parse(text)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._add(mod, node.name, node)
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self._add(mod, f"{node.name}.{item.name}", item)
                            self.class_methods[(mod, node.name)].add(item.name)
                        else:
                            self._add_const(mod, f"{node.name}.", item)
                else:
                    self._add_const(mod, "", node)
            self._imports(mod, tree, is_pkg)

    def segment(self, mod: str, node: ast.AST) -> str:
        """``ast.get_source_segment(self.text[mod], node) or ""``, by position.

        The stdlib helper re-splits the whole module into lines on every call
        (a per-character loop before Python 3.12), so calling it once per node
        cost O(nodes x module size) per scan and timed out the slowest CI row.
        Positions are UTF-8 byte columns, so this slices the encoded module."""
        end_lineno = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if end_lineno is None or end_col is None:
            return ""
        starts = self._line_starts[mod]
        begin = starts[node.lineno - 1] + node.col_offset  # type: ignore[attr-defined]
        return self._utf8[mod][begin : starts[end_lineno - 1] + end_col].decode("utf-8")

    def _add(self, mod: str, qual: str, node: ast.AST) -> None:
        key = f"{mod}.{qual}"
        self.funcs[key] = node
        self.module_of[key] = mod
        self.by_leaf[qual.rsplit(".", 1)[-1]].add(key)

    def _add_const(self, mod: str, prefix: str, node: ast.AST) -> None:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            targets = [node.target]
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    key = f"{mod}.{prefix}{name.id}"
                    self.consts[key] = node
                    self.const_module[key] = mod

    def _imports(self, mod: str, tree: ast.AST, is_pkg: bool) -> None:
        table: dict[str, tuple[str, str | None]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    pkg = mod.split(".") if is_pkg else mod.split(".")[:-1]
                    pkg = pkg[: len(pkg) - (node.level - 1)]
                    base = ".".join(pkg + ([base] if base else []))
                for alias in node.names:
                    table[alias.asname or alias.name] = (base, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    table[alias.asname or alias.name.split(".")[0]] = (alias.name if alias.asname else alias.name.split(".")[0], None)
        self._import_table = getattr(self, "_import_table", {})
        self._import_table[mod] = table

    def _resolve_name(self, mod: str, name: str) -> str | None:
        """``name`` as seen in *mod* -> "module.Qual" (a function or a class)."""
        imported = self._import_table[mod].get(name)
        if imported and imported[1]:
            return f"{imported[0]}.{imported[1]}"
        return f"{mod}.{name}"

    def references(self, key: str) -> set[str]:
        """Function keys *key* may call: names resolved through imports, methods
        through ``Class(...)``/``var = Class(...)``/``self``, else a leaf name
        when every definition of it is a candidate (decided by the caller)."""
        if key in self.consts:
            mod = self.const_module[key]
            return self._refs(mod, self.consts[key], key[len(mod) + 1 :].split(".")[0] if "." in key[len(mod) + 1 :] else None)
        mod = self.module_of[key]
        return self._refs(mod, self.funcs[key], key[len(mod) + 1 :].split(".")[0] if "." in key[len(mod) + 1 :] else None)

    def _refs(self, mod: str, node: ast.AST, owner: str | None) -> set[str]:
        local_cls: dict[str, str] = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call) and isinstance(sub.value.func, ast.Name):
                for target in sub.targets:
                    if isinstance(target, ast.Name):
                        local_cls[target.id] = self._resolve_name(mod, sub.value.func.id) or ""
        out: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                out.add(self._resolve_name(mod, sub.id) or "")
            elif isinstance(sub, ast.Attribute):
                value = sub.value
                cls: str | None = None
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                    cls = self._resolve_name(mod, value.func.id)
                elif isinstance(value, ast.Name) and value.id == "self" and owner:
                    cls = f"{mod}.{owner}"
                elif isinstance(value, ast.Name) and value.id in local_cls:
                    cls = local_cls[value.id]
                elif isinstance(value, ast.Name) and self._import_table[mod].get(value.id, ("", "x"))[1] is None:
                    out.add(f"{self._import_table[mod][value.id][0]}.{sub.attr}")
                    continue
                elif isinstance(value, ast.Name) and f"{self._resolve_name(mod, value.id)}" in self.text:
                    # ``from pkg import module`` then ``module.NAME``
                    out.add(f"{self._resolve_name(mod, value.id)}.{sub.attr}")
                    continue
                if cls:
                    out.add(f"{cls}.{sub.attr}")
                else:
                    out.add(f"*.{sub.attr}")
        return out

    def touching(self, token: str, terminals: frozenset[str]) -> set[str]:
        """Every function reaching *token*, stopping at classified terminals."""
        pattern = re.compile(rf"\b{re.escape(token)}\b")

        def names(k: str, n: ast.AST, mod: str) -> bool:
            return bool(pattern.search(self.segment(mod, n)))

        # Constants first, to a fixed point: one naming the store, or built
        # from another that does, is the store as far as a reader is concerned.
        hits = {k for k, n in self.consts.items() if names(k, n, self.const_module[k])}
        const_refs = {k: self.references(k) for k in self.consts}
        while True:
            grown = {k for k, r in const_refs.items() if k not in hits and r & hits}
            if not grown:
                break
            hits |= grown
        by_leaf: dict[str, set[str]] = defaultdict(set)
        for k in self.consts:
            by_leaf[k.rsplit(".", 1)[-1]].add(k)
        # ``obj.NAME`` the scan cannot type: matched by leaf when every
        # constant of that name is a hit (the same rule methods get below).
        hits |= {f"*.{leaf}" for leaf, keys in by_leaf.items() if keys <= hits}
        refs = {k: self.references(k) for k in self.funcs}
        found = {k for k, n in self.funcs.items() if names(k, n, self.module_of[k]) or refs[k] & hits}
        while True:
            live = {k for k in found if k not in terminals}
            names = set(live)
            for k in live:
                leaf = k.rsplit(".", 1)[-1]
                if self.by_leaf[leaf] <= found:
                    names.add(f"*.{leaf}")
            grown = {k for k, r in refs.items() if k not in found and r & names}
            if not grown:
                return found
            found |= grown


@pytest.fixture(scope="module")
def source() -> _Source:
    return _Source()


def _terminals(token: str) -> frozenset[str]:
    return frozenset(k for k, r in STAGING_STORE_READERS[token].items() if r.role != "plumbing")


#: Every store whose readers are held to the code: each one a user-scope tool
#: stages into, plus each one taken OUT of user scope because a reader serves
#: it to other clients (the table is the evidence for that decision).
_SCANNED_TOKENS = sorted({staging_store_token(s) for v in USER_STAGING_WRITES.values() for s in v} | set(STAGING_STORE_READERS))


@pytest.mark.parametrize("token", _SCANNED_TOKENS)
def test_every_function_touching_a_staged_store_is_classified(token: str, source: _Source) -> None:
    """A new reader of a staged store fails here until it is classified.

    Classifying a reader that serves other clients means naming the config keys
    that turn it on, and those keys are what the gate probes and what the
    all-readers-on template enables -- so neither can drift from the code.
    """
    assert token in STAGING_STORE_READERS, f"staged store {token!r} has no reader table"
    touching = source.touching(token, _terminals(token))
    unclassified = sorted(touching - set(STAGING_STORE_READERS[token]))
    assert unclassified == [], f"functions touching {token!r} with no reader classification: {unclassified}"
    stale = sorted(set(STAGING_STORE_READERS[token]) - set(source.funcs))
    assert stale == [], f"classified functions that no longer exist: {stale}"


#: The blind spot the previous scan had: the store named only in a
#: module-level constant (calibration.py's schema is written this way).
_CONST_READER = (
    "\n_OUTCOME_SQL = \"SELECT block_id, COUNT(*) FROM recall_outcome WHERE outcome='failure' GROUP BY block_id\"\n"
    "def _sneaky_outcome_penalty(conn):\n    return dict(conn.execute(_OUTCOME_SQL).fetchall())\n"
)


def _with(source: _Source, mod: str, tail: str) -> _Source:
    return _Source({mod: source.text[mod] + tail})


def test_the_scan_sees_a_store_named_through_a_module_constant(source: _Source) -> None:
    mutated = _with(source, "mind_mem.validity_gate", _CONST_READER)
    touching = mutated.touching("recall_outcome", _terminals("recall_outcome"))
    assert "mind_mem.validity_gate._sneaky_outcome_penalty" in touching
    assert sorted(touching - set(STAGING_STORE_READERS["recall_outcome"])) == ["mind_mem.validity_gate._sneaky_outcome_penalty"]


def test_the_scan_follows_constants_across_imports_and_composition() -> None:
    mutated = _Source(
        {
            "mind_mem._scan_probe_a": '_TABLE = "recall_outcome"\n_SQL = "SELECT * FROM " + _TABLE\n',
            "mind_mem._scan_probe_b": (
                "from mind_mem._scan_probe_a import _SQL\n"
                "from mind_mem import _scan_probe_a\n"
                "def via_import(conn):\n    return conn.execute(_SQL)\n"
                "def via_module(conn):\n    return conn.execute(_scan_probe_a._TABLE)\n"
            ),
        }
    )
    touching = mutated.touching("recall_outcome", _terminals("recall_outcome"))
    assert {"mind_mem._scan_probe_b.via_import", "mind_mem._scan_probe_b.via_module"} <= touching


def test_the_scan_is_not_blind(source: _Source) -> None:
    """Positive control: the validity gate reaches recall_outcome only through
    three hops of plumbing; the scan must find it, or it proves nothing."""
    touching = source.touching("recall_outcome", frozenset())
    assert "mind_mem.validity_gate._load_outcome_signals" in touching
    assert "mind_mem.outcome_attribution.report_outcome" in touching


def test_serving_readers_name_their_enabling_keys() -> None:
    """A serving reader that is always on makes its store un-stageable."""
    for token, readers in STAGING_STORE_READERS.items():
        for func, reader in readers.items():
            if reader.role == "serving":
                assert reader.enabled_by, f"{token}: {func} serves other clients unconditionally"


def test_approval_readers_are_admin_tools() -> None:
    """A reader that applies a staged row to shared state is an admin tool."""
    for token, readers in STAGING_STORE_READERS.items():
        for func, reader in readers.items():
            if reader.role == "approval" and func.startswith("mind_mem.mcp.tools."):
                assert func.rsplit(".", 1)[-1] in ADMIN_TOOLS, f"{token}: {func} applies staging but is not admin"


def test_reader_config_keys_cover_the_serving_readers() -> None:
    keys = set(cross_client_reader_config_keys())
    for readers in STAGING_STORE_READERS.values():
        for reader in readers.values():
            if reader.role == "serving":
                assert set(reader.enabled_by) <= keys


# ---------------------------------------------------------------------------
# The gate, end to end: report_outcome under the outcome-attribution reader
# ---------------------------------------------------------------------------

D1 = "D-20260101-001"
_OUTCOME = dict(block_ids=[D1], outcome="failure", task_id="t-reader")
_DECISION = f"\n[{D1}]\nDate: 2026-01-01\nStatus: active\nScope: global\nStatement: Use frost telemetry.\nTags: frost\n\n"
_OA_ON = {"enabled": True, "outcome_attribution": {"enabled": True}}


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    from mind_mem.init_workspace import init

    root = tempfile.mkdtemp(prefix="mm_acl_readers_")
    ws = os.path.join(root, "ws")
    init(ws)
    with open(os.path.join(ws, "decisions", "DECISIONS.md"), "a", encoding="utf-8") as handle:
        handle.write(_DECISION)
    monkeypatch.setenv("MIND_MEM_WORKSPACE", ws)
    monkeypatch.setenv("MIND_MEM_CONFIG", os.path.join(ws, "mind-mem.json"))
    monkeypatch.delenv("MIND_MEM_ACL_DISABLED", raising=False)
    try:
        yield ws
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _set_validity_gate(ws: str, section: dict[str, Any] | None) -> None:
    path = os.path.join(ws, "mind-mem.json")
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    recall = cfg.setdefault("recall", {})
    if section is None:
        recall.pop("validity_gate", None)
    else:
        recall["validity_gate"] = section
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)


def _outcome_rows(ws: str) -> int:
    db = os.path.join(ws, ".mind-mem-index", "recall.db")
    if not os.path.exists(db):
        return 0
    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT COUNT(*) FROM recall_outcome").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        con.close()


def _report(ws: str, scope: str, monkeypatch: pytest.MonkeyPatch) -> str:
    from mind_mem.mcp.tools.calibration import report_outcome

    monkeypatch.setenv("MIND_MEM_SCOPE", scope)
    from mind_mem.mcp.infra import rate_limit as _rate_limit

    with _rate_limit._rate_limiters_lock:
        _rate_limit._rate_limiters.clear()
    with use_workspace(ws):
        return str(report_outcome(**_OUTCOME))


_READER_CONFIGS = [
    None,
    {"enabled": False, "outcome_attribution": {"enabled": True}},
    {"enabled": True, "outcome_attribution": {"enabled": False}},
    {"enabled": True},
    _OA_ON,
]
_READER_CONFIG_IDS = ["no-gate", "gate-off", "attribution-off", "attribution-absent", "attribution-on"]


@pytest.mark.parametrize("section", _READER_CONFIGS, ids=_READER_CONFIG_IDS)
def test_user_scope_report_outcome_is_denied_whatever_the_reader_config(
    section: Any, workspace: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row written while the reader is OFF is served once the operator turns
    it ON (load_outcome_signals counts every row, no actor or scope filter), so
    a config probe at write time only delays the demotion. The write is denied
    at user scope under every config; the admin call lands it (positive control)."""
    _set_validity_gate(workspace, section)
    result = _report(workspace, "user", monkeypatch)
    assert "requires admin scope" in result, result[:300]
    assert _outcome_rows(workspace) == 0
    admin = json.loads(_report(workspace, "admin", monkeypatch))
    assert admin.get("status") == "recorded", admin
    assert _outcome_rows(workspace) == 1


def test_no_user_row_is_waiting_when_the_reader_is_turned_on(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-seed: user reports while the opt-in is off, then the operator
    enables it. Nothing the user sent may reach the served signals."""
    from mind_mem.outcome_attribution import load_outcome_signals

    _set_validity_gate(workspace, {"enabled": True, "outcome_attribution": {"enabled": False}})
    for _ in range(3):
        _report(workspace, "user", monkeypatch)
    _set_validity_gate(workspace, _OA_ON)
    assert _outcome_rows(workspace) == 0
    assert not load_outcome_signals(workspace, [D1])
    # Positive control: an admin report does reach the signals the reader serves.
    _report(workspace, "admin", monkeypatch)
    assert load_outcome_signals(workspace, [D1])[D1].failure == 1


def _mcp_writers(token: str) -> set[str]:
    return {
        func.rsplit(".", 1)[-1]
        for func, reader in STAGING_STORE_READERS[token].items()
        if reader.role == "writer" and func.startswith("mind_mem.mcp.tools.")
    }


def _serving(token: str) -> bool:
    return any(r.role == "serving" for r in STAGING_STORE_READERS.get(token, {}).values())


def test_a_store_some_reader_serves_to_others_has_only_admin_writers() -> None:
    """Derived from the reader table, config-independent: a serving reader --
    on or off today -- makes every MCP writer of its store admin-scope."""
    served = [t for t in STAGING_STORE_READERS if _serving(t)]
    assert served, "positive control: recall_outcome has a serving reader"
    for token in served:
        assert _mcp_writers(token), f"{token}: no MCP writer classified"
        assert _mcp_writers(token) <= ADMIN_TOOLS, (
            f"{token} is served to other clients but user tools write it: {_mcp_writers(token) - ADMIN_TOOLS}"
        )
        staged_by = sorted(name for name, stores in USER_STAGING_WRITES.items() if token in {staging_store_token(s) for s in stores})
        assert staged_by == [], f"{token} is served to other clients but staged at user scope by {staged_by}"


def test_the_runtime_gate_denies_a_served_store_with_every_reader_off(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backstop and mutation: pin a user staging write onto a served store and
    the gate denies it even with no reader config at all; on an unserved store
    it stays staging."""
    from mind_mem.mcp.infra import acl

    _set_validity_gate(workspace, None)
    monkeypatch.setitem(acl.USER_STAGING_WRITES, "probe_writer", (".mind-mem-index/recall.db::recall_outcome",))
    mode = acl.write_mode_of("probe_writer", {})
    assert mode is not None and "outcome_attribution" in mode.capability, mode
    monkeypatch.setitem(acl.USER_STAGING_WRITES, "probe_writer", ("memory/interaction_signals.jsonl",))
    assert acl.write_mode_of("probe_writer", {}) is None


def test_the_reader_the_gate_names_really_reads_the_store(workspace: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The enabling keys are the right ones: under them the validity gate
    actually loads outcome signals; with them off it does not."""
    from mind_mem.validity_gate import _load_outcome_signals

    assert _load_outcome_signals({"enabled": True}, workspace, [D1]) is None
    assert _load_outcome_signals(_OA_ON, workspace, [D1]) is not None


def test_admin_scope_is_never_gated_by_a_reader(workspace: str) -> None:
    _set_validity_gate(workspace, _OA_ON)
    assert check_write_mode_acl("report_outcome", dict(_OUTCOME, project_to_calibration=False), "admin") is None
