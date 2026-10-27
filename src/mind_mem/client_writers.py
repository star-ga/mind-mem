# Copyright 2026 STARGA, Inc.
"""MCP config writers and removers, one per client config dialect.

Every writer is a pure function ``(existing, workspace, srv) -> (new,
changed)``: JSON writers take and return a dict, text writers (TOML, YAML)
take and return the file text. Every remover is ``existing -> (new,
changed)``. File I/O, backups and refusal policy live in
:mod:`mind_mem.client_mcp`; nothing here touches the disk.

JSON dialects are data: a key path plus an entry shape (see
:data:`JSON_FORMATS`). Text dialects edit only the mind-mem section, entry
or list item, so comments and every other server survive. A text config
that cannot be edited safely raises :class:`UnsafeConfigError`.
"""

from __future__ import annotations

import json
import re
from functools import partial
from typing import Any, Callable

from mind_mem.client_config_io import UnsafeConfigError

SERVER = "mind-mem"


# ---------------------------------------------------------------------------
# JSON dialects
# ---------------------------------------------------------------------------


def _base(srv: dict[str, Any]) -> dict[str, Any]:
    return {"command": srv["command"], "args": list(srv["args"]), "env": dict(srv["env"])}


_SHAPES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    # {command, args, env} -- the de-facto standard.
    "standard": _base,
    # Explicit stdio transport tag (Claude Code, Cursor, VS Code, Crush).
    "stdio": lambda s: {"type": "stdio", **_base(s)},
    # GitHub Copilot CLI: "local" type plus a tool allowlist.
    "copilot": lambda s: {"type": "local", **_base(s), "tools": ["*"]},
    # Cline / Kiro: explicit enable flag and an empty auto-approve list.
    "cline": lambda s: {**_base(s), "disabled": False, "autoApprove": []},
    # Factory Droid.
    "factory": lambda s: {"type": "stdio", **_base(s), "disabled": False},
    # OpenHands CLI.
    "openhands": lambda s: {**_base(s), "transport": "stdio"},
    # AnythingLLM: start the server with the app.
    "anythingllm": lambda s: {**_base(s), "anythingllm": {"autoStart": True}},
    # Jan: entries are inactive unless flagged.
    "jan": lambda s: {**_base(s), "active": True},
    # Zed context_servers.
    "zed": lambda s: {"source": "custom", **_base(s)},
}

# mcp_fmt -> (key path to the server map, entry shape). A key that contains
# a dot (Amp's "amp.mcpServers") is ONE flat key, not a path.
JSON_FORMATS: dict[str, tuple[tuple[str, ...], str]] = {
    "mcp-json-servers": (("mcpServers",), "standard"),
    "mcp-json-windsurf": (("mcpServers",), "standard"),
    "mcp-json-cursor": (("mcpServers",), "stdio"),
    "mcp-json-claude": (("mcpServers",), "stdio"),
    "mcp-json-zed": (("context_servers",), "zed"),
    "mcp-json-vscode": (("servers",), "stdio"),
    "mcp-json-visualstudio": (("servers",), "standard"),
    "mcp-json-openclaw": (("mcp", "servers"), "standard"),
    "mcp-json-amp": (("amp.mcpServers",), "standard"),
    "mcp-json-crush": (("mcp",), "stdio"),
    "mcp-json-copilot": (("mcpServers",), "copilot"),
    "mcp-json-cline": (("mcpServers",), "cline"),
    "mcp-json-factory": (("mcpServers",), "factory"),
    "mcp-json-openhands": (("mcpServers",), "openhands"),
    "mcp-json-anythingllm": (("mcpServers",), "anythingllm"),
    "mcp-json-jan": (("mcpServers",), "jan"),
}


def json_entry(fmt: str, srv: dict[str, Any]) -> dict[str, Any]:
    """The mind-mem entry a JSON dialect writes."""
    return _SHAPES[JSON_FORMATS[fmt][1]](srv)


def _descend(out: dict[str, Any], keys: tuple[str, ...], create: bool) -> dict[str, Any] | None:
    node: Any = out
    for key in keys:
        if not isinstance(node, dict):
            raise UnsafeConfigError(f"`{key}` parent is not an object; left untouched")
        if key not in node:
            if not create:
                return None
            node[key] = {}
        node = node[key]
    if not isinstance(node, dict):
        raise UnsafeConfigError(f"`{'.'.join(keys)}` is not an object; left untouched")
    return node


def merge_json_mcp(
    keys: tuple[str, ...], shape: str, existing: dict[str, Any], ws: str, srv: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Set ``<keys>.mind-mem`` to the shaped entry; keep every other key."""
    out = json.loads(json.dumps(existing))
    servers = _descend(out, keys, create=True)
    assert servers is not None
    target = _SHAPES[shape](srv)
    if servers.get(SERVER) == target:
        return out, False
    servers[SERVER] = target
    return out, True


def remove_json_mcp(keys: tuple[str, ...], existing: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Delete ``<keys>.mind-mem``; everything else is left as it was."""
    out = json.loads(json.dumps(existing))
    servers = _descend(out, keys, create=False)
    if servers is None or SERVER not in servers:
        return out, False
    del servers[SERVER]
    return out, True


JSON_WRITERS: dict[str, Callable[..., tuple[dict[str, Any], bool]]] = {
    fmt: partial(merge_json_mcp, keys, shape) for fmt, (keys, shape) in JSON_FORMATS.items()
}


def remove_opencode_mcp(existing: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """OpenCode-style configs keep the entry at ``mcp.mind-mem`` (1.x) or
    ``mcp.servers.mind-mem`` (2.x native); remove whichever is present."""
    out, a = remove_json_mcp(("mcp",), existing)
    servers = out.get("mcp", {}).get("servers") if isinstance(out.get("mcp"), dict) else None
    b = False
    if isinstance(servers, dict) and "type" not in servers:
        out, b = remove_json_mcp(("mcp", "servers"), out)
    return out, a or b


# ---------------------------------------------------------------------------
# TOML dialects
# ---------------------------------------------------------------------------


def _toml_str(value: str) -> str:
    """A TOML basic string. JSON escaping is a valid subset (\\\\, \\", \\uXXXX),
    so Windows paths with backslashes stay well-formed."""
    return json.dumps(str(value), ensure_ascii=False)


def _toml_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_str(v) for v in values) + "]"


def _toml_inline_table(mapping: dict[str, str]) -> str:
    return "{ " + ", ".join(f"{k} = {_toml_str(v)}" for k, v in mapping.items()) + " }"


_CODEX_SECTION = re.compile(
    r"\[mcp_servers\.(?:\"mind-mem\"|mind-mem)(?:\.[^\]]+)?\][\s\S]*?(?=\n\[(?!mcp_servers\.(?:\"mind-mem\"|mind-mem)[\].])[^\]]+\]|\Z)"
)


def merge_codex_toml(existing_text: str, ws: str, srv: dict[str, Any]) -> tuple[str, bool]:
    """``[mcp_servers.mind-mem]`` section (Codex, Grok Build, Open Interpreter).

    Replaces any earlier mind-mem section and its sub-tables; everything
    outside it is preserved byte for byte.
    """
    new_section = (
        "[mcp_servers.mind-mem]\n"
        f"command = {_toml_str(srv['command'])}\n"
        f"args = {_toml_array(srv['args'])}\n"
        f"env = {_toml_inline_table(srv['env'])}\n"
    )
    text = existing_text or ""
    if new_section.strip() in text:
        return text, False
    text = _CODEX_SECTION.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text and not text.endswith("\n"):
        text += "\n"
    text += ("\n" + new_section) if text.strip() else new_section
    return text, True


def remove_codex_toml(existing_text: str) -> tuple[str, bool]:
    text = existing_text or ""
    new = _CODEX_SECTION.sub("", text)
    if new == text:
        return text, False
    new = re.sub(r"\n{3,}", "\n\n", new)
    return new, True


_VIBE_INLINE = re.compile(r"^mcp_servers\s*=\s*\[([\s\S]*?)\]\s*$", re.MULTILINE)
_VIBE_AOT_HEADER = re.compile(r"^\[\[mcp_servers\]\]\s*$", re.MULTILINE)
_TABLE_HEADER = re.compile(r"^\[", re.MULTILINE)


def _vibe_aot_blocks(text: str) -> list[tuple[int, int, str]]:
    """``[[mcp_servers]]`` blocks as (start, end, body), each running to the
    next table header that is not one of its own sub-tables."""
    blocks = []
    for m in _VIBE_AOT_HEADER.finditer(text):
        end = len(text)
        for h in _TABLE_HEADER.finditer(text, m.end()):
            line = text[h.start() : text.find("\n", h.start()) if "\n" in text[h.start() :] else len(text)]
            if line.startswith("[mcp_servers.") and not line.startswith("[["):
                continue  # sub-table of this array element
            end = h.start()
            break
        blocks.append((m.start(), end, text[m.start() : end]))
    return blocks


def _is_mm_block(body: str) -> bool:
    return re.search(r'^\s*name\s*=\s*"mind-mem"\s*$', body, re.MULTILINE) is not None


def merge_vibe_toml(existing_text: str, ws: str, srv: dict[str, Any]) -> tuple[str, bool]:
    """Mistral Vibe: ``[[mcp_servers]]`` array of tables (documented form).

    A file that already uses the inline ``mcp_servers = [ ... ]`` array keeps
    that form -- adding ``[[mcp_servers]]`` beside it would define the key
    twice. Any earlier mind-mem element is replaced; others are kept.
    """
    text = existing_text or ""
    m = _VIBE_INLINE.search(text)
    if m:
        fields = (
            f"name = {_toml_str(SERVER)}, transport = {_toml_str('stdio')}, command = {_toml_str(srv['command'])}, "
            f"args = {_toml_array(srv['args'])}, env = {_toml_inline_table(srv['env'])}"
        )
        entry = f"  {{ {fields} }}"
        kept = [ln for ln in m.group(1).split("\n") if not re.search(r'\{\s*name\s*=\s*"mind-mem"', ln)]
        body = "\n".join(kept).strip().rstrip(",").strip()
        inner = f"{body},\n{entry}" if body else entry
        new = text[: m.start()] + f"mcp_servers = [\n{inner}\n]" + text[m.end() :]
        return new, new != text
    block = (
        "[[mcp_servers]]\n"
        f"name = {_toml_str(SERVER)}\n"
        f"transport = {_toml_str('stdio')}\n"
        f"command = {_toml_str(srv['command'])}\n"
        f"args = {_toml_array(srv['args'])}\n"
        f"env = {_toml_inline_table(srv['env'])}\n"
    )
    mm = [b for b in _vibe_aot_blocks(text) if _is_mm_block(b[2])]
    if len(mm) == 1 and mm[0][2].strip() == block.strip():
        return text, False
    stripped, _ = remove_vibe_toml(text)
    if stripped and not stripped.endswith("\n"):
        stripped += "\n"
    new = stripped + ("\n" if stripped.strip() else "") + block
    return new, new != text


def remove_vibe_toml(existing_text: str) -> tuple[str, bool]:
    text = existing_text or ""
    m = _VIBE_INLINE.search(text)
    if m:
        kept = [ln for ln in m.group(1).split("\n") if not re.search(r'\{\s*name\s*=\s*"mind-mem"', ln)]
        body = "\n".join(kept).strip().rstrip(",").strip()
        new = text[: m.start()] + (f"mcp_servers = [\n{body}\n]" if body else "mcp_servers = []") + text[m.end() :]
        return new, new != text
    new = text
    for start, end, body in reversed(_vibe_aot_blocks(text)):
        if _is_mm_block(body):
            new = new[:start] + new[end:]
    if new == text:
        return text, False
    return re.sub(r"\n{3,}", "\n\n", new).rstrip("\n") + ("\n" if new.strip() else ""), True


# ---------------------------------------------------------------------------
# YAML dialects (line-based: comments and unrelated keys are untouched)
# ---------------------------------------------------------------------------


def _ind(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _skip(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith("#")


def _key_rest(line: str, key: str) -> str | None:
    m = re.match(r"^\s*([\"']?)" + re.escape(key) + r"\1\s*:(?P<rest>.*)$", line)
    if m is None:
        return None
    rest = m.group("rest")
    if rest.strip().startswith("#"):
        return ""
    return re.sub(r"\s+#.*$", "", rest).strip()


def _block_end(lines: list[str], idx: int, indent: int, dash_same: bool = False) -> int:
    last = idx
    for i in range(idx + 1, len(lines)):
        if _skip(lines[i]):
            continue
        n = _ind(lines[i])
        if n > indent or (dash_same and n == indent and lines[i].lstrip().startswith("- ")):
            last = i
            continue
        break
    return last + 1


def _child_indent(lines: list[str], lo: int, hi: int, default: int) -> int:
    for i in range(lo, hi):
        if not _skip(lines[i]):
            return _ind(lines[i])
    return default


def _find(lines: list[str], key: str, lo: int, hi: int, indent: int) -> tuple[int, str] | None:
    for i in range(lo, hi):
        if _skip(lines[i]) or _ind(lines[i]) != indent:
            continue
        rest = _key_rest(lines[i], key)
        if rest is not None:
            return i, rest
    return None


def _yaml_value(v: Any) -> str:
    """JSON is a subset of YAML 1.2 flow style."""
    return json.dumps(v, ensure_ascii=False)


def _locate_map(lines: list[str], path: tuple[str, ...], create: bool) -> tuple[int, int, int] | None:
    """Return ``(lo, hi, child_indent)`` of the mapping at *path*, creating
    missing levels at the end of their parent when *create* is set."""
    lo, hi, indent = 0, len(lines), 0
    for depth, key in enumerate(path):
        found = _find(lines, key, lo, hi, indent)
        if found is None:
            if not create:
                return None
            insert_at = hi
            new = [" " * (indent + 2 * j) + f"{k}:" for j, k in enumerate(path[depth:])]
            lines[insert_at:insert_at] = new
            return insert_at + len(new), insert_at + len(new), indent + 2 * len(new)
        idx, rest = found
        if rest in ("{}", "null", "~"):
            lines[idx] = " " * indent + f"{key}:"
        elif rest:
            raise UnsafeConfigError(f"`{key}` uses YAML flow style; left untouched — add the snippet by hand")
        end = _block_end(lines, idx, indent)
        indent = _child_indent(lines, idx + 1, end, indent + 2)
        lo, hi = idx + 1, end
    return lo, hi, indent


def _validate_yaml(old: str, new: str, path: tuple[str, ...], present: bool) -> None:
    """Cross-check a line edit with PyYAML when it is installed: the edit must
    parse, and must change nothing but the mind-mem entry."""
    try:
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:  # pragma: no cover - optional
        return
    try:
        before = yaml.safe_load(old) or {}
        after = yaml.safe_load(new) or {}
    except yaml.YAMLError as exc:
        raise UnsafeConfigError(f"config is not valid YAML ({exc.__class__.__name__}); left untouched") from exc

    def strip(doc: Any) -> Any:
        """*doc* without the mind-mem entry and without any container on
        *path* that holds nothing else (the edit may have created them)."""
        doc = json.loads(json.dumps(doc, default=str))
        chain: list[dict[str, Any]] = []
        node = doc
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return doc
            chain.append(node)
            node = node[key]
        if isinstance(node, dict):
            node.pop(SERVER, None)
        elif isinstance(node, list):
            node[:] = [x for x in node if not (isinstance(x, dict) and x.get("name") == SERVER)]
        for parent, key in reversed(list(zip(chain, path))):
            if parent[key] in ({}, [], None):
                del parent[key]
            else:
                break
        return doc

    if strip(before) != strip(after):
        raise UnsafeConfigError("YAML edit would change more than the mind-mem entry; left untouched")
    if present:
        node: Any = after
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        ok = (isinstance(node, dict) and SERVER in node) or (
            isinstance(node, list) and any(isinstance(x, dict) and x.get("name") == SERVER for x in node)
        )
        if not ok:
            raise UnsafeConfigError("YAML edit did not produce the mind-mem entry; left untouched")


def _join(lines: list[str]) -> str:
    return "\n".join(lines).rstrip("\n") + "\n" if any(ln.strip() for ln in lines) else ""


def yaml_upsert_map(existing_text: str, path: tuple[str, ...], body: dict[str, Any]) -> tuple[str, bool]:
    """Set ``<path>.mind-mem`` to *body* in a block-style YAML mapping."""
    text = existing_text or ""
    lines = text.splitlines()
    lo, hi, ci = _locate_map(lines, path, create=True)  # type: ignore[misc]
    entry = [" " * ci + f"{SERVER}:"] + [" " * (ci + 2) + f"{k}: {_yaml_value(v)}" for k, v in body.items()]
    found = _find(lines, SERVER, lo, hi, ci)
    if found is not None:
        idx = found[0]
        end = _block_end(lines, idx, ci)
        if lines[idx:end] == entry:
            return text, False
        lines[idx:end] = entry
    else:
        lines[hi:hi] = entry
    new = _join(lines)
    _validate_yaml(text, new, path, present=True)
    return new, new != text


def yaml_remove_map(existing_text: str, path: tuple[str, ...]) -> tuple[str, bool]:
    text = existing_text or ""
    lines = text.splitlines()
    loc = _locate_map(lines, path, create=False)
    if loc is None:
        return text, False
    lo, hi, ci = loc
    found = _find(lines, SERVER, lo, hi, ci)
    if found is None:
        return text, False
    del lines[found[0] : _block_end(lines, found[0], ci)]
    new = _join(lines)
    _validate_yaml(text, new, path, present=False)
    return new, True


def _list_items(lines: list[str], lo: int, hi: int) -> tuple[int, list[tuple[int, int]]]:
    item_ind = -1
    items: list[tuple[int, int]] = []
    for i in range(lo, hi):
        if _skip(lines[i]):
            continue
        if item_ind < 0:
            item_ind = _ind(lines[i])
        if _ind(lines[i]) == item_ind and lines[i].lstrip().startswith("- "):
            items.append((i, _block_end(lines, i, item_ind)))
    return item_ind, items


def _is_mm_item(lines: list[str], start: int, end: int) -> bool:
    return any(re.match(r"^\s*(?:-\s+)?name\s*:\s*[\"']?mind-mem[\"']?\s*(?:#.*)?$", ln) for ln in lines[start:end])


def _locate_list(lines: list[str], key: str, create: bool) -> tuple[int, int] | None:
    found = _find(lines, key, 0, len(lines), 0)
    if found is None:
        if not create:
            return None
        lines.append(f"{key}:")
        return len(lines), len(lines)
    idx, rest = found
    if rest in ("[]", "null", "~"):
        lines[idx] = f"{key}:"
    elif rest:
        raise UnsafeConfigError(f"`{key}` uses YAML flow style; left untouched — add the snippet by hand")
    return idx + 1, _block_end(lines, idx, 0, dash_same=True)


def yaml_upsert_list(existing_text: str, key: str, body: dict[str, Any]) -> tuple[str, bool]:
    """Set the ``- name: mind-mem`` item of the top-level list *key*."""
    text = existing_text or ""
    lines = text.splitlines()
    lo, hi = _locate_list(lines, key, create=True)  # type: ignore[misc]
    item_ind, items = _list_items(lines, lo, hi)
    ii = item_ind if item_ind >= 0 else 2
    pairs = [f"{k}: {_yaml_value(v)}" for k, v in body.items()]
    entry = [" " * ii + "- " + pairs[0]] + [" " * (ii + 2) + p for p in pairs[1:]]
    mine = [(s, e) for s, e in items if _is_mm_item(lines, s, e)]
    if mine:
        s, e = mine[0]
        if lines[s:e] == entry and len(mine) == 1:
            return text, False
        for s2, e2 in reversed(mine[1:]):
            del lines[s2:e2]
        lines[s:e] = entry
    else:
        lines[hi:hi] = entry
    new = _join(lines)
    _validate_yaml(text, new, (key,), present=True)
    return new, new != text


def yaml_remove_list(existing_text: str, key: str) -> tuple[str, bool]:
    text = existing_text or ""
    lines = text.splitlines()
    loc = _locate_list(lines, key, create=False)
    if loc is None:
        return text, False
    _, items = _list_items(lines, *loc)
    mine = [(s, e) for s, e in items if _is_mm_item(lines, s, e)]
    if not mine:
        return text, False
    for s, e in reversed(mine):
        del lines[s:e]
    new = _join(lines)
    _validate_yaml(text, new, (key,), present=False)
    return new, True


def _goose(srv: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": True,
        "type": "stdio",
        "name": SERVER,
        "cmd": srv["command"],
        "args": list(srv["args"]),
        "envs": dict(srv["env"]),
        "timeout": 300,
    }


def _plain(srv: dict[str, Any]) -> dict[str, Any]:
    return _base(srv)


def _fast_agent(srv: dict[str, Any]) -> dict[str, Any]:
    return {"transport": "stdio", **_base(srv)}


def _continue_item(srv: dict[str, Any]) -> dict[str, Any]:
    return {"name": SERVER, **_base(srv)}


# mcp_fmt -> (writer, remover) for every text dialect.
TEXT_FORMATS: dict[str, tuple[Callable[..., tuple[str, bool]], Callable[[str], tuple[str, bool]]]] = {
    "mcp-toml-codex": (merge_codex_toml, remove_codex_toml),
    "mcp-toml-vibe": (merge_vibe_toml, remove_vibe_toml),
    "mcp-yaml-goose": (
        lambda t, ws, srv: yaml_upsert_map(t, ("extensions",), _goose(srv)),
        lambda t: yaml_remove_map(t, ("extensions",)),
    ),
    "mcp-yaml-hermes": (
        lambda t, ws, srv: yaml_upsert_map(t, ("mcp_servers",), _plain(srv)),
        lambda t: yaml_remove_map(t, ("mcp_servers",)),
    ),
    "mcp-yaml-fast-agent": (
        lambda t, ws, srv: yaml_upsert_map(t, ("mcp", "servers"), _fast_agent(srv)),
        lambda t: yaml_remove_map(t, ("mcp", "servers")),
    ),
    "mcp-yaml-continue": (
        lambda t, ws, srv: yaml_upsert_list(t, "mcpServers", _continue_item(srv)),
        lambda t: yaml_remove_list(t, "mcpServers"),
    ),
}

TEXT_WRITERS: dict[str, Callable[..., tuple[str, bool]]] = {k: v[0] for k, v in TEXT_FORMATS.items()}


__all__ = [
    "JSON_FORMATS",
    "JSON_WRITERS",
    "SERVER",
    "TEXT_FORMATS",
    "TEXT_WRITERS",
    "json_entry",
    "merge_codex_toml",
    "merge_json_mcp",
    "merge_vibe_toml",
    "remove_codex_toml",
    "remove_json_mcp",
    "remove_opencode_mcp",
    "remove_vibe_toml",
    "yaml_remove_list",
    "yaml_remove_map",
    "yaml_upsert_list",
    "yaml_upsert_map",
]
