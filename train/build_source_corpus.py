#!/usr/bin/env python3
"""Build a source-only MCP contract corpus.

This module intentionally uses only the standard library.  It reads the
checked-in registration wiring and tool modules as text/AST, so producing the
artifact cannot execute the package or import an old corpus/evaluation
pipeline.  The artifact is preparation evidence; it is not a training or
evaluation certificate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = "mind-mem/source-mcp-contract-corpus@1"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_FILES = 64
MAX_DOCSTRING_BYTES = 64 * 1024
MAX_RECORDS = 512
MAX_MESSAGE_BYTES = 128 * 1024
EXPECTED_SOURCE_FILES = (
    Path("src/mind_mem/mcp_server.py"),
    Path("src/mind_mem/mcp/server.py"),
)


class SourceCorpusError(ValueError):
    """A fail-closed source or registration contract error."""


@dataclass(frozen=True)
class Registration:
    name: str
    module: str
    path: Path
    definition_line: int
    registration_line: int
    function: ast.FunctionDef | ast.AsyncFunctionDef


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_source_path(root: Path, relative: Path) -> tuple[Path, bytes]:
    if relative.is_absolute() or ".." in relative.parts:
        raise SourceCorpusError(f"source path is outside the fixed source set: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise SourceCorpusError(f"required source is missing or symlinked: {relative}")
    resolved = path.resolve()
    if resolved != path.absolute() or resolved.parent != path.absolute().parent:
        raise SourceCorpusError(f"source path resolves outside checkout: {relative}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise SourceCorpusError(f"source exceeds {MAX_SOURCE_BYTES} bytes: {relative}")
    with path.open("rb") as handle:
        data = handle.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise SourceCorpusError(f"source exceeds {MAX_SOURCE_BYTES} bytes: {relative}")
    return path, data


def _source_files(root: Path) -> list[tuple[Path, bytes]]:
    if not root.is_dir() or root.is_symlink():
        raise SourceCorpusError(f"repository root is not a real directory: {root}")
    paths = list(EXPECTED_SOURCE_FILES)
    tools_root = root / "src/mind_mem/mcp/tools"
    if not tools_root.is_dir() or tools_root.is_symlink():
        raise SourceCorpusError("MCP tools source directory is missing or symlinked")
    tools = sorted(
        (p.relative_to(root) for p in tools_root.glob("*.py") if p.is_file()),
        key=lambda p: p.as_posix(),
    )
    paths.extend(tools)
    if len(paths) > MAX_SOURCE_FILES:
        raise SourceCorpusError(f"source file count exceeds {MAX_SOURCE_FILES}")
    return [_safe_source_path(root, path) for path in paths]


def _literal_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef, path: Path) -> str:
    if not node.body or not isinstance(node.body[0], ast.Expr):
        raise SourceCorpusError(f"registered tool has no literal docstring: {path}:{node.lineno}")
    value = node.body[0].value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        raise SourceCorpusError(f"registered tool docstring is not a literal: {path}:{node.lineno}")
    doc = value.value
    if len(doc.encode("utf-8")) > MAX_DOCSTRING_BYTES:
        raise SourceCorpusError(f"registered tool docstring is too large: {path}:{node.lineno}")
    return doc


def _annotation(node: ast.expr | None) -> str | None:
    return ast.unparse(node) if node is not None else None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, object]:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [False] * (len(positional) - len(args.defaults)) + [True] * len(args.defaults)

    def parameter(arg: ast.arg, has_default: bool = False) -> dict[str, object]:
        return {
            "name": arg.arg,
            "annotation": _annotation(arg.annotation),
            "has_default": has_default,
        }

    params: list[dict[str, object]] = []
    for index, arg in enumerate(args.posonlyargs):
        params.append({**parameter(arg, defaults[index]), "kind": "positional_only"})
    offset = len(args.posonlyargs)
    for index, arg in enumerate(args.args, offset):
        params.append({**parameter(arg, defaults[index]), "kind": "positional_or_keyword"})
    if args.vararg is not None:
        params.append({**parameter(args.vararg), "kind": "var_positional"})
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append({**parameter(arg, default is not None), "kind": "keyword_only"})
    if args.kwarg is not None:
        params.append({**parameter(args.kwarg), "kind": "var_keyword"})
    return {"parameters": params, "return": _annotation(node.returns), "async": isinstance(node, ast.AsyncFunctionDef)}


def _module_definitions(path: Path, source: bytes) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    try:
        tree = ast.parse(source.decode("utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SourceCorpusError(f"cannot parse source {path}: {exc}") from exc
    definitions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in definitions:
                raise SourceCorpusError(f"ambiguous function definition {node.name!r} in {path}")
            definitions[node.name] = node
    return definitions


def _registration_modules(server_source: bytes) -> list[tuple[str, str, int]]:
    try:
        tree = ast.parse(server_source.decode("utf-8"), filename="mcp/server.py")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SourceCorpusError(f"cannot parse registration wiring: {exc}") from exc
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "mind_mem.mcp.tools":
            for item in node.names:
                alias = item.asname or item.name
                previous = aliases.get(alias)
                if previous is not None:
                    raise SourceCorpusError(f"ambiguous registration import alias {alias!r}")
                aliases[alias] = item.name
    registrations: list[tuple[str, str, int]] = []
    direct_calls: set[int] = set()
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "register":
            continue
        if len(call.args) != 1 or call.keywords or not isinstance(call.func.value, ast.Name):
            raise SourceCorpusError(f"unsupported register call at mcp/server.py:{node.lineno}")
        if not isinstance(call.args[0], ast.Name) or call.args[0].id != "mcp":
            raise SourceCorpusError(f"registration receiver is not mcp at mcp/server.py:{node.lineno}")
        direct_calls.add(id(call))
        alias = call.func.value.id
        module = aliases.get(alias)
        if module is None:
            # Resources are registered through the same ``register`` shape,
            # but are outside the MCP tool contract surface.  Any unknown
            # tool-looking alias remains an explicit resolution failure.
            if alias == "_resources":
                continue
            raise SourceCorpusError(f"unresolved registration module {alias!r} at mcp/server.py:{node.lineno}")
        registrations.append((alias, module, node.lineno))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and id(node) not in direct_calls
        ):
            raise SourceCorpusError(f"nested or conditional register call at mcp/server.py:{node.lineno}")
    if not registrations:
        raise SourceCorpusError("no MCP registration calls found in mcp/server.py")
    return registrations


def _registered_tools(root: Path, loaded: dict[Path, bytes]) -> list[Registration]:
    server_path = root / "src/mind_mem/mcp/server.py"
    modules = _registration_modules(loaded[server_path])
    result: list[Registration] = []
    seen: dict[str, Registration] = {}
    for _alias, module, register_line in modules:
        path = root / "src/mind_mem/mcp/tools" / f"{module}.py"
        source = loaded.get(path)
        if source is None:
            raise SourceCorpusError(f"registered module source was not loaded: {module}")
        try:
            tree = ast.parse(source.decode("utf-8"), filename=str(path))
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise SourceCorpusError(f"cannot parse tool module {path}: {exc}") from exc
        register_defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "register"]
        if len(register_defs) != 1:
            raise SourceCorpusError(f"registration function is missing or ambiguous: {path}")
        definitions = _module_definitions(path, source)
        register_body = register_defs[0].body
        direct_tool_attributes: set[int] = set()
        for nested in register_body:
            if any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tool"
                for node in ast.walk(nested)
            ) and not (isinstance(nested, ast.Expr) and isinstance(nested.value, ast.Call)):
                raise SourceCorpusError(f"nested or conditional mcp.tool registration at {path}:{nested.lineno}")
        for node in register_body:
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
                continue
            call = node.value
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "tool":
                continue
            if not isinstance(call.func.value, ast.Name) or call.func.value.id != "mcp" or len(call.args) != 1:
                raise SourceCorpusError(f"unsupported mcp.tool registration at {path}:{node.lineno}")
            if call.keywords:
                raise SourceCorpusError(f"mcp.tool keyword options are unsupported at {path}:{node.lineno}")
            target = call.args[0]
            if not isinstance(target, ast.Name):
                raise SourceCorpusError(f"non-symbol mcp.tool registration at {path}:{node.lineno}")
            function = definitions.get(target.id)
            if function is None:
                raise SourceCorpusError(f"registered symbol {target.id!r} is missing in {path}:{node.lineno}")
            if target.id in seen:
                previous = seen[target.id]
                raise SourceCorpusError(
                    f"ambiguous MCP tool name {target.id!r}: {previous.path}:{previous.registration_line} and {path}:{node.lineno}"
                )
            item = Registration(target.id, module, path, function.lineno, node.lineno, function)
            seen[target.id] = item
            result.append(item)
            direct_tool_attributes.add(id(call.func))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "tool"
                and isinstance(node.value, ast.Name)
                and node.value.id == "mcp"
                and id(node) not in direct_tool_attributes
            ):
                raise SourceCorpusError(f"unsupported nested or decorator mcp.tool use at {path}:{node.lineno}")
    if len(result) > MAX_RECORDS:
        raise SourceCorpusError(f"registered tool count exceeds {MAX_RECORDS}")
    return result


def _record(index: int, item: Registration, source_sha: str, root: Path) -> dict[str, object]:
    doc = _literal_docstring(item.function, item.path)
    family = f"mind_mem.mcp.tools.{item.module}:{item.name}"
    signature = _signature(item.function)
    answer = (
        f"MCP tool `{item.name}`. Family: `{family}`.\n"
        f"Signature: `{item.name}(...)` with source-level parameters and return annotation below.\n"
        f"Parameters: {json.dumps(signature['parameters'], sort_keys=True, separators=(',', ':'))}\n"
        f"Signature metadata: {json.dumps({'async': signature['async']}, sort_keys=True, separators=(',', ':'))}\n"
        f"Return annotation: {signature['return'] or 'unspecified'}.\n\n{doc}"
    )
    if len(answer.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise SourceCorpusError(f"generated contract message is too large: {family}")
    return {
        "id": f"mcp-contract-{index:04d}",
        "family": family,
        "tool_name": item.name,
        "messages": [
            {"role": "user", "content": f"What is the source-level MCP contract for `{item.name}`?"},
            {"role": "assistant", "content": answer},
        ],
        "provenance": {
            "kind": "source_contract",
            "source_path": item.path.relative_to(root).as_posix(),
            "source_sha256": source_sha,
            "definition_line": item.definition_line,
            "registration_line": item.registration_line,
            "registration_authority": "src/mind_mem/mcp/server.py",
            "family_key": family,
        },
        "status": "PREPARATION_ONLY",
    }


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current /= part
        if current.exists() and current.is_symlink():
            raise SourceCorpusError(f"output path contains symlink component: {current}")


def _prepare_output(path: Path, root: Path) -> None:
    if not path.is_absolute():
        raise SourceCorpusError("--output-dir must be an explicit absolute path")
    _reject_symlink_components(path)
    resolved = path.resolve(strict=False)
    source_root = root.resolve()
    if resolved == source_root or source_root in resolved.parents:
        raise SourceCorpusError("output directory must be outside the repository checkout")
    if path.exists():
        if not path.is_dir():
            raise SourceCorpusError(f"output path is not a directory: {path}")
        if any(path.iterdir()):
            raise SourceCorpusError(f"refusing to clobber non-empty output directory: {path}")
    else:
        path.mkdir(parents=True)


def build(repo_root: Path, output_dir: Path) -> dict[str, object]:
    root = repo_root.resolve()
    files = _source_files(root)
    loaded = {path: data for path, data in files}
    registrations = _registered_tools(root, loaded)
    if not registrations:
        raise SourceCorpusError("MCP registration surface is empty")
    source_entries = [{"path": path.relative_to(root).as_posix(), "bytes": len(data), "sha256": _sha256(data)} for path, data in files]
    source_hashes = {path: entry["sha256"] for path, entry in zip(loaded, source_entries)}
    records = [_record(index, item, source_hashes[item.path], root) for index, item in enumerate(registrations, 1)]
    corpus = b"".join(_canonical_bytes(record) for record in records)
    if not corpus:
        raise SourceCorpusError("refusing to publish an empty corpus")
    corpus_path = output_dir / "mcp_source_contracts.jsonl"
    manifest_path = output_dir / "mcp_source_contracts.manifest.json"
    corpus_path.write_bytes(corpus)
    manifest = {
        "schema": SCHEMA,
        "status": "PREPARATION_ONLY",
        "independent_evaluation": "NOT_ESTABLISHED",
        "source_scope": {
            "registration_wiring": "src/mind_mem/mcp/server.py",
            "compatibility_shim": "src/mind_mem/mcp_server.py",
            "tool_modules": "src/mind_mem/mcp/tools/*.py",
        },
        "source_files": source_entries,
        "generator": {"path": "train/build_source_corpus.py", "sha256": _sha256(Path(__file__).read_bytes())},
        "output": {
            "path": corpus_path.name,
            "bytes": len(corpus),
            "sha256": _sha256(corpus),
            "records": len(records),
        },
        "counts": {"registered_tools": len(registrations), "records": len(records), "families": len({r["family"] for r in records})},
        "limits": {
            "max_source_bytes": MAX_SOURCE_BYTES,
            "max_source_files": MAX_SOURCE_FILES,
            "max_docstring_bytes": MAX_DOCSTRING_BYTES,
            "max_message_bytes": MAX_MESSAGE_BYTES,
            "max_records": MAX_RECORDS,
        },
        "coverage": "registered MCP symbols in registration order; one contract record per symbol",
    }
    manifest_path.write_bytes(_canonical_bytes(manifest))
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _prepare_output(args.output_dir, args.repo_root)
        manifest = build(args.repo_root, args.output_dir)
    except (OSError, SourceCorpusError) as exc:
        print(f"source corpus refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps({"status": "PREPARATION_ONLY", "records": manifest["counts"]["records"], "output": str(args.output_dir)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
