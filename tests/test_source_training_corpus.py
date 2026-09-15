"""Controls for the source-only MCP contract preparation artifact."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "train" / "build_source_corpus.py"


def _run(repo: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BUILDER), "--repo-root", str(repo), "--output-dir", str(output)],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )


def _manifest(output: Path) -> dict[str, object]:
    return json.loads((output / "mcp_source_contracts.manifest.json").read_text(encoding="utf-8"))


def _records(output: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (output / "mcp_source_contracts.jsonl").read_text(encoding="utf-8").splitlines() if line]


def _copy_source_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    shutil.copytree(REPO / "src", checkout / "src")
    return checkout


def test_current_registered_surface_is_source_bound_and_complete(tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = _run(REPO, output)
    assert result.returncode == 0, result.stdout + result.stderr

    manifest = _manifest(output)
    records = _records(output)
    assert manifest["status"] == "PREPARATION_ONLY"
    assert manifest["independent_evaluation"] == "NOT_ESTABLISHED"
    assert manifest["counts"] == {"families": 107, "records": 107, "registered_tools": 107}
    assert len(records) == 107
    assert len({record["family"] for record in records}) == 107
    assert all(record["status"] == "PREPARATION_ONLY" for record in records)
    assert all(record["provenance"]["source_sha256"] for record in records)

    persona = next(record for record in records if record["tool_name"] == "recall")
    signature = json.loads(persona["messages"][1]["content"].split("Parameters: ", 1)[1].split("\n", 1)[0])
    assert any(parameter["kind"] == "keyword_only" for parameter in signature)
    assert any(parameter["name"] == "block_id" for parameter in signature)


def test_async_definition_shape_is_serialized_without_execution(tmp_path: Path) -> None:
    checkout = _copy_source_checkout(tmp_path)
    public = checkout / "src/mind_mem/mcp/tools/public.py"
    source = public.read_text(encoding="utf-8")
    public.write_text(source.replace("def recall(\n", "async def recall(\n", 1), encoding="utf-8")
    output = tmp_path / "async"
    result = _run(checkout, output)
    assert result.returncode == 0, result.stdout + result.stderr
    record = next(item for item in _records(output) if item["tool_name"] == "recall")
    assert json.loads(record["messages"][1]["content"].split("Parameters: ", 1)[1].split("\n", 1)[0])
    metadata = record["messages"][1]["content"].split("Signature metadata: ", 1)[1].split("\n", 1)[0]
    assert json.loads(metadata) == {"async": True}


def test_two_runs_have_identical_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    assert _run(REPO, first).returncode == 0
    assert _run(REPO, second).returncode == 0
    assert (first / "mcp_source_contracts.jsonl").read_bytes() == (second / "mcp_source_contracts.jsonl").read_bytes()
    assert (first / "mcp_source_contracts.manifest.json").read_bytes() == (second / "mcp_source_contracts.manifest.json").read_bytes()


def test_registered_source_change_changes_binding_and_output(tmp_path: Path) -> None:
    checkout = _copy_source_checkout(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    assert _run(checkout, first).returncode == 0
    target = checkout / "src/mind_mem/mcp/tools/public.py"
    original = target.read_text(encoding="utf-8")
    target.write_text(original.replace("Unified retrieval entry point.", "Changed source contract."), encoding="utf-8")
    assert _run(checkout, second).returncode == 0
    first_manifest, second_manifest = _manifest(first), _manifest(second)
    assert first_manifest["output"]["sha256"] != second_manifest["output"]["sha256"]
    first_sha = next(item["sha256"] for item in first_manifest["source_files"] if item["path"].endswith("tools/public.py"))
    second_sha = next(item["sha256"] for item in second_manifest["source_files"] if item["path"].endswith("tools/public.py"))
    assert first_sha != second_sha


def test_missing_registered_symbol_refuses_without_partial_output(tmp_path: Path) -> None:
    checkout = _copy_source_checkout(tmp_path)
    public = checkout / "src/mind_mem/mcp/tools/public.py"
    source = public.read_text(encoding="utf-8")
    public.write_text(source.replace("mcp.tool(recall)", "mcp.tool(missing_symbol)"), encoding="utf-8")
    output = tmp_path / "refused"
    result = _run(checkout, output)
    assert result.returncode == 2
    assert "missing_symbol" in result.stderr
    assert list(output.iterdir()) == []


def test_duplicate_registration_refuses_without_partial_output(tmp_path: Path) -> None:
    checkout = _copy_source_checkout(tmp_path)
    public = checkout / "src/mind_mem/mcp/tools/public.py"
    source = public.read_text(encoding="utf-8")
    public.write_text(source.replace("mcp.tool(recall)\n", "mcp.tool(recall)\n    mcp.tool(recall)\n", 1), encoding="utf-8")
    output = tmp_path / "refused"
    result = _run(checkout, output)
    assert result.returncode == 2
    assert "ambiguous MCP tool name 'recall'" in result.stderr
    assert list(output.iterdir()) == []


def test_undocumented_registered_symbol_refuses_without_partial_output(tmp_path: Path) -> None:
    checkout = _copy_source_checkout(tmp_path)
    public = checkout / "src/mind_mem/mcp/tools/public.py"
    source = public.read_text(encoding="utf-8")
    public.write_text(source.replace('"""Unified retrieval entry point.', '42\n    """Unified retrieval entry point.', 1), encoding="utf-8")
    output = tmp_path / "refused"
    result = _run(checkout, output)
    assert result.returncode == 2
    assert "docstring is not a literal" in result.stderr
    assert list(output.iterdir()) == []


@pytest.mark.parametrize(
    ("label", "target", "rewrite", "needle"),
    [
        (
            "tool-keyword",
            "public",
            lambda server, public: public.replace("mcp.tool(recall)\n", "mcp.tool(recall, name='renamed')\n", 1),
            "keyword options",
        ),
        (
            "wrong-receiver",
            "server",
            lambda server, public: server.replace("_tools_recall.register(mcp)", "_tools_recall.register(other_server)", 1),
            "receiver is not mcp",
        ),
        (
            "conditional-register",
            "server",
            lambda server, public: server.replace("_tools_recall.register(mcp)", "if True:\n    _tools_recall.register(mcp)", 1),
            "nested or conditional register",
        ),
        (
            "decorator-tool",
            "public",
            lambda server, public: public.replace("@mcp_tool_observe\ndef recall(\n", "@mcp.tool\n@mcp_tool_observe\ndef recall(\n", 1),
            "decorator mcp.tool",
        ),
    ],
)
def test_unsupported_registration_shapes_refuse(tmp_path: Path, label: str, target: str, rewrite: object, needle: str) -> None:
    checkout = _copy_source_checkout(tmp_path / label)
    server_path = checkout / "src/mind_mem/mcp/server.py"
    public_path = checkout / "src/mind_mem/mcp/tools/public.py"
    server, public = server_path.read_text(encoding="utf-8"), public_path.read_text(encoding="utf-8")
    rewritten = rewrite(server, public)  # type: ignore[operator]
    if target == "server":
        server_path.write_text(rewritten, encoding="utf-8")
    else:
        public_path.write_text(rewritten, encoding="utf-8")
    output = tmp_path / label / "refused"
    result = _run(checkout, output)
    assert result.returncode == 2
    assert needle in result.stderr
    assert list(output.iterdir()) == []


def test_oversized_source_refuses_before_unbounded_read(tmp_path: Path) -> None:
    checkout = _copy_source_checkout(tmp_path)
    source = checkout / "src/mind_mem/mcp_server.py"
    source.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    output = tmp_path / "refused"
    result = _run(checkout, output)
    assert result.returncode == 2
    assert "exceeds" in result.stderr
    assert list(output.iterdir()) == []


def test_nonempty_output_is_never_clobbered(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep")
    result = _run(REPO, output)
    assert result.returncode == 2
    assert sentinel.read_bytes() == b"keep"
    assert not (output / "mcp_source_contracts.jsonl").exists()


def test_symlinked_tool_source_refuses_before_read(tmp_path: Path) -> None:
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlink support unavailable")
    checkout = _copy_source_checkout(tmp_path)
    target = checkout / "src/mind_mem/mcp/tools/public.py"
    real = checkout / "src/mind_mem/mcp/tools/public-real.py"
    target.rename(real)
    try:
        target.symlink_to(real.name)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    output = tmp_path / "refused"
    result = _run(checkout, output)
    assert result.returncode == 2
    assert "symlinked" in result.stderr
    assert list(output.iterdir()) == []


def test_builder_is_stdlib_only_and_has_no_legacy_pipeline_import() -> None:
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    imported = {alias.name.split(".", 1)[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported.update(node.module.split(".", 1)[0] if node.module else "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert imported <= {"__future__", "argparse", "ast", "hashlib", "json", "os", "sys", "dataclasses", "pathlib", "typing"}
    assert "build_corpus" not in BUILDER.read_text(encoding="utf-8")


def test_source_hash_in_manifest_matches_exact_bytes(tmp_path: Path) -> None:
    output = tmp_path / "out"
    assert _run(REPO, output).returncode == 0
    manifest = _manifest(output)
    for item in manifest["source_files"]:
        data = (REPO / item["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
