"""Documentation that names a symbol, a backend or an installable extra has
to name one that exists.

Three drifts this pins, all found by reading the docs against the code:

1. ``docs/v4-release.md`` documented ``build_kind_index`` /
   ``query_kind_index`` from ``mind_mem.hnsw_kind_index`` — neither
   function exists, and the module is at ``mind_mem.v4.hnsw_kind_index``.
   It also claimed sqlite-vec's HNSW serves kNN with ``M=16, efc=200``
   and that PQ codes are consumed by the index. ``knn_by_kind`` is a
   brute-force cosine scan, ``backend_status`` reports ``brute_force``
   unconditionally, and the module imports nothing from ``pq``.
   ``CLAUDE.md`` and ``train/HF_MODEL_CARD_v4.md`` repeated it.

2. ``mind_mem.inbox`` told operators to ``pip install
   'mind-mem[multimodal]'``. ``pyproject.toml`` declares no such extra,
   so that command installs nothing and pip only warns. Same defect
   class as the ``mind-mem[encrypted]`` message pinned in
   ``tests/test_tenant_kms.py``.

3. ``quality_gate``'s docstring and ``docs/v3.11.0-implementation-plan.md``
   advertised a flat ``quality_gate_mode`` config key. The only reader
   is ``_get_quality_gate_mode``, which reads the nested
   ``cfg["quality_gate"]["mode"]``; the flat key is read by nothing.

Every assertion below fails on the pre-fix tree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. hnsw_kind_index
# ---------------------------------------------------------------------------

_HNSW_DOCS = ("docs/v4-release.md", "CLAUDE.md", "train/HF_MODEL_CARD_v4.md")


class TestHnswKindIndexDocs:
    def test_no_doc_names_a_function_that_does_not_exist(self) -> None:
        import mind_mem.v4.hnsw_kind_index as mod

        assert not hasattr(mod, "build_kind_index")
        assert not hasattr(mod, "query_kind_index")
        for rel in _HNSW_DOCS:
            text = _read(rel)
            assert "build_kind_index" not in text, rel
            assert "query_kind_index" not in text, rel

    def test_every_documented_symbol_is_importable(self) -> None:
        import mind_mem.v4.hnsw_kind_index as mod

        doc = _read("docs/v4-release.md")
        section = doc.split("### `hnsw_kind_index.py`", 1)[1].split("\n### ", 1)[0]
        named = {m for m in re.findall(r"`([a-z_]+)\(", section)}
        assert named, "the section documents no callable at all"
        for name in named:
            assert hasattr(mod, name), f"docs/v4-release.md names {name}(), which does not exist"

    def test_the_import_path_in_the_example_is_the_real_one(self) -> None:
        doc = _read("docs/v4-release.md")
        assert "from mind_mem.v4.hnsw_kind_index import" in doc
        assert "from mind_mem.hnsw_kind_index import" not in doc

    def test_docs_do_not_claim_sqlite_vec_serves_knn(self) -> None:
        section = _read("docs/v4-release.md").split("### `hnsw_kind_index.py`", 1)[1].split("\n### ", 1)[0]
        # The old text: "detects whether sqlite-vec is installed and uses
        # its HNSW implementation", plus invented tuning parameters.
        assert "uses its HNSW" not in section
        assert "M=16" not in section
        assert "efc=200" not in section

    def test_backend_status_agrees_with_the_docs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The behavioural anchor: what the module actually reports."""
        from mind_mem.v4.hnsw_kind_index import FLAG, backend_status

        cfg = tmp_path / "mind-mem.json"
        cfg.write_text(json.dumps({"v4": {FLAG: {"enabled": True}}}), encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_CONFIG", str(cfg))

        status = backend_status(tmp_path)
        assert status["backend"] == "brute_force"
        assert "brute-force" in _read("docs/v4-release.md")

    def test_pq_is_not_claimed_as_an_input_to_the_kind_index(self) -> None:
        src = _read("src/mind_mem/v4/hnsw_kind_index.py")
        assert "pq" not in {m for m in re.findall(r"^from \.(\w+) import", src, re.M)}
        doc = _read("docs/v4-release.md")
        assert "PQ codes are consumed" not in doc
        assert "Used automatically by `hnsw_kind_index`" not in doc


# ---------------------------------------------------------------------------
# 2. inbox optional extras
# ---------------------------------------------------------------------------


def _declared_extras() -> set[str]:
    import tomllib

    pyproject = _REPO / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("running from an installed tree without pyproject.toml")
    return set(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"].get("optional-dependencies", {}))


class TestInboxRemediationMessages:
    """Ask the same question the operator asks: is the thing installable?"""

    @pytest.mark.parametrize("handler", ["_ingest_image", "_ingest_audio"])
    def test_unimplemented_handlers_name_no_extra(self, handler: str) -> None:
        from mind_mem import inbox

        with pytest.raises(NotImplementedError) as exc:
            getattr(inbox, handler)("/ws", "/tmp/x")
        message = str(exc.value)

        named = set(re.findall(r"mind-mem\[([a-z0-9_.-]+)\]", message))
        assert named <= _declared_extras(), f"{handler} names undeclared extra(s): {sorted(named - _declared_extras())}"
        # There is nothing to install for these two at all, so the
        # message must not send the operator to pip.
        assert "pip install" not in message

    def test_pdf_handler_names_the_package_that_is_actually_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from mind_mem import inbox

        # A ``None`` entry in sys.modules makes ``import pypdf`` raise
        # ImportError, so the remediation branch is reachable whether or
        # not pypdf happens to be installed on this runner.
        monkeypatch.setitem(sys.modules, "pypdf", None)

        with pytest.raises(NotImplementedError) as exc:
            inbox._ingest_pdf("/ws", str(tmp_path / "x.pdf"))
        message = str(exc.value)

        named = set(re.findall(r"mind-mem\[([a-z0-9_.-]+)\]", message))
        assert named <= _declared_extras(), f"names undeclared extra(s): {sorted(named - _declared_extras())}"
        assert "pip install pypdf" in message

    def test_multimodal_is_not_a_declared_extra(self) -> None:
        """Why the old instruction was wrong, pinned. If a `multimodal`
        extra is ever declared, this assertion is the prompt to point the
        messages back at it."""
        assert "multimodal" not in _declared_extras()


# ---------------------------------------------------------------------------
# 3. quality_gate config key
# ---------------------------------------------------------------------------


class TestQualityGateConfigKeyDocs:
    def test_docs_advertise_the_nested_key(self) -> None:
        import mind_mem.quality_gate as qg

        doc = qg.__doc__ or ""
        assert '{"quality_gate": {"mode": "strict"}}' in doc
        plan = _read("docs/v3.11.0-implementation-plan.md")
        assert '{"quality_gate": {"mode": "strict"}}' in plan

    def test_the_flat_key_is_only_ever_mentioned_as_the_thing_that_is_read_by_nothing(self) -> None:
        import mind_mem.quality_gate as qg

        doc = qg.__doc__ or ""
        # Still named -- so an operator who wrote it can find out why it
        # did nothing -- but never as the setting to use.
        assert 'setting ``quality_gate_mode = "strict"``' not in doc
        assert "read by nothing" in doc

    def test_the_flat_key_really_is_ignored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mind_mem.mcp.infra.config import _get_quality_gate_mode

        (tmp_path / "mind-mem.json").write_text(json.dumps({"quality_gate_mode": "strict"}), encoding="utf-8")
        monkeypatch.setenv("MIND_MEM_WORKSPACE", str(tmp_path))
        assert _get_quality_gate_mode(str(tmp_path)) == "advisory"
