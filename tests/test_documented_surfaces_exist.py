"""Documentation that names a symbol, a backend or an installable extra has
to name one that exists.

Two drifts this pins, both found by reading the docs against the code:

1. ``mind_mem.inbox`` told operators to ``pip install
   'mind-mem[multimodal]'``. ``pyproject.toml`` declares no such extra,
   so that command installs nothing and pip only warns.

2. ``quality_gate``'s docstring and ``docs/v3.11.0-implementation-plan.md``
   advertised a flat ``quality_gate_mode`` config key. The only reader
   is ``_get_quality_gate_mode``, which reads the nested
   ``cfg["quality_gate"]["mode"]``; the flat key is read by nothing.

Every assertion below fails on the pre-fix tree.

Dropped in the 5.0.0 module sweep: a third group pinned the
``hnsw_kind_index`` docs -- a ``mind_mem.hnsw_kind_index`` import path
that was really ``mind_mem.v4.hnsw_kind_index``, invented ``M=16,
efc=200`` tuning, and a claim that PQ codes fed the index. The sweep
deleted ``src/mind_mem/v4/hnsw_kind_index.py``, so those assertions
import and read a module that no longer exists and were removed with it.

deferred: ``docs/v4-release.md`` (a ``### hnsw_kind_index.py`` section
with a live import example, plus a Full-module-list table row) and
``train/HF_MODEL_CARD_v4.md`` still describe ``hnsw_kind_index`` as a
shipping surface, and so do the same twelve other v4 modules the sweep
deleted -- ``cognitive_kernel``, ``surprise_retrieval``, ``block_kinds``,
``block_metadata``, ``kind_summaries``, ``embedding_pipeline``, ``pq``,
``backpressure``, ``health``, ``logging_context``, ``v4.observability``.
This file was their only test coverage. Not patched here: a
``hnsw_kind_index``-only edit would leave the same two documents making
the identical claim about eleven more modules, which reads as audited
when it is not. Upgrade path: one pass over both documents against the
surviving ``src/mind_mem/v4`` tree, then re-pin the result here.
(``CLAUDE.md`` is already correct -- it lists these modules as removed.)
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
# 1. inbox optional extras
# ---------------------------------------------------------------------------


def _declared_extras() -> set[str]:
    # Via the compat shim: a bare `import tomllib` here is a
    # ModuleNotFoundError on 3.10, which requires-python still supports.
    from _toml_compat import declared_extras

    extras = declared_extras()
    if extras is None:
        pytest.skip("cannot read pyproject.toml (installed tree, or no TOML parser)")
    return extras


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
# 2. quality_gate config key
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


# ---------------------------------------------------------------------------
# 3. Docs that describe REMOVED surfaces must say so
# ---------------------------------------------------------------------------

#: v4 modules deleted by the 5.0.0 reachability sweep. A doc may still discuss
#: them -- release notes are a historical record and rewriting one falsifies it
#: -- but it must not read as a guide to a live API.
_REMOVED_V4_SURFACES = (
    "cognitive_kernel",
    "surprise_retrieval",
    "block_kinds",
    "block_metadata",
    "kind_summaries",
    "embedding_pipeline",
    "pq",
    "hnsw_kind_index",
    "backpressure",
    "health",
    "logging_context",
)


class TestRemovedV4SurfacesAreMarkedHistorical:
    """The replacement pin for the deleted ``TestHnswKindIndexDocs`` group.

    That group asserted documentation against ``v4/hnsw_kind_index.py``. The
    module is gone, so four of its six tests could not survive -- but deleting
    the class outright left NOTHING pinning the drift it existed to catch, and
    ``docs/v4-release.md`` still carries 53 references to removed surfaces
    including a runnable import example.

    The honest pin is not "no doc may name these" (that would force rewriting a
    release record) but "a doc that names them must not present them as live".
    """

    def test_the_v4_release_note_is_marked_historical(self) -> None:
        doc = _read("docs/v4-release.md")
        head = doc[:1600]
        assert "HISTORICAL" in head, "v4-release.md must announce itself as a historical record"
        assert "5.0.0" in head, "the banner must name the release that removed these surfaces"

    def test_the_banner_names_every_surface_the_page_still_documents(self) -> None:
        """A banner that omits a surface the page teaches is worse than none."""
        doc = _read("docs/v4-release.md")
        head, body = doc[:1600], doc[1600:]
        missing = [n for n in _REMOVED_V4_SURFACES if n in body and n not in head]
        assert not missing, f"documented but not named as removed in the banner: {missing}"

    def test_no_removed_surface_is_importable(self) -> None:
        """Pins the claim the banner makes, against the tree rather than prose."""
        import importlib.util

        alive = [n for n in _REMOVED_V4_SURFACES if importlib.util.find_spec(f"mind_mem.v4.{n}") is not None]
        assert not alive, f"banner says these were removed, but they import: {alive}"

    def test_live_docs_do_not_teach_a_removed_surface(self) -> None:
        """The non-historical docs are held to the stricter rule.

        ``docs/v4-release.md`` and the frozen v4 model card are exempt: both are
        records of a shipped artifact. Everything else is read as current.
        """
        import glob
        import os

        exempt = {"docs/v4-release.md"}
        offenders: list[str] = []
        for path in sorted(glob.glob(str(_REPO / "docs" / "*.md"))):
            rel = "docs/" + os.path.basename(path)
            if rel in exempt:
                continue
            text = _read(rel)
            for name in _REMOVED_V4_SURFACES:
                if f"from mind_mem.v4.{name} import" in text or f"mind_mem.v4.{name}(" in text:
                    offenders.append(f"{rel}:{name}")
        assert not offenders, f"live docs teach removed surfaces: {offenders}"
