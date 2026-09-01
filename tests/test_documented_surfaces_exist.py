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

#: v4 modules that were deleted by the 5.0.0 sweep and RESTORED for 5.0.1.
#: The sweep removed them as unreachable; the operator's ruling was that
#: "nothing imports it" was never sufficient grounds, and they are being wired
#: rather than deleted. This tuple now pins the RESTORATION, not the removal.
_RESTORED_V4_SURFACES = (
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
    "observability",
    "vocabulary",
)


class TestRestoredV4SurfacesAreImportable:
    """The inverse of what this class asserted in 5.0.0, and deliberately so.

    In 5.0.0 this pinned that these modules must NOT import — the sweep had
    deleted them. That sweep removed 13,594 lines on the grounds that nothing
    referenced them, which conflated "unreachable" with "worthless". Two of the
    44 were not even unreachable: ``session_summarizer`` had a shell caller in
    ``hooks/session-end.sh`` and a Python importer in ``bootstrap_corpus``.

    They are restored and being wired. This class now fails if one goes missing
    again, so a future cleanup cannot quietly re-run the same mistake.
    """

    def test_every_restored_v4_surface_imports(self) -> None:
        import importlib.util

        missing = [n for n in _RESTORED_V4_SURFACES if importlib.util.find_spec(f"mind_mem.v4.{n}") is None]
        assert not missing, (
            f"restored v4 modules are missing again: {missing}. They were deleted "
            "in 5.0.0 and restored by operator ruling; removing one needs that "
            "ruling reversed, not a reachability argument."
        )

    def test_the_v4_release_note_no_longer_claims_they_are_removed(self) -> None:
        """The 5.0.0 banner said these were removed. That is no longer true."""
        doc = _read("docs/v4-release.md")
        head = doc[:1800]
        assert "removed most of the modules named below" not in head, (
            "docs/v4-release.md still carries the 5.0.0 removal banner, but the modules are restored — the banner is now the false claim"
        )
