"""Regression tests for the version-consistency gate.

The gate compared only the readings that succeeded, so with two of the
three sources unresolvable exactly one version remained and the mismatch
branch was arithmetically unreachable: the checker printed "NOT FOUND"
twice and then "All versions consistent" with exit 0.
"""

from __future__ import annotations

from pathlib import Path

from mind_mem.check_version import main


def _write_all(root: Path, pyproject: str, init: str, changelog: str) -> None:
    (root / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{pyproject}"\n', encoding="utf-8")
    (root / "src" / "mind_mem").mkdir(parents=True)
    (root / "src" / "mind_mem" / "__init__.py").write_text(f'__version__ = "{init}"\n', encoding="utf-8")
    (root / "CHANGELOG.md").write_text(f"# Changelog\n\n## [{changelog}] - 2026-01-01\n", encoding="utf-8")


class TestVersionGate:
    def test_unresolved_sources_are_an_error_not_a_consensus(self, tmp_path, monkeypatch, capsys) -> None:
        (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [0.10.2] - 2026-01-01\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        rc = main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "All versions consistent" not in out
        assert "pyproject.toml" in out.split("ERROR")[1]

    def test_consistent_versions_pass(self, tmp_path, monkeypatch, capsys) -> None:
        _write_all(tmp_path, "1.2.3", "1.2.3", "1.2.3")
        monkeypatch.chdir(tmp_path)
        assert main() == 0
        assert "All versions consistent (1.2.3)" in capsys.readouterr().out

    def test_mismatch_still_fails(self, tmp_path, monkeypatch, capsys) -> None:
        _write_all(tmp_path, "1.2.3", "1.2.4", "1.2.3")
        monkeypatch.chdir(tmp_path)
        assert main() == 1
        assert "Version mismatch" in capsys.readouterr().out
