"""The shared-service pre-exclusion must scan everything the harness executes.

The mining rule collects two different sets from one commit:

* ``added_test_files`` — status ``A`` and ``tests/test_*.py``. Only this set
  was read by the ``SHARED_SERVICE_PATTERN`` exclusion.
* ``test_patch_paths`` — status ``A`` or ``M`` anywhere under ``tests/`` plus
  the root ``conftest.py``. This is the set the validation harness copies into
  the extracted tree, and pytest imports ``tests/conftest.py`` from it at
  collection time.

So a commit that added a benign ``tests/test_x.py`` and *modified*
``tests/conftest.py`` to add a psycopg fixture passed the exclusion — only the
added file's text had been read — and then ran that fixture against the live
Postgres on this host. Files added under ``tests/`` that are not ``test_*.py``
(``tests/conftest.py`` itself, ``tests/fixtures/*.py``) evaded it the same way.

That is the exact outcome the module docstring says the exclusion exists to
prevent: "not a benchmark, it is an outage".
"""

from __future__ import annotations

import os
import subprocess

import pytest

from mind_mem.bench import repo_task_mining as mining


def _run(cwd: str, *args: str) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8")


def _write(root: str, rel: str, text: str) -> None:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _seed(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    _run(root, "git", "init", "-q", "-b", "main")
    _run(root, "git", "config", "user.email", "noreply@star.ga")
    _run(root, "git", "config", "user.name", "STARGA Inc")
    _write(root, "src/toy/__init__.py", "def add(a, b):\n    return a - b\n")
    _write(root, "tests/conftest.py", "import pytest\n")
    _run(root, "git", "add", "-A")
    _run(root, "git", "commit", "-q", "-m", "feat: seed the toy package")


def _fix_commit(root: str, extra: dict[str, str]) -> None:
    _write(root, "src/toy/__init__.py", "def add(a, b):\n    return a + b\n")
    _write(root, "tests/test_add.py", "from toy import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    for rel, text in extra.items():
        _write(root, rel, text)
    _run(root, "git", "add", "-A")
    _run(root, "git", "commit", "-q", "-m", "fix(toy): addition subtracted")


CONFTEST_WITH_PG = (
    'import psycopg\nimport pytest\n\n\n@pytest.fixture\ndef db():\n    return psycopg.connect("postgresql://127.0.0.1:5432/mindmem")\n'
)


@pytest.fixture
def repo_with_pg_conftest(tmp_path) -> str:
    """A fix commit that adds a clean test and MODIFIES conftest to reach PG."""
    root = str(tmp_path / "pg_conftest")
    _seed(root)
    _fix_commit(root, {"tests/conftest.py": CONFTEST_WITH_PG})
    return root


@pytest.fixture
def repo_with_pg_helper(tmp_path) -> str:
    """A fix commit that ADDS a tests/ helper (not ``test_*.py``) reaching PG."""
    root = str(tmp_path / "pg_helper")
    _seed(root)
    _fix_commit(root, {"tests/fixtures/pg.py": "import psycopg\n"})
    return root


@pytest.fixture
def clean_repo(tmp_path) -> str:
    root = str(tmp_path / "clean")
    _seed(root)
    _fix_commit(root, {})
    return root


class TestExclusionCoversTheExecutedSet:
    """What is scanned must be what is run."""

    def test_modified_conftest_reaching_postgres_is_excluded(self, repo_with_pg_conftest) -> None:
        """Before the fix this commit was selected and its conftest executed."""
        selected, stats = mining.select_candidates(repo_with_pg_conftest, "HEAD", limit=10)
        assert stats.rule_matched == 1
        assert stats.excluded_shared_service == 1
        assert selected == []
        assert stats.excluded_detail[0]["reason"] == "shared_service:psycopg"

    def test_added_tests_helper_reaching_postgres_is_excluded(self, repo_with_pg_helper) -> None:
        """``tests/fixtures/pg.py`` is copied into the tree but is not ``test_*.py``."""
        selected, stats = mining.select_candidates(repo_with_pg_helper, "HEAD", limit=10)
        assert stats.excluded_shared_service == 1
        assert selected == []

    def test_a_clean_commit_is_still_selected(self, clean_repo) -> None:
        """The widened scan must not exclude everything."""
        selected, stats = mining.select_candidates(clean_repo, "HEAD", limit=10)
        assert stats.excluded_shared_service == 0
        assert [c.subject for c in selected] == ["fix(toy): addition subtracted"]
        assert selected[0].added_test_files == ("tests/test_add.py",)

    def test_scanned_set_is_the_patch_set(self, repo_with_pg_conftest) -> None:
        """The exclusion reads exactly the paths the harness lays down."""
        sha = subprocess.run(
            ["git", "-C", repo_with_pg_conftest, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        source = mining._test_patch_sources(repo_with_pg_conftest, sha, ("tests/conftest.py",))
        assert "psycopg" in source
