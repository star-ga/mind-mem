"""``mind-mem-connect`` — the join, and the three ways a join can go wrong.

The command writes one file, so the tests are about what it writes: that the
merged config is complete enough for the workspace to actually read the
federation (all four keys, not three), that a pre-existing config survives, that
a credential never leaves the file, and that an unusable URL is refused rather
than handed to a driver.

Every "the secret is absent" assertion is paired with proof the secret was
present in the source and that the check can see it — otherwise a redaction test
passes on a run where no password was ever supplied.
"""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from mind_mem.federation_connect import (
    DEFAULT_SCHEMA,
    POSTGRES_RECALL_BACKEND,
    ConnectError,
    build_federation_config,
    connect,
    main,
    redact_url,
)

PASSWORD = "hunter2-not-a-real-password"
DSN = f"postgresql://mind:{PASSWORD}@u1.internal:5432/mindmem"
REDIS = f"redis://:{PASSWORD}@u1.internal:6379/0"


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------


class TestTheMergedConfig:
    def test_a_dsn_sets_every_key_the_workspace_needs(self) -> None:
        """Three keys is the trap: without ``recall.backend`` recall reads an empty tree."""
        config = build_federation_config(None, dsn=DSN)

        assert config["block_store"] == {"backend": "postgres", "dsn": DSN, "schema": DEFAULT_SCHEMA}
        assert config["recall"]["backend"] == POSTGRES_RECALL_BACKEND

    def test_a_redis_url_turns_the_cache_on(self) -> None:
        """Wiring a URL into a cache nobody enabled would be a silent no-op."""
        config = build_federation_config(None, redis_url=REDIS)

        assert config["cache"]["redis_url"] == REDIS
        assert config["cache"]["enabled"] is True

    def test_an_explicit_cache_disable_is_respected(self) -> None:
        """``setdefault``, not overwrite: the operator's own choice stands."""
        config = build_federation_config({"cache": {"enabled": False}}, redis_url=REDIS)

        assert config["cache"]["enabled"] is False
        assert config["cache"]["redis_url"] == REDIS

    def test_unrelated_settings_survive_the_join(self) -> None:
        existing = {
            "governance_mode": "self_correcting",
            "limits": {"max_recall_results": 42},
            "recall": {"vector_enabled": True},
        }

        config = build_federation_config(existing, dsn=DSN, redis_url=REDIS)

        assert config["governance_mode"] == "self_correcting"
        assert config["limits"] == {"max_recall_results": 42}
        assert config["recall"]["vector_enabled"] is True, "an unrelated recall key was dropped"
        assert config["recall"]["backend"] == POSTGRES_RECALL_BACKEND

    def test_the_caller_s_config_is_not_mutated(self) -> None:
        existing: dict = {"recall": {"vector_enabled": True}}

        build_federation_config(existing, dsn=DSN)

        assert existing == {"recall": {"vector_enabled": True}}

    def test_a_reconnect_keeps_the_schema_it_already_had(self) -> None:
        """Naming only a new DSN must not silently relocate the corpus."""
        existing = {"block_store": {"backend": "postgres", "dsn": "postgresql://old/db", "schema": "tenant_b"}}

        config = build_federation_config(existing, dsn=DSN)

        assert config["block_store"]["schema"] == "tenant_b"

    def test_an_explicit_schema_wins(self) -> None:
        existing = {"block_store": {"backend": "postgres", "dsn": "postgresql://old/db", "schema": "tenant_b"}}

        config = build_federation_config(existing, dsn=DSN, schema="tenant_c")

        assert config["block_store"]["schema"] == "tenant_c"


class TestRefusals:
    def test_naming_nothing_is_refused(self) -> None:
        with pytest.raises(ConnectError, match="nothing to connect"):
            build_federation_config(None)

    @pytest.mark.parametrize("bad", ["file:///etc/passwd", "http://u1/db", "mysql://u1/db", "/tmp/db", ""])
    def test_a_dsn_that_is_not_postgres_is_refused(self, bad: str) -> None:
        """POSITIVE CONTROL: the same call with a real DSN succeeds."""
        with pytest.raises(ConnectError):
            build_federation_config(None, dsn=bad)

        assert build_federation_config(None, dsn=DSN)["block_store"]["dsn"] == DSN

    @pytest.mark.parametrize("bad", ["file:///etc/passwd", "http://u1", "postgresql://u1/db"])
    def test_a_redis_url_that_is_not_redis_is_refused(self, bad: str) -> None:
        with pytest.raises(ConnectError):
            build_federation_config(None, redis_url=bad)

        assert build_federation_config(None, redis_url=REDIS)["cache"]["redis_url"] == REDIS

    def test_a_refused_url_is_not_echoed_back_with_its_password(self) -> None:
        bad = f"mysql://mind:{PASSWORD}@u1/db"

        with pytest.raises(ConnectError) as caught:
            build_federation_config(None, dsn=bad)

        assert PASSWORD not in str(caught.value)
        assert "mysql" in str(caught.value), "the message must still say what was wrong"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_a_password_is_replaced(self) -> None:
        assert PASSWORD in DSN, "the fixture carries no password — the assertion below would be vacuous"

        assert PASSWORD not in redact_url(DSN)
        assert "***" in redact_url(DSN)

    def test_the_rest_of_the_url_survives(self) -> None:
        redacted = redact_url(DSN)

        assert redacted.startswith("postgresql://mind:***@u1.internal:5432")
        assert redacted.endswith("/mindmem")

    def test_a_url_with_no_password_is_unchanged(self) -> None:
        plain = "postgresql://u1.internal:5432/mindmem"

        assert redact_url(plain) == plain

    def test_an_empty_url_stays_empty(self) -> None:
        assert redact_url("") == ""


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class TestWriting:
    def test_the_config_is_created_and_readable_back(self, tmp_path) -> None:
        result = connect(str(tmp_path), dsn=DSN, redis_url=REDIS)

        assert result.written is True
        with open(result.config_path, encoding="utf-8") as fh:
            written = json.load(fh)
        assert written["block_store"]["dsn"] == DSN
        assert written["cache"]["redis_url"] == REDIS

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits do not exist on Windows; owner-only there is the directory ACL")
    def test_the_file_holding_a_password_is_owner_only_whatever_the_umask(self, tmp_path) -> None:
        """The mode must come from the writer, not from the caller's umask.

        Asserted under a fully permissive umask on purpose. A plain
        ``open(path, "w")`` would produce ``0666 & ~umask`` — world-readable
        here — and pass a version of this test run under a normal ``022``
        umask, which is how a credential file ends up readable in production
        while the suite stays green.
        """
        previous = os.umask(0o000)
        try:
            result = connect(str(tmp_path), dsn=DSN)
        finally:
            os.umask(previous)

        mode = stat.S_IMODE(os.stat(result.config_path).st_mode)
        assert mode == 0o600, f"the config holds a database password and is mode {mode:o}"

    def test_it_is_idempotent(self, tmp_path) -> None:
        first = connect(str(tmp_path), dsn=DSN, redis_url=REDIS)
        with open(first.config_path, encoding="utf-8") as fh:
            after_one = fh.read()

        connect(str(tmp_path), dsn=DSN, redis_url=REDIS)
        with open(first.config_path, encoding="utf-8") as fh:
            after_two = fh.read()

        assert after_one == after_two

    def test_a_dry_run_writes_nothing(self, tmp_path) -> None:
        """POSITIVE CONTROL: the identical call without dry_run does create it."""
        result = connect(str(tmp_path), dsn=DSN, dry_run=True)
        assert not os.path.exists(result.config_path)
        assert result.config["block_store"]["dsn"] == DSN, "the dry run computed nothing to show"

        connect(str(tmp_path), dsn=DSN)
        assert os.path.exists(result.config_path)

    def test_a_malformed_existing_config_is_not_overwritten(self, tmp_path) -> None:
        path = tmp_path / "mind-mem.json"
        path.write_text("{ this is not json", encoding="utf-8")

        with pytest.raises(ConnectError, match="refusing to overwrite"):
            connect(str(tmp_path), dsn=DSN)

        assert path.read_text(encoding="utf-8") == "{ this is not json"

    def test_a_non_object_config_is_not_overwritten(self, tmp_path) -> None:
        path = tmp_path / "mind-mem.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ConnectError, match="refusing to overwrite"):
            connect(str(tmp_path), dsn=DSN)

        assert path.read_text(encoding="utf-8") == "[1, 2, 3]"

    def test_no_temporary_file_is_left_behind(self, tmp_path) -> None:
        connect(str(tmp_path), dsn=DSN)

        leftovers = [n for n in os.listdir(tmp_path) if n.startswith(".mind-mem-connect-")]
        assert leftovers == []


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


class TestTheCommandLine:
    def test_credentials_come_from_the_environment_by_default(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("MIND_MEM_DSN", DSN)
        monkeypatch.setenv("MIND_MEM_REDIS_URL", REDIS)

        code = main(["--workspace", str(tmp_path)])

        assert code == 0
        written = json.loads((tmp_path / "mind-mem.json").read_text(encoding="utf-8"))
        assert written["block_store"]["dsn"] == DSN
        assert written["cache"]["redis_url"] == REDIS
        assert PASSWORD not in capsys.readouterr().out

    def test_the_printed_summary_never_carries_the_password(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.delenv("MIND_MEM_DSN", raising=False)
        monkeypatch.delenv("MIND_MEM_REDIS_URL", raising=False)

        main(["--workspace", str(tmp_path), "--dsn", DSN, "--redis-url", REDIS])
        out = capsys.readouterr().out

        assert PASSWORD in DSN, "the fixture carries no password — this assertion would be vacuous"
        assert PASSWORD not in out
        assert "u1.internal" in out, "the summary printed nothing identifying at all"

    def test_a_bad_url_exits_two_and_writes_nothing(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.delenv("MIND_MEM_DSN", raising=False)
        monkeypatch.delenv("MIND_MEM_REDIS_URL", raising=False)

        code = main(["--workspace", str(tmp_path), "--dsn", "file:///etc/passwd"])

        assert code == 2
        assert not (tmp_path / "mind-mem.json").exists()
        assert PASSWORD not in capsys.readouterr().out

    def test_naming_no_credential_at_all_exits_two(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.delenv("MIND_MEM_DSN", raising=False)
        monkeypatch.delenv("MIND_MEM_REDIS_URL", raising=False)

        assert main(["--workspace", str(tmp_path)]) == 2
        assert "nothing to connect" in capsys.readouterr().out

    def test_a_flag_beats_the_environment(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("MIND_MEM_DSN", "postgresql://from-env/db")

        main(["--workspace", str(tmp_path), "--dsn", DSN])
        capsys.readouterr()

        written = json.loads((tmp_path / "mind-mem.json").read_text(encoding="utf-8"))
        assert written["block_store"]["dsn"] == DSN

    def test_dry_run_reports_but_does_not_write(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setenv("MIND_MEM_DSN", DSN)

        code = main(["--workspace", str(tmp_path), "--dry-run"])
        out = capsys.readouterr().out

        assert code == 0
        assert "Dry run" in out
        assert not (tmp_path / "mind-mem.json").exists()


class TestTheScriptIsRegistered:
    def test_pyproject_declares_the_entry_point(self) -> None:
        """A command nobody can invoke is not a command.

        Read from the packaging metadata rather than asserted from memory, so a
        rename that forgets the script entry fails here instead of on a user's
        machine.
        """
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        text = (root / "pyproject.toml").read_text(encoding="utf-8")

        assert 'mind-mem-connect = "mind_mem.federation_connect:main"' in text
        assert text.count("mind-mem-connect =") == 1
