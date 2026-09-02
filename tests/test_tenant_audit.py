"""v4.0 prep — per-tenant audit chain façade."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mind_mem import tenant_audit


@pytest.fixture(autouse=True)
def _reset_between_tests() -> None:
    tenant_audit.reset()
    yield
    tenant_audit.reset()


class _FakeChain:
    """Minimal chain stand-in — records appended entries for assertion."""

    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id
        self.entries: list[dict] = []

    def append(self, operation: str, **kwargs) -> None:
        self.entries.append({"operation": operation, **kwargs})

    def verify(self) -> dict:
        return {"verified": True, "records": len(self.entries)}


@pytest.fixture
def fake_factory_installed() -> None:
    created: list[_FakeChain] = []

    def fake_factory(tenant_id: str, base_path: str) -> _FakeChain:
        chain = _FakeChain(tenant_id)
        created.append(chain)
        return chain

    tenant_audit.register_chain_factory(fake_factory)
    yield created
    # Clear the factory override so the next test gets the default.
    tenant_audit._factory = None  # type: ignore[attr-defined]


class TestGetChain:
    def test_rejects_empty_tenant_id(self, tmp_path: Path, fake_factory_installed) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            tenant_audit.get_chain("", base_path=str(tmp_path), root_secret=b"x" * 32)

    def test_rejects_short_secret(self, tmp_path: Path, fake_factory_installed) -> None:
        with pytest.raises(ValueError, match="root_secret"):
            tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"short")

    def test_returns_cached_chain_on_second_call(self, tmp_path: Path, fake_factory_installed) -> None:
        first = tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
        second = tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
        assert first is second  # cached — factory called once
        assert len(fake_factory_installed) == 1

    def test_different_tenants_get_independent_chains(self, tmp_path: Path, fake_factory_installed) -> None:
        a = tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
        b = tenant_audit.get_chain("globex", base_path=str(tmp_path), root_secret=b"x" * 32)
        assert a.chain is not b.chain
        assert a.genesis != b.genesis

    def test_genesis_deterministic_for_same_secret(self, tmp_path: Path, fake_factory_installed) -> None:
        """Same tenant + same root_secret → same genesis across resets."""
        first = tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"deterministic-secret-32bytes!")
        tenant_audit.reset()
        second = tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"deterministic-secret-32bytes!")
        assert first.genesis == second.genesis

    def test_different_secrets_produce_different_genesis(self, tmp_path: Path, fake_factory_installed) -> None:
        a = tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"a" * 32)
        tenant_audit.reset()
        b = tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"b" * 32)
        assert a.genesis != b.genesis

    def test_spec_change_refreshes_handle(self, tmp_path: Path, fake_factory_installed) -> None:
        """Changing the spec invalidates the cache → factory called again."""
        first = tenant_audit.get_chain(
            "acme",
            base_path=str(tmp_path),
            root_secret=b"x" * 32,
            spec=b"spec-v1",
        )
        second = tenant_audit.get_chain(
            "acme",
            base_path=str(tmp_path),
            root_secret=b"x" * 32,
            spec=b"spec-v2",
        )
        assert first.spec_hash != second.spec_hash
        assert len(fake_factory_installed) == 2


class TestListTenants:
    def test_empty_registry_returns_empty(self) -> None:
        assert tenant_audit.list_tenants() == []

    def test_after_init_returns_tenant_ids(self, tmp_path: Path, fake_factory_installed) -> None:
        for tid in ("acme", "globex", "initech"):
            tenant_audit.get_chain(tid, base_path=str(tmp_path), root_secret=b"x" * 32)
        assert tenant_audit.list_tenants() == ["acme", "globex", "initech"]


class TestVerifyTenant:
    def test_clean_chain_verifies(self, tmp_path: Path, fake_factory_installed) -> None:
        result = tenant_audit.verify_tenant(
            "acme",
            base_path=str(tmp_path),
            root_secret=b"x" * 32,
        )
        assert result["verified"] is True
        assert result["tenant_id"] == "acme"
        assert "genesis" in result
        assert "spec_hash" in result
        assert len(result["genesis"]) == 64  # hex of 32 bytes

    def test_chain_with_records_reports_count(self, tmp_path: Path, fake_factory_installed) -> None:
        handle = tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
        handle.chain.append("TEST_OP")
        handle.chain.append("TEST_OP")
        result = tenant_audit.verify_tenant(
            "acme",
            base_path=str(tmp_path),
            root_secret=b"x" * 32,
        )
        assert result["records"] == 2

    def test_verify_failure_returns_false(self, tmp_path: Path, fake_factory_installed) -> None:
        handle = tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
        # Corrupt the chain so verify() raises.
        handle.chain.verify = lambda: (_ for _ in ()).throw(RuntimeError("corrupted"))
        result = tenant_audit.verify_tenant(
            "acme",
            base_path=str(tmp_path),
            root_secret=b"x" * 32,
        )
        assert result["verified"] is False


class _TupleChain:
    """Chain stand-in with the real AuditChain contract: (is_valid, errors)."""

    def __init__(self, ok: bool) -> None:
        self._ok = ok

    def verify(self) -> tuple:
        return (self._ok, [] if self._ok else ["Line 3 (seq 3): prev_hash mismatch"])


class TestVerifyTenantPairResult:
    """A chain reporting (is_valid, errors) must not be read as truthy.

    Every non-empty tuple is truthy, so ``bool(result)`` reports
    ``(False, [...])`` — a tampered chain — as verified.
    """

    def test_pair_false_is_reported_unverified(self, tmp_path: Path) -> None:
        def factory(tenant_id: str, base_path: str) -> _TupleChain:
            return _TupleChain(ok=False)

        tenant_audit.register_chain_factory(factory)
        try:
            result = tenant_audit.verify_tenant("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
        finally:
            tenant_audit._factory = None  # type: ignore[attr-defined]
        assert result["verified"] is False

    def test_pair_true_is_reported_verified(self, tmp_path: Path) -> None:
        def factory(tenant_id: str, base_path: str) -> _TupleChain:
            return _TupleChain(ok=True)

        tenant_audit.register_chain_factory(factory)
        try:
            result = tenant_audit.verify_tenant("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
        finally:
            tenant_audit._factory = None  # type: ignore[attr-defined]
        assert result["verified"] is True


class TestVerifyTenantDefaultChain:
    """End-to-end over the real audit chain the default factory builds."""

    @staticmethod
    def _seed(tmp_path: Path, count: int):
        tenant_audit._factory = None  # type: ignore[attr-defined]  # exercise _default_factory
        handle = tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
        for i in range(count):
            handle.chain.append("create_block", f"note-{i}.md", agent="tester", reason="seed")
        return handle

    @staticmethod
    def _chain_file(tmp_path: Path) -> Path:
        return next(tmp_path.rglob("chain.jsonl"))

    def test_clean_chain_reports_record_count(self, tmp_path: Path) -> None:
        self._seed(tmp_path, 2)
        result = tenant_audit.verify_tenant("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
        assert result["verified"] is True
        assert result["records"] == 2

    def test_broken_prev_hash_linkage_is_not_verified(self, tmp_path: Path) -> None:
        self._seed(tmp_path, 3)
        path = self._chain_file(tmp_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[2])
        entry["prev_hash"] = "0" * 64
        lines[2] = json.dumps(entry, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Ground truth: the chain itself knows it is broken.
        assert tenant_audit.get_chain("acme", base_path=str(tmp_path), root_secret=b"x" * 32).chain.verify()[0] is False

        result = tenant_audit.verify_tenant("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
        assert result["verified"] is False
        assert result["records"] == 3


class _DictChain:
    """Chain stand-in whose ``verify()`` answers a summary dict."""

    def __init__(self, summary: dict) -> None:
        self._summary = summary

    def verify(self) -> dict:
        return dict(self._summary)


def _verify_with(summary: dict, tmp_path: Path) -> dict:
    def factory(tenant_id: str, base_path: str) -> _DictChain:
        return _DictChain(summary)

    tenant_audit.register_chain_factory(factory)
    try:
        return tenant_audit.verify_tenant("acme", base_path=str(tmp_path), root_secret=b"x" * 32)
    finally:
        tenant_audit._factory = None  # type: ignore[attr-defined]


class TestVerifyTenantUnknownDictShape:
    """A summary dict with no ``verified`` key must fail CLOSED.

    ``verify_tenant`` is the per-tenant compliance-export surface, so the one
    direction it must never fail in is "nothing checked this, call it verified".
    A registered chain impl that returns ``{"records": 5}`` — a different
    vocabulary, a renamed key, a partially-built adapter — used to be read as
    verified by the ``.get("verified", True)`` default.
    """

    def test_a_dict_without_the_key_is_not_verified(self, tmp_path: Path) -> None:
        result = _verify_with({"records": 5}, tmp_path)
        assert result["verified"] is False
        # The count is still reported: refusing the verdict must not also
        # discard the detail an operator needs to go looking.
        assert result["records"] == 5

    def test_a_dict_saying_true_is_still_verified(self, tmp_path: Path) -> None:
        """Positive control — the refusal above discriminates, it is not blanket.

        Without this, ``verified is False`` would also pass on an implementation
        that had simply stopped returning True for anything.
        """
        result = _verify_with({"verified": True, "records": 5}, tmp_path)
        assert result["verified"] is True
        assert result["records"] == 5

    def test_a_dict_saying_false_is_not_verified(self, tmp_path: Path) -> None:
        assert _verify_with({"verified": False, "records": 5}, tmp_path)["verified"] is False

    def test_an_empty_dict_is_not_verified(self, tmp_path: Path) -> None:
        assert _verify_with({}, tmp_path)["verified"] is False
