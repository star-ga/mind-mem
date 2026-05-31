"""v4.0 prep — per-tenant audit chain façade."""

from __future__ import annotations

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
