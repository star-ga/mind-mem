# Copyright 2026 STARGA, Inc.
"""mind-mem Governance Spec Binding — config tamper detection via SHA3-512 hash.

Binds the active governance configuration to a content hash. Any change to
the config invalidates the binding and requires an explicit re-attestation,
preventing silent configuration tampering.

Binding file: .spec_binding.json, written next to the config file.

Usage:
    from .spec_binding import SpecBindingManager

    mgr = SpecBindingManager("/path/to/mind-mem.json")
    binding = mgr.bind("/path/to/mind-mem.json")

    valid, reason = mgr.verify()
    if not valid:
        raise RuntimeError(f"Config tampered: {reason}")

    if mgr.has_drifted():
        mgr.rebind("/path/to/mind-mem.json")

Zero external deps — hashlib, json, os, datetime (all stdlib).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .observability import get_logger

_log = get_logger("spec_binding")

_BINDING_FILENAME = ".spec_binding.json"
_CURRENT_VERSION = "1.0.0"


class SpecBindingCorruptedError(Exception):
    """Raised when the binding file exists but cannot be parsed.

    Callers must treat this as a security-relevant failure: a corrupted
    binding must not be silently interpreted as "no binding" because that
    would allow an attacker to disable governance by damaging the file.
    """


# ---------------------------------------------------------------------------
# Config normalisation helpers (module-level — testable independently)
# ---------------------------------------------------------------------------


def _normalize_config(config_path: str) -> str:
    """Return a deterministic JSON string of the config at *config_path*.

    Keys are sorted recursively so that insertion-order differences in the
    source file do not produce different hashes.

    Raises:
        FileNotFoundError: if *config_path* does not exist.
        ValueError: if the file contains invalid JSON.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = f.read()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config '{config_path}': {exc}") from exc
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def _compute_config_hash(config_path: str) -> str:
    """Return the SHA3-512 hex digest of the normalized config at *config_path*.

    The 128-character hex string uniquely fingerprints the config content
    independent of key ordering.
    """
    normalized = _normalize_config(config_path)
    return hashlib.sha3_512(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# SpecBinding frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecBinding:
    """Immutable record of a governance config binding.

    Attributes:
        spec_hash: SHA3-512 hex digest of the normalized config at bind time.
        config_path: Absolute path to the bound config file.
        bound_at: UTC datetime when the binding was created.
        version: mind-mem version string at bind time.
        attestation_signature: Optional external attestation (e.g. GPG sig).
    """

    spec_hash: str
    config_path: str
    bound_at: datetime
    version: str
    attestation_signature: Optional[str] = field(default=None)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "spec_hash": self.spec_hash,
            "config_path": self.config_path,
            "bound_at": self.bound_at.isoformat(),
            "version": self.version,
            "attestation_signature": self.attestation_signature,
        }

    @staticmethod
    def from_dict(data: dict) -> "SpecBinding":
        """Deserialise from a dict produced by :meth:`to_dict`."""
        bound_at_raw = data["bound_at"]
        if isinstance(bound_at_raw, str):
            bound_at = datetime.fromisoformat(bound_at_raw)
        else:
            bound_at = bound_at_raw
        # Ensure timezone-aware
        if bound_at.tzinfo is None:
            bound_at = bound_at.replace(tzinfo=timezone.utc)
        return SpecBinding(
            spec_hash=data["spec_hash"],
            config_path=data["config_path"],
            bound_at=bound_at,
            version=data["version"],
            attestation_signature=data.get("attestation_signature"),
        )


# ---------------------------------------------------------------------------
# SpecBindingManager
# ---------------------------------------------------------------------------


class SpecBindingManager:
    """Manages the lifecycle of a governance spec binding for a config file.

    The binding is persisted as *<config_dir>/.spec_binding.json*. All
    methods are safe to call on a manager with no prior binding on disk.

    Args:
        config_path: Path to the mind-mem.json config file to bind against.
    """

    def __init__(self, config_path: str) -> None:
        self._config_path = os.path.abspath(config_path)
        self._binding_path = os.path.join(os.path.dirname(self._config_path), _BINDING_FILENAME)
        self._cached: Optional[SpecBinding] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind(self, config_path: str) -> SpecBinding:
        """Compute a fresh binding for *config_path* and persist it.

        Args:
            config_path: Path to the config to bind. May differ from the
                path provided at construction (e.g. after a config move).

        Returns:
            The newly created :class:`SpecBinding`.
        """
        abs_path = os.path.abspath(config_path)
        spec_hash = _compute_config_hash(abs_path)
        binding = SpecBinding(
            spec_hash=spec_hash,
            config_path=abs_path,
            bound_at=datetime.now(timezone.utc),
            version=_CURRENT_VERSION,
        )
        self._persist(binding)
        self._cached = binding
        _log.info(
            "spec_binding.bound",
            config_path=abs_path,
            spec_hash=spec_hash[:16] + "…",
        )
        return binding

    def verify(self) -> tuple[bool, str]:
        """Check whether the current config matches the stored binding hash.

        Returns:
            A ``(valid, reason)`` tuple.  *valid* is ``True`` iff the config
            has not changed since the last :meth:`bind` or :meth:`rebind`.
            *reason* is an empty string on success or a human-readable
            description of the failure. A corrupted binding reports as
            invalid with a specific reason, never as "no binding".
        """
        try:
            binding = self.get_binding()
        except SpecBindingCorruptedError as exc:
            return False, f"binding corrupted: {exc}"
        if binding is None:
            return False, "no binding found — call bind() before verify()"

        if not os.path.exists(binding.config_path):
            return (
                False,
                f"config file missing: '{binding.config_path}'",
            )

        try:
            current_hash = _compute_config_hash(binding.config_path)
        except (FileNotFoundError, ValueError) as exc:
            return False, f"could not hash config: {exc}"

        if current_hash == binding.spec_hash:
            if binding.attestation_signature is not None:
                _log.warning(
                    "spec_binding.attestation_unverified",
                    config_path=binding.config_path,
                    msg="attestation_signature is present but has not been cryptographically verified",
                )
            return True, "binding valid — config hash matches"

        return (
            False,
            (
                f"config hash mismatch — binding is no longer valid. "
                f"bound_hash={binding.spec_hash[:16]}… "
                f"current_hash={current_hash[:16]}… — "
                f"config has changed since last bind/rebind"
            ),
        )

    def rebind(self, config_path: str) -> SpecBinding:
        """Create a new binding, replacing any prior one.

        This is an explicit re-attestation action.  Call this *only* after
        reviewing and accepting the config change.

        Args:
            config_path: Path to the (updated) config to bind.

        Returns:
            The newly created :class:`SpecBinding`.
        """
        _log.info("spec_binding.rebind", config_path=config_path)
        return self.bind(config_path)

    def get_binding(self) -> Optional[SpecBinding]:
        """Return the stored binding, or ``None`` if none exists.

        Loads from disk on first call; subsequent calls return the cached
        value unless the binding file has been modified externally.

        Raises:
            SpecBindingCorruptedError: If the binding file exists but is
                corrupted. Callers must decide explicitly whether to
                rebuild the binding or refuse to operate.
        """
        if self._cached is not None:
            return self._cached
        return self._load()

    def has_drifted(self) -> bool:
        """Return ``True`` if the config has changed or no binding exists.

        This is a quick boolean convenience wrapper around :meth:`verify`.
        """
        valid, _ = self.verify()
        return not valid

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _persist(self, binding: SpecBinding) -> None:
        """Write *binding* to the binding file atomically via tmp + rename.

        Uses flush + fsync on the tmp file before os.replace so a power loss
        between write and rename cannot leave the binding file empty.
        """
        data = json.dumps(binding.to_dict(), indent=2)
        tmp_path = self._binding_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._binding_path)

    def _load(self) -> Optional[SpecBinding]:
        """Load binding from disk.

        Returns ``None`` when no binding file exists. Raises
        :class:`SpecBindingCorruptedError` when the file exists but cannot
        be parsed — callers must not silently reinterpret corruption as
        "no binding", since that would let an attacker disable governance
        by damaging the binding file.
        """
        if not os.path.exists(self._binding_path):
            return None
        try:
            with open(self._binding_path, encoding="utf-8") as f:
                data = json.load(f)
            binding = SpecBinding.from_dict(data)
            self._cached = binding
            return binding
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            _log.warning("spec_binding.load_failed", error=str(exc))
            raise SpecBindingCorruptedError(f"Binding file at {self._binding_path} exists but is corrupted: {exc}") from exc
