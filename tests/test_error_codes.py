"""Tests for mind-mem Error Codes module."""

from __future__ import annotations

import pytest

from scripts.error_codes import (
    _ERROR_METADATA,
    ErrorCategory,
    ErrorCode,
    ErrorSeverity,
    error_category,
    error_message,
    error_severity,
    is_critical,
)

# ---------------------------------------------------------------------------
# Completeness: every ErrorCode has a metadata entry
# ---------------------------------------------------------------------------

class TestMetadataCompleteness:
    """Every ErrorCode member must have metadata."""

    def test_all_codes_have_metadata(self):
        missing = [c for c in ErrorCode if c not in _ERROR_METADATA]
        assert missing == [], f"ErrorCodes missing metadata: {missing}"

    def test_metadata_has_no_extra_keys(self):
        extras = [k for k in _ERROR_METADATA if k not in ErrorCode]
        assert extras == [], f"Metadata has keys not in ErrorCode: {extras}"

    def test_metadata_tuple_structure(self):
        for code, meta in _ERROR_METADATA.items():
            assert isinstance(meta, tuple), f"{code}: metadata is not a tuple"
            assert len(meta) == 3, f"{code}: metadata tuple length != 3"
            cat, sev, msg = meta
            assert isinstance(cat, ErrorCategory), f"{code}: bad category type"
            assert isinstance(sev, ErrorSeverity), f"{code}: bad severity type"
            assert isinstance(msg, str) and len(msg) > 0, f"{code}: bad message"


# ---------------------------------------------------------------------------
# Numeric range conventions
# ---------------------------------------------------------------------------

_RANGE_MAP: dict[str, tuple[int, int, ErrorCategory]] = {
    "WORKSPACE": (1000, 1999, ErrorCategory.WORKSPACE),
    "CONFIG": (2000, 2999, ErrorCategory.CONFIG),
    "STORAGE": (3000, 3999, ErrorCategory.STORAGE),
    "RECALL": (4000, 4999, ErrorCategory.RETRIEVAL),
    "INDEX": (5000, 5999, ErrorCategory.INDEX),
    "VALIDATE": (6000, 6999, ErrorCategory.VALIDATION),
    "NETWORK": (7000, 7999, ErrorCategory.NETWORK),
    "PERMISSION": (8000, 8999, ErrorCategory.PERMISSION),
}


class TestNumericRanges:
    """Error code numeric values must match category conventions."""

    @pytest.mark.parametrize("code", list(ErrorCode))
    def test_code_in_expected_range(self, code: ErrorCode):
        prefix = code.name.split("_")[0]
        lo, hi, expected_cat = _RANGE_MAP[prefix]
        assert lo <= code.value <= hi, (
            f"{code.name} value {code.value} outside [{lo}, {hi}]"
        )
        assert error_category(code) == expected_cat, (
            f"{code.name} category {error_category(code)} != {expected_cat}"
        )


# ---------------------------------------------------------------------------
# error_message()
# ---------------------------------------------------------------------------

class TestErrorMessage:
    """error_message() returns formatted strings."""

    @pytest.mark.parametrize("code", list(ErrorCode))
    def test_format_prefix(self, code: ErrorCode):
        msg = error_message(code)
        assert msg.startswith(f"[MM-{code.value}]"), (
            f"Message for {code.name} missing prefix: {msg}"
        )

    def test_workspace_not_found_message(self):
        msg = error_message(ErrorCode.WORKSPACE_NOT_FOUND)
        assert msg == "[MM-1001] Workspace directory does not exist"

    def test_storage_disk_full_message(self):
        msg = error_message(ErrorCode.STORAGE_DISK_FULL)
        assert msg == "[MM-3004] Insufficient disk space for write operation"

    def test_message_is_nonempty_after_prefix(self):
        for code in ErrorCode:
            msg = error_message(code)
            after_prefix = msg.split("] ", 1)[1]
            assert len(after_prefix) > 0


# ---------------------------------------------------------------------------
# error_category()
# ---------------------------------------------------------------------------

class TestErrorCategory:
    """error_category() returns the correct ErrorCategory."""

    def test_workspace_codes(self):
        for code in (
            ErrorCode.WORKSPACE_NOT_FOUND,
            ErrorCode.WORKSPACE_NOT_INITIALIZED,
            ErrorCode.WORKSPACE_CORRUPTED,
        ):
            assert error_category(code) == ErrorCategory.WORKSPACE

    def test_config_codes(self):
        for code in (
            ErrorCode.CONFIG_INVALID_JSON,
            ErrorCode.CONFIG_MISSING_KEY,
            ErrorCode.CONFIG_VALUE_OUT_OF_RANGE,
            ErrorCode.CONFIG_FILE_UNREADABLE,
        ):
            assert error_category(code) == ErrorCategory.CONFIG

    def test_storage_codes(self):
        for code in (
            ErrorCode.STORAGE_WRITE_FAILED,
            ErrorCode.STORAGE_READ_FAILED,
            ErrorCode.STORAGE_LOCK_TIMEOUT,
            ErrorCode.STORAGE_DISK_FULL,
            ErrorCode.STORAGE_BLOCK_CORRUPTED,
        ):
            assert error_category(code) == ErrorCategory.STORAGE

    def test_retrieval_codes(self):
        for code in (
            ErrorCode.RECALL_NO_RESULTS,
            ErrorCode.RECALL_INDEX_STALE,
            ErrorCode.RECALL_QUERY_TOO_LONG,
            ErrorCode.RECALL_TIMEOUT,
        ):
            assert error_category(code) == ErrorCategory.RETRIEVAL

    def test_index_codes(self):
        for code in (
            ErrorCode.INDEX_BUILD_FAILED,
            ErrorCode.INDEX_CORRUPTED,
            ErrorCode.INDEX_VERSION_MISMATCH,
            ErrorCode.INDEX_REINDEX_REQUIRED,
        ):
            assert error_category(code) == ErrorCategory.INDEX

    def test_validation_codes(self):
        for code in (
            ErrorCode.VALIDATE_BLOCK_MALFORMED,
            ErrorCode.VALIDATE_SCHEMA_MISMATCH,
            ErrorCode.VALIDATE_DUPLICATE_ID,
            ErrorCode.VALIDATE_CONTENT_HASH_MISMATCH,
        ):
            assert error_category(code) == ErrorCategory.VALIDATION

    def test_network_codes(self):
        for code in (
            ErrorCode.NETWORK_LLM_UNAVAILABLE,
            ErrorCode.NETWORK_EMBEDDING_FAILED,
            ErrorCode.NETWORK_TIMEOUT,
        ):
            assert error_category(code) == ErrorCategory.NETWORK

    def test_permission_codes(self):
        for code in (
            ErrorCode.PERMISSION_DENIED,
            ErrorCode.PERMISSION_READONLY_WORKSPACE,
        ):
            assert error_category(code) == ErrorCategory.PERMISSION


# ---------------------------------------------------------------------------
# error_severity()
# ---------------------------------------------------------------------------

class TestErrorSeverity:
    """error_severity() returns the correct ErrorSeverity."""

    def test_critical_codes(self):
        critical_codes = [
            ErrorCode.WORKSPACE_CORRUPTED,
            ErrorCode.STORAGE_DISK_FULL,
        ]
        for code in critical_codes:
            assert error_severity(code) == ErrorSeverity.CRITICAL, (
                f"{code.name} should be CRITICAL"
            )

    def test_high_codes(self):
        high_codes = [
            ErrorCode.WORKSPACE_NOT_FOUND,
            ErrorCode.WORKSPACE_NOT_INITIALIZED,
            ErrorCode.STORAGE_WRITE_FAILED,
            ErrorCode.STORAGE_READ_FAILED,
            ErrorCode.STORAGE_BLOCK_CORRUPTED,
            ErrorCode.INDEX_BUILD_FAILED,
            ErrorCode.INDEX_CORRUPTED,
            ErrorCode.VALIDATE_CONTENT_HASH_MISMATCH,
            ErrorCode.PERMISSION_DENIED,
        ]
        for code in high_codes:
            assert error_severity(code) == ErrorSeverity.HIGH, (
                f"{code.name} should be HIGH"
            )

    def test_low_codes(self):
        low_codes = [
            ErrorCode.CONFIG_MISSING_KEY,
            ErrorCode.CONFIG_VALUE_OUT_OF_RANGE,
            ErrorCode.RECALL_NO_RESULTS,
            ErrorCode.RECALL_QUERY_TOO_LONG,
            ErrorCode.VALIDATE_DUPLICATE_ID,
        ]
        for code in low_codes:
            assert error_severity(code) == ErrorSeverity.LOW, (
                f"{code.name} should be LOW"
            )


# ---------------------------------------------------------------------------
# is_critical()
# ---------------------------------------------------------------------------

class TestIsCritical:
    """is_critical() returns True only for CRITICAL severity codes."""

    def test_critical_true(self):
        assert is_critical(ErrorCode.WORKSPACE_CORRUPTED) is True
        assert is_critical(ErrorCode.STORAGE_DISK_FULL) is True

    def test_critical_false(self):
        assert is_critical(ErrorCode.WORKSPACE_NOT_FOUND) is False
        assert is_critical(ErrorCode.CONFIG_MISSING_KEY) is False
        assert is_critical(ErrorCode.RECALL_NO_RESULTS) is False
        assert is_critical(ErrorCode.NETWORK_TIMEOUT) is False

    @pytest.mark.parametrize("code", list(ErrorCode))
    def test_is_critical_matches_severity(self, code: ErrorCode):
        expected = error_severity(code) == ErrorSeverity.CRITICAL
        assert is_critical(code) is expected


# ---------------------------------------------------------------------------
# Enum value uniqueness
# ---------------------------------------------------------------------------

class TestEnumIntegrity:
    """ErrorCode enum values must be unique integers."""

    def test_all_values_are_ints(self):
        for code in ErrorCode:
            assert isinstance(code.value, int)

    def test_no_duplicate_values(self):
        values = [c.value for c in ErrorCode]
        assert len(values) == len(set(values)), "Duplicate ErrorCode values found"

    def test_total_count(self):
        assert len(ErrorCode) == 29, f"Expected 29 error codes, got {len(ErrorCode)}"
