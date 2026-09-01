"""mind-mem Error Codes — structured error classification.

Provides a canonical enum of error codes used across mind-mem modules.
Each code maps to a category, severity, and human-readable description.

Usage:
    from mind_mem.error_codes import ErrorCode, error_message

    raise ValueError(error_message(ErrorCode.WORKSPACE_NOT_FOUND))
"""

from __future__ import annotations

from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    WORKSPACE = "workspace"
    CONFIG = "config"
    STORAGE = "storage"
    RETRIEVAL = "retrieval"
    INDEX = "index"
    VALIDATION = "validation"
    NETWORK = "network"
    PERMISSION = "permission"


class ErrorCode(Enum):
    """Canonical error codes for mind-mem operations."""

    # Workspace errors (1xxx)
    WORKSPACE_NOT_FOUND = 1001
    WORKSPACE_NOT_INITIALIZED = 1002
    WORKSPACE_CORRUPTED = 1003

    # Config errors (2xxx)
    CONFIG_INVALID_JSON = 2001
    CONFIG_MISSING_KEY = 2002
    CONFIG_VALUE_OUT_OF_RANGE = 2003
    CONFIG_FILE_UNREADABLE = 2004

    # Storage errors (3xxx)
    STORAGE_WRITE_FAILED = 3001
    STORAGE_READ_FAILED = 3002
    STORAGE_LOCK_TIMEOUT = 3003
    STORAGE_DISK_FULL = 3004
    STORAGE_BLOCK_CORRUPTED = 3005

    # Retrieval errors (4xxx)
    RECALL_NO_RESULTS = 4001
    RECALL_INDEX_STALE = 4002
    RECALL_QUERY_TOO_LONG = 4003
    RECALL_TIMEOUT = 4004

    # Index errors (5xxx)
    INDEX_BUILD_FAILED = 5001
    INDEX_CORRUPTED = 5002
    INDEX_VERSION_MISMATCH = 5003
    INDEX_REINDEX_REQUIRED = 5004

    # Validation errors (6xxx)
    VALIDATE_BLOCK_MALFORMED = 6001
    VALIDATE_SCHEMA_MISMATCH = 6002
    VALIDATE_DUPLICATE_ID = 6003
    VALIDATE_CONTENT_HASH_MISMATCH = 6004

    # Network errors (7xxx)
    NETWORK_LLM_UNAVAILABLE = 7001
    NETWORK_EMBEDDING_FAILED = 7002
    NETWORK_TIMEOUT = 7003

    # Permission errors (8xxx)
    PERMISSION_DENIED = 8001
    PERMISSION_READONLY_WORKSPACE = 8002


# Error metadata registry
_ERROR_METADATA: dict[ErrorCode, tuple[ErrorCategory, ErrorSeverity, str]] = {
    ErrorCode.WORKSPACE_NOT_FOUND: (ErrorCategory.WORKSPACE, ErrorSeverity.HIGH, "Workspace directory does not exist"),
    ErrorCode.WORKSPACE_NOT_INITIALIZED: (
        ErrorCategory.WORKSPACE,
        ErrorSeverity.HIGH,
        "Workspace has not been initialized (missing MEMORY.md)",
    ),
    ErrorCode.WORKSPACE_CORRUPTED: (
        ErrorCategory.WORKSPACE,
        ErrorSeverity.CRITICAL,
        "Workspace structure is corrupted or incomplete",
    ),
    ErrorCode.CONFIG_INVALID_JSON: (
        ErrorCategory.CONFIG,
        ErrorSeverity.MEDIUM,
        "Configuration file contains invalid JSON",
    ),
    ErrorCode.CONFIG_MISSING_KEY: (ErrorCategory.CONFIG, ErrorSeverity.LOW, "Required configuration key is missing"),
    ErrorCode.CONFIG_VALUE_OUT_OF_RANGE: (
        ErrorCategory.CONFIG,
        ErrorSeverity.LOW,
        "Configuration value is outside acceptable range",
    ),
    ErrorCode.CONFIG_FILE_UNREADABLE: (ErrorCategory.CONFIG, ErrorSeverity.MEDIUM, "Configuration file cannot be read"),
    ErrorCode.STORAGE_WRITE_FAILED: (ErrorCategory.STORAGE, ErrorSeverity.HIGH, "Failed to write data to storage"),
    ErrorCode.STORAGE_READ_FAILED: (ErrorCategory.STORAGE, ErrorSeverity.HIGH, "Failed to read data from storage"),
    ErrorCode.STORAGE_LOCK_TIMEOUT: (ErrorCategory.STORAGE, ErrorSeverity.MEDIUM, "File lock acquisition timed out"),
    ErrorCode.STORAGE_DISK_FULL: (
        ErrorCategory.STORAGE,
        ErrorSeverity.CRITICAL,
        "Insufficient disk space for write operation",
    ),
    ErrorCode.STORAGE_BLOCK_CORRUPTED: (ErrorCategory.STORAGE, ErrorSeverity.HIGH, "Memory block data is corrupted"),
    ErrorCode.RECALL_NO_RESULTS: (ErrorCategory.RETRIEVAL, ErrorSeverity.LOW, "No results found for recall query"),
    ErrorCode.RECALL_INDEX_STALE: (
        ErrorCategory.RETRIEVAL,
        ErrorSeverity.MEDIUM,
        "Search index is stale and may return incomplete results",
    ),
    ErrorCode.RECALL_QUERY_TOO_LONG: (ErrorCategory.RETRIEVAL, ErrorSeverity.LOW, "Query exceeds maximum length"),
    ErrorCode.RECALL_TIMEOUT: (ErrorCategory.RETRIEVAL, ErrorSeverity.MEDIUM, "Recall operation timed out"),
    ErrorCode.INDEX_BUILD_FAILED: (ErrorCategory.INDEX, ErrorSeverity.HIGH, "Failed to build search index"),
    ErrorCode.INDEX_CORRUPTED: (ErrorCategory.INDEX, ErrorSeverity.HIGH, "Search index is corrupted"),
    ErrorCode.INDEX_VERSION_MISMATCH: (
        ErrorCategory.INDEX,
        ErrorSeverity.MEDIUM,
        "Index version does not match expected schema",
    ),
    ErrorCode.INDEX_REINDEX_REQUIRED: (
        ErrorCategory.INDEX,
        ErrorSeverity.MEDIUM,
        "Full reindex required after schema change",
    ),
    ErrorCode.VALIDATE_BLOCK_MALFORMED: (
        ErrorCategory.VALIDATION,
        ErrorSeverity.MEDIUM,
        "Memory block structure is malformed",
    ),
    ErrorCode.VALIDATE_SCHEMA_MISMATCH: (
        ErrorCategory.VALIDATION,
        ErrorSeverity.MEDIUM,
        "Data does not match expected schema",
    ),
    ErrorCode.VALIDATE_DUPLICATE_ID: (ErrorCategory.VALIDATION, ErrorSeverity.LOW, "Duplicate block ID detected"),
    ErrorCode.VALIDATE_CONTENT_HASH_MISMATCH: (
        ErrorCategory.VALIDATION,
        ErrorSeverity.HIGH,
        "Content hash does not match stored value",
    ),
    ErrorCode.NETWORK_LLM_UNAVAILABLE: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, "LLM provider is unavailable"),
    ErrorCode.NETWORK_EMBEDDING_FAILED: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, "Embedding generation failed"),
    ErrorCode.NETWORK_TIMEOUT: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, "Network request timed out"),
    ErrorCode.PERMISSION_DENIED: (
        ErrorCategory.PERMISSION,
        ErrorSeverity.HIGH,
        "Permission denied for requested operation",
    ),
    ErrorCode.PERMISSION_READONLY_WORKSPACE: (
        ErrorCategory.PERMISSION,
        ErrorSeverity.MEDIUM,
        "Workspace is in read-only mode",
    ),
}


def error_message(code: ErrorCode) -> str:
    """Get human-readable error message for an error code."""
    meta = _ERROR_METADATA.get(code)
    if meta is None:
        return f"Unknown error (code {code.value})"
    _, _, message = meta
    return f"[MM-{code.value}] {message}"


def error_category(code: ErrorCode) -> ErrorCategory:
    """Get the category for an error code."""
    meta = _ERROR_METADATA.get(code)
    return meta[0] if meta else ErrorCategory.WORKSPACE


def error_severity(code: ErrorCode) -> ErrorSeverity:
    """Get the severity for an error code."""
    meta = _ERROR_METADATA.get(code)
    return meta[1] if meta else ErrorSeverity.MEDIUM


def is_critical(code: ErrorCode) -> bool:
    """Check if an error code is critical severity."""
    return error_severity(code) == ErrorSeverity.CRITICAL
