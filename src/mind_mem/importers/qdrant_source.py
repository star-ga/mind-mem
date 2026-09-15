# Copyright 2026 STARGA, Inc.
"""Bounded, read-only Qdrant REST collection export for migration imports.

Qdrant's scroll endpoint is an endpoint-backed source, so it is intentionally
separate from the local dump readers.  This module owns HTTP framing and
pagination only; payload-to-record mapping stays in :mod:`parsers`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .records import ImportParseError

__all__ = [
    "MAX_PAGE_SIZE",
    "MAX_PAGES",
    "MAX_RECORDS",
    "MAX_RESPONSE_BYTES",
    "MAX_TOTAL_RESPONSE_BYTES",
    "scroll_qdrant",
]

MAX_PAGE_SIZE = 100
MAX_PAGES = 1_000
MAX_RECORDS = 100_000
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_COLLECTION_LENGTH = 255
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _DuplicateKeyError(ValueError):
    """A JSON object repeated a key and cannot be trusted as a receipt."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


class _RejectRedirect(HTTPRedirectHandler):
    """Keep an API key bound to the explicitly supplied Qdrant host."""

    def redirect_request(self, _request: Request, _fp: Any, _code: int, _msg: str, _headers: Any, _new_url: str) -> None:
        raise ImportParseError("qdrant endpoint returned a redirect; refusing to follow it")


_OPENER = build_opener(_RejectRedirect)


def _endpoint_url(endpoint: str, collection: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ImportParseError("qdrant endpoint must be a non-empty URL")
    if not isinstance(collection, str) or not collection.strip():
        raise ImportParseError("qdrant collection must be a non-empty name")
    if len(collection.strip()) > MAX_COLLECTION_LENGTH:
        raise ImportParseError(f"qdrant collection name exceeds {MAX_COLLECTION_LENGTH} characters")
    try:
        parsed = urlsplit(endpoint.strip())
        hostname = parsed.hostname
    except ValueError as exc:
        raise ImportParseError("qdrant endpoint is not a valid URL") from exc
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ImportParseError("qdrant endpoint must use http or https and include a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ImportParseError("qdrant endpoint must not contain credentials, query parameters, or a fragment")
    base_path = parsed.path.rstrip("/")
    path = f"{base_path}/collections/{quote(collection.strip(), safe='')}/points/scroll"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _api_key(api_key_env: str | None) -> str | None:
    if api_key_env is None or api_key_env == "":
        return None
    if not isinstance(api_key_env, str) or not _ENV_NAME_RE.fullmatch(api_key_env):
        raise ImportParseError("qdrant api-key environment name is invalid")
    value = os.environ.get(api_key_env)
    if not value:
        raise ImportParseError(f"qdrant api-key environment variable {api_key_env!r} is not set")
    return value


def _decode_page(raw: bytes, page: int) -> tuple[list[Mapping[str, Any]], Any]:
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, _DuplicateKeyError) as exc:
        if isinstance(exc, _DuplicateKeyError):
            raise ImportParseError(f"qdrant page {page} contains duplicate JSON object keys") from exc
        raise ImportParseError(f"qdrant page {page} is not valid JSON") from exc
    status = payload.get("status") if isinstance(payload, Mapping) else None
    if not isinstance(payload, Mapping) or (status is not None and status != "ok"):
        raise ImportParseError(f"qdrant page {page} has an invalid status envelope")
    result = payload.get("result")
    if not isinstance(result, Mapping) or not isinstance(result.get("points"), list):
        raise ImportParseError(f"qdrant page {page} is missing result.points")
    points = result["points"]
    if not all(isinstance(point, Mapping) for point in points):
        raise ImportParseError(f"qdrant page {page} contains a non-object point")
    return points, result.get("next_page_offset")


def scroll_qdrant(
    endpoint: str,
    collection: str,
    *,
    api_key_env: str | None = None,
    page_size: int = 100,
    max_pages: int = MAX_PAGES,
    max_records: int = MAX_RECORDS,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    max_total_response_bytes: int = MAX_TOTAL_RESPONSE_BYTES,
    timeout: float = 30.0,
) -> tuple[Mapping[str, Any], ...]:
    """Read points from one Qdrant collection using bounded REST scrolls.

    The endpoint is read only: requests are POSTs to Qdrant's ``points/scroll``
    reader and request bodies never include vectors.  A point's payload is
    returned unchanged for the parser's explicit text-field mapping.
    """
    if type(page_size) is not int or not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ImportParseError(f"qdrant page size must be an integer from 1 to {MAX_PAGE_SIZE}")
    if type(max_pages) is not int or not 1 <= max_pages <= MAX_PAGES:
        raise ImportParseError(f"qdrant max pages must be an integer from 1 to {MAX_PAGES}")
    if type(max_records) is not int or not 1 <= max_records <= MAX_RECORDS:
        raise ImportParseError(f"qdrant max records must be an integer from 1 to {MAX_RECORDS}")
    if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= MAX_RESPONSE_BYTES:
        raise ImportParseError(f"qdrant response bound must be an integer from 1 to {MAX_RESPONSE_BYTES}")
    if type(max_total_response_bytes) is not int or not 1 <= max_total_response_bytes <= MAX_TOTAL_RESPONSE_BYTES:
        raise ImportParseError(f"qdrant cumulative response bound must be an integer from 1 to {MAX_TOTAL_RESPONSE_BYTES}")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 300:
        raise ImportParseError("qdrant timeout must be a number greater than 0 and at most 300 seconds")

    url = _endpoint_url(endpoint, collection)
    key = _api_key(api_key_env)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key is not None:
        headers["api-key"] = key

    points: list[Mapping[str, Any]] = []
    offset: Any = None
    seen_offsets: set[str] = set()
    total_response_bytes = 0
    for page in range(1, max_pages + 1):
        remaining = max_total_response_bytes - total_response_bytes
        if remaining <= 0:
            raise ImportParseError(f"qdrant cumulative response exceeds {max_total_response_bytes} bytes")
        body: dict[str, Any] = {"limit": page_size, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        request = Request(url, data=json.dumps(body, separators=(",", ":")).encode("utf-8"), headers=headers, method="POST")
        try:
            with _OPENER.open(request, timeout=float(timeout)) as response:  # nosec B310 — endpoint is scheme/host validated above
                raw = response.read(min(max_response_bytes, remaining) + 1)
        except HTTPError as exc:
            raise ImportParseError(f"qdrant endpoint returned HTTP {exc.code}") from exc
        except (OSError, URLError, TimeoutError) as exc:
            raise ImportParseError(f"qdrant endpoint request failed ({type(exc).__name__})") from exc
        if len(raw) > max_response_bytes:
            raise ImportParseError(f"qdrant response exceeds {max_response_bytes} bytes")
        if total_response_bytes + len(raw) > max_total_response_bytes:
            raise ImportParseError(f"qdrant cumulative response exceeds {max_total_response_bytes} bytes")
        total_response_bytes += len(raw)
        page_points, next_offset = _decode_page(raw, page)
        if next_offset is not None and (
            isinstance(next_offset, bool) or not isinstance(next_offset, (str, int)) or not str(next_offset).strip()
        ):
            raise ImportParseError(f"qdrant page {page} has an invalid pagination offset")
        if len(points) + len(page_points) > max_records:
            raise ImportParseError(f"qdrant result exceeds {max_records} records")
        points.extend(page_points)
        if next_offset is None:
            return tuple(points)
        marker = json.dumps(next_offset, sort_keys=True, separators=(",", ":"))
        if marker in seen_offsets or (offset is not None and marker == json.dumps(offset, sort_keys=True, separators=(",", ":"))):
            raise ImportParseError("qdrant pagination repeated an offset; refusing a non-progressing export")
        seen_offsets.add(marker)
        offset = next_offset
    raise ImportParseError(f"qdrant export exceeded the {max_pages}-page bound before reaching the end")
