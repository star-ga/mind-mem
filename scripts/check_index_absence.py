#!/usr/bin/env python3
"""Release gate: the version being released must not exist on the index in ANY state.

What this would have caught
---------------------------
On 2026-09-01 version 5.1.0 was published to PyPI and then yanked. A yank does
not free the version number: the index permanently refuses a re-upload of those
filenames, and a resolver asked for ``mind-mem>=5.1.0`` now reports

    Ignored the following yanked versions: 5.1.0
    ERROR: No matching distribution found for mind-mem>=5.1.0

so every consumer pinned at or above 5.1.0 is broken and re-attempting 5.1.0
could not repair a single one of them. The release workflow, meanwhile, carried
``skip-existing: true`` on the publish step, so a re-pushed ``v5.1.0`` tag would
have produced an all-green release run that uploaded nothing at all -- a false
green over a version that can never be republished. This gate makes that attempt
a loud, early failure instead.

Why the raw ``releases`` map and not an install probe
-----------------------------------------------------
An installer's resolver deliberately hides yanked versions (PEP 592), so any
check phrased as "can pip find it?" answers the wrong question and would report
5.1.0 as available for use. The JSON API's ``releases`` mapping lists every
version the index has ever accepted, yanked ones included, which is the only
view that distinguishes "never used" from "burned".

Fail-closed contract
--------------------
Only one outcome is a pass: the index was successfully read and the version is
absent from the full ``releases`` map. An unreachable index, a non-200 response,
a body that is not JSON, a payload with no ``releases`` mapping, or a version
present in any state (live, yanked, or with its files deleted) all exit non-zero.
A check that cannot determine the answer must never pass.

Usage:
    python scripts/check_index_absence.py --version 5.2.0
    python scripts/check_index_absence.py --version 5.1.0 --payload-file cached.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_PROJECT = "mind-mem"
DEFAULT_API = "https://pypi.org/pypi/{project}/json"
DEFAULT_TIMEOUT = 30.0

# Outcome states. Only ABSENT is releasable.
ABSENT = "absent"
LIVE = "live"
YANKED = "yanked"
FILELESS = "fileless"


class GateError(Exception):
    """The gate could not determine the answer, so it fails."""


def canonical(version: str) -> str:
    """Fold a version string to a comparison key.

    Deliberately narrow: it lowercases, strips surrounding whitespace and a
    leading ``v``, drops a local-version suffix, and removes trailing zero
    segments from the release part so ``5.2`` and ``5.2.0`` compare equal (PEP
    440 treats them as the same version, so the index would too). Anything it
    does not recognise is returned lowercased and untouched rather than guessed
    at -- and ``classify`` additionally compares raw strings, so an unfolded
    exotic form still trips the gate. Every deviation from full PEP 440
    normalisation therefore errs toward failing, never toward passing.
    """
    text = version.strip().lower()
    if text.startswith("v"):
        text = text[1:]
    text = text.split("+", 1)[0]
    head = text
    tail = ""
    for index, char in enumerate(text):
        if not (char.isdigit() or char == "."):
            head, tail = text[:index], text[index:]
            break
    parts = [part for part in head.split(".") if part != ""]
    if not parts or not all(part.isdigit() for part in parts):
        return text
    while len(parts) > 1 and parts[-1] == "0":
        parts.pop()
    return ".".join(parts) + tail


def classify(payload: Any, version: str) -> tuple[str, str]:
    """Return ``(state, detail)`` for ``version`` in an index JSON payload.

    Raises ``GateError`` when the payload cannot be interpreted -- a malformed
    body must not be read as "the version is absent".
    """
    if not isinstance(payload, dict):
        raise GateError(f"index payload is not a JSON object (got {type(payload).__name__})")
    releases = payload.get("releases")
    if releases is None:
        raise GateError("index payload has no 'releases' mapping; cannot enumerate yanked versions")
    if not isinstance(releases, dict):
        raise GateError(f"index payload 'releases' is not a mapping (got {type(releases).__name__})")
    if not releases:
        raise GateError("index payload 'releases' mapping is empty; refusing to read that as 'version absent'")

    wanted_raw = version.strip()
    wanted = canonical(version)
    matches = [key for key in releases if key.strip() == wanted_raw or canonical(key) == wanted]
    if not matches:
        return ABSENT, f"{version} is absent from all {len(releases)} versions on the index"

    details = []
    state = FILELESS
    for key in sorted(matches):
        files = releases[key]
        if not isinstance(files, list):
            raise GateError(f"'releases[{key!r}]' is not a list (got {type(files).__name__})")
        if not files:
            details.append(f"{key}: present with no files (version number is still burned)")
            continue
        flags = set()
        for entry in files:
            if not isinstance(entry, dict):
                raise GateError(f"'releases[{key!r}]' contains a non-object entry")
            if "yanked" not in entry:
                raise GateError(f"'releases[{key!r}]' entry {entry.get('filename', '?')!r} has no 'yanked' field")
            flags.add(bool(entry["yanked"]))
        if flags == {True}:
            state = YANKED
            details.append(f"{key}: {len(files)} file(s), YANKED — the number is spent and cannot be re-uploaded")
        elif flags == {False}:
            state = LIVE if state != YANKED else state
            details.append(f"{key}: {len(files)} file(s), live")
        else:
            state = YANKED
            details.append(f"{key}: {len(files)} file(s), partially yanked")
    return state, "; ".join(details)


def fetch_payload(project: str, api: str, timeout: float) -> Any:
    """Read the index JSON for ``project``. Any failure raises ``GateError``."""
    url = api.format(project=project)
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "mind-mem-release-gate"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https index URL
            if response.status != 200:
                raise GateError(f"{url} returned HTTP {response.status}")
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise GateError(f"{url} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GateError(f"{url} unreachable: {exc}") from exc
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{url} did not return valid JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="version being released, e.g. 5.2.0")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"index project name (default: {DEFAULT_PROJECT})")
    parser.add_argument("--api", default=DEFAULT_API, help="JSON API template containing {project}")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="network timeout in seconds")
    parser.add_argument("--payload-file", help="read a captured JSON payload instead of the network (for tests/demos)")
    args = parser.parse_args(argv)

    try:
        if args.payload_file:
            with open(args.payload_file, "rb") as handle:
                payload = json.loads(handle.read().decode("utf-8"))
        else:
            payload = fetch_payload(args.project, args.api, args.timeout)
        state, detail = classify(payload, args.version)
    except GateError as exc:
        print(f"FAIL: index-absence gate could not determine the answer: {exc}")
        print("FAIL: a gate that cannot answer must not pass — refusing to release.")
        return 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"FAIL: index-absence gate could not read the payload: {exc}")
        return 1

    if state == ABSENT:
        print(f"OK: {args.project} {args.version} is not on the index in any state — {detail}")
        return 0

    print(f"FAIL: {args.project} {args.version} already exists on the index — {detail}")
    print("FAIL: a version number is spent the moment the index accepts it; a yank does not return it.")
    print("FAIL: choose the next unused version instead of re-attempting this one.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
