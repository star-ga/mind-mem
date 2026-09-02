#!/usr/bin/env python3
"""Release gate: zero open code-scanning alerts, AND proof the scanner actually ran.

What this replaces
------------------
Until 2026-09-02 the ``alerts-gate`` job in ``.github/workflows/release.yml``
ended its unreadable-API branch like this::

    echo "::warning::code-scanning alerts API not readable; failing open"
    exit 0

So a 403, a 404, a missing ``security-events`` permission, a rate limit or a
network blip all produced a green gate. The release then shipped with its only
security check having never run, and the run log said so in a warning nobody
reads. That is the same failure mode that let 5.0.1 ship from a commit whose CI
had failed every Windows matrix row: the release path looked for a problem, could
not look, and carried on. A verifier that died is not a verifier that passed.

The four outcomes, which are not the same thing
-----------------------------------------------
``clean``
    The alerts endpoint answered, reported zero open alerts, AND at least one
    code-scanning analysis exists for the repository. Only this passes.
``alerts-open``
    The endpoint answered and returned one or more open alerts. Fails, listing
    them. Never bypassable -- see ``--bypass`` below.
``not-enabled``
    The API says, in so many words, that there is nothing to read: HTTP 404
    ``no analysis found``, or HTTP 403 naming Advanced Security, or a readable
    analyses list that is empty. Code scanning is not producing results for this
    repository, so the gate genuinely cannot run. Fails by default; this is the
    only state an explicit human bypass can clear.
``unreadable``
    Anything else: an authorization failure, an unexpected status, a body that
    is not JSON, an empty body, a payload whose shape is wrong, or repeated
    transient failures. Fails, naming the actual status or error. Never
    bypassable, because every cause is a fixable misconfiguration on our side --
    most commonly a workflow missing ``permissions: security-events: read``.

Measured ground truth for those states, captured 2026-09-02:

  * ``star-ga/mind-mem``   -> 200, body ``[]``, 3 analyses (CodeQL x2, Bandit)
  * ``star-ga/mind-nerve`` -> 404 ``{"message":"no analysis found",...}``
  * ``octocat/Hello-World``-> 403 ``{"message":"You are not authorized to read
    code scanning alerts.",...}``

Why zero alerts is not enough on its own
----------------------------------------
``200 []`` reads as "clean", but a repository whose scanner never ran would also
like to report zero problems, and a negative assertion with no positive control
is the most common way a security check proves nothing. So a zero is only
accepted alongside evidence that an analysis exists. If the scanner stopped
uploading results, this gate says ``not-enabled`` rather than ``clean``.

The bypass, and why it defaults to off
--------------------------------------
Code scanning IS enabled here today (advanced setup: a CodeQL workflow plus a
Bandit SARIF upload), so the ordinary answer to "can this gate run?" is yes and
no bypass is needed. But the repository could be made private without an
Advanced Security licence, or a fork could run this workflow with scanning off,
and a gate that can then never pass would block every release until someone
edits a security check under release pressure -- which is when checks get
weakened. So there is exactly one escape hatch, and it is deliberately awkward:

  * it must be typed, per run, as the literal value below -- no default, no
    repository variable, nothing sticky that could silently skip a later
    release;
  * it is offered only as a ``workflow_dispatch`` input, and a tag push carries
    no inputs at all, so the normal release trigger has no bypass path;
  * it clears ONLY the ``not-enabled`` state. It cannot launder an open alert
    and it cannot paper over a permission error;
  * using it prints an unmissable annotation naming the actor who set it.

Usage:
    python scripts/check_code_scanning_alerts.py --repo star-ga/mind-mem
    python scripts/check_code_scanning_alerts.py --alerts-file body.json --http-status 403
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

DEFAULT_REPO = "star-ga/mind-mem"
DEFAULT_TIMEOUT = 60.0
DEFAULT_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 15.0
PER_PAGE = 100

# The exact string a human must type into the workflow_dispatch input. Long and
# unpleasant on purpose: nobody types this by reflex, and it reads in the run
# log as what it is.
BYPASS_VALUE = "I-ACCEPT-AN-UNGATED-RELEASE"

# Outcome states. Only CLEAN is releasable; only NOT_ENABLED is bypassable.
CLEAN = "clean"
ALERTS_OPEN = "alerts-open"
NOT_ENABLED = "not-enabled"
UNREADABLE = "unreadable"

# Substrings GitHub uses when the answer is "there is nothing here to read",
# as opposed to "you may not read it". Matched case-insensitively against the
# response body.
NOT_ENABLED_MARKERS = (
    "no analysis found",
    "advanced security",
    "code scanning is not enabled",
)


class GateError(Exception):
    """The gate could not determine the answer, so it fails."""


class TransientError(Exception):
    """The API failed in a way that is worth one more attempt."""


class Response:
    """An HTTP status plus a raw body, the two things a verdict needs."""

    __slots__ = ("status", "body")

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body


def _obj(value: Any) -> dict[str, Any]:
    """Narrow a decoded JSON field to a mapping, or an empty one.

    Optional fields on an alert are absent often enough that reading them
    defensively is the difference between a listing and a crash — and a crash
    here would surface as ``unreadable`` on a page that DID report open alerts,
    turning a precise finding into a vague one.
    """
    return value if isinstance(value, dict) else {}


def _looks_not_enabled(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in NOT_ENABLED_MARKERS)


def _describe_alert(alert: dict[str, Any]) -> str:
    """One log line per open alert, tolerant of missing optional fields."""
    rule = _obj(alert.get("rule"))
    location = _obj(_obj(alert.get("most_recent_instance")).get("location"))
    return "  #{number} [{severity}] {rule_id} — {path}:{line}".format(
        number=alert.get("number", "?"),
        severity=rule.get("severity", "?"),
        rule_id=rule.get("id", "?"),
        path=location.get("path", "?"),
        line=location.get("start_line", "?"),
    )


def classify_alerts_body(body: str) -> tuple[str, str]:
    """Classify a 200 response body from the open-alerts endpoint.

    Raises ``GateError`` when the body cannot be interpreted. An empty or
    malformed body must never be read as "no alerts found" -- that is precisely
    the mistake this gate exists to remove.
    """
    if body.strip() == "":
        raise GateError("the alerts endpoint returned HTTP 200 with an empty body; an absent answer is not a clean one")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GateError(f"the alerts endpoint returned HTTP 200 with a body that is not JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise GateError(f"alerts payload is not a JSON array (got {type(payload).__name__})")

    for entry in payload:
        if not isinstance(entry, dict):
            raise GateError("alerts payload contains a non-object entry")
        if "state" not in entry:
            raise GateError(f"alert {entry.get('number', '?')} has no 'state' field; open cannot be told from dismissed")
        if entry["state"] != "open":
            # We asked the API for state=open. If it hands back anything else,
            # the filter did not apply and the page says nothing about how many
            # open alerts exist, so there is no answer to report.
            raise GateError(
                f"asked for state=open but the API returned alert {entry.get('number', '?')} in state "
                f"{entry['state']!r}; the filter did not apply, so this page cannot be trusted"
            )

    if not payload:
        return CLEAN, "the alerts endpoint answered and reported 0 open alerts"

    count = len(payload)
    # A full page means there may be more; the verdict is already FAIL either
    # way, so this only affects how the count is worded.
    counted = f"at least {count}" if count >= PER_PAGE else f"{count}"
    listing = "\n".join(_describe_alert(alert) for alert in payload)
    return ALERTS_OPEN, f"{counted} open code-scanning alert(s):\n{listing}"


def classify_response(response: Response) -> tuple[str, str]:
    """Turn an alerts-endpoint response into ``(state, detail)``.

    Raises ``TransientError`` for statuses worth retrying, so a rate limit or a
    momentary 5xx does not become a release-blocking verdict on the first try.
    """
    status = response.status
    if status == 200:
        return classify_alerts_body(response.body)
    if status in (429, 500, 502, 503, 504):
        raise TransientError(f"HTTP {status} from the alerts endpoint")
    snippet = " ".join(response.body.split())[:300]
    if status in (403, 404) and _looks_not_enabled(response.body):
        return NOT_ENABLED, f"HTTP {status}: {snippet}"
    if status == 403:
        return UNREADABLE, (
            f"HTTP 403: {snippet} — this is an authorization failure, not a disabled feature. "
            "The usual cause is a job missing 'permissions: security-events: read'."
        )
    if status == 404:
        return UNREADABLE, f"HTTP 404: {snippet} — the endpoint was not found and did not say scanning is off."
    return UNREADABLE, f"HTTP {status}: {snippet}"


def classify_analyses_body(body: str) -> tuple[bool, str]:
    """Positive control on a zero: does any code-scanning analysis exist?

    Returns ``(has_analysis, detail)``. Raises ``GateError`` on a body that
    cannot be interpreted, because "I could not tell" is not "yes".
    """
    if body.strip() == "":
        raise GateError("the analyses endpoint returned HTTP 200 with an empty body")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GateError(f"the analyses endpoint returned HTTP 200 with a body that is not JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise GateError(f"analyses payload is not a JSON array (got {type(payload).__name__})")
    if not payload:
        return False, "the analyses endpoint answered and reported 0 analyses; nothing has ever been scanned"
    newest = _obj(payload[0])
    tool = _obj(newest.get("tool"))
    return True, (
        f"{len(payload)} analysis/analyses recorded; newest id={newest.get('id', '?')} "
        f"tool={tool.get('name', '?')} created={newest.get('created_at', '?')}"
    )


def classify_analyses_response(response: Response) -> tuple[str, str]:
    """``(state, detail)`` for the analyses probe, in the same vocabulary."""
    if response.status == 200:
        has_analysis, detail = classify_analyses_body(response.body)
        return (CLEAN if has_analysis else NOT_ENABLED), detail
    if response.status in (429, 500, 502, 503, 504):
        raise TransientError(f"HTTP {response.status} from the analyses endpoint")
    snippet = " ".join(response.body.split())[:300]
    if response.status in (403, 404) and _looks_not_enabled(response.body):
        return NOT_ENABLED, f"analyses endpoint HTTP {response.status}: {snippet}"
    return UNREADABLE, f"analyses endpoint HTTP {response.status}: {snippet}"


def gh_get(path: str, timeout: float) -> Response:
    """Read one API path with ``gh api -i`` and return its status and body.

    ``-i`` is what makes requirement (c) answerable: the verdict has to name the
    actual HTTP status, and ``gh`` only exposes it in the response headers. The
    ``gh`` CLI is already this workflow's API client, so no new credential path
    is introduced -- it inherits ``GH_TOKEN``.

    ``--paginate`` is deliberately not used: it concatenates responses, which
    would make the status line ambiguous. A single 100-item page cannot turn a
    non-zero count into zero, so the verdict is unaffected.
    """
    command = ["gh", "api", "-i", "-X", "GET", path]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise GateError("the 'gh' CLI is not available; cannot read code-scanning alerts") from exc
    except subprocess.TimeoutExpired as exc:
        raise TransientError(f"'gh api {path}' timed out after {timeout}s") from exc

    raw = completed.stdout
    if not raw.strip():
        raise GateError(f"'gh api {path}' produced no response at all (exit {completed.returncode}): {completed.stderr.strip()[:300]}")

    first_line, _, rest = raw.partition("\n")
    fields = first_line.split()
    if len(fields) < 2 or not fields[0].upper().startswith("HTTP") or not fields[1].isdigit():
        raise GateError(f"'gh api {path}' did not begin with an HTTP status line (got {first_line.strip()[:120]!r})")
    status = int(fields[1])

    # Headers and body are separated by the first blank line.
    body = ""
    for separator in ("\r\n\r\n", "\n\n"):
        if separator in rest:
            body = rest.split(separator, 1)[1]
            break
    return Response(status, body)


def _with_retries(
    read: Callable[[], tuple[str, str]],
    attempts: int,
    delay: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str]:
    """Run a ``() -> (state, detail)`` reader, retrying only transient failures.

    Retries exist so operability never depends on the bypass. Once the attempts
    are spent the result is ``unreadable``, never a pass.
    """
    last = "no attempt was made"
    for attempt in range(1, attempts + 1):
        try:
            return read()
        except TransientError as exc:
            last = str(exc)
            print(f"transient failure on attempt {attempt}/{attempts}: {last}")
            if attempt < attempts:
                sleep(delay)
    return UNREADABLE, f"{attempts} attempt(s) all failed transiently; last was {last}"


def report(state: str, detail: str, bypass: str, actor: str) -> tuple[int, list[str]]:
    """Turn a state into ``(exit_code, lines)``.

    The bypass is consulted in exactly one branch. Ordering matters: the open-
    alert branch is evaluated before the bypass is ever looked at, so a typed
    bypass cannot clear a real finding.
    """
    lines: list[str] = []
    if state == CLEAN:
        lines.append(f"OK: {detail}")
        lines.append("OK: code-scanning gate ran and found nothing open.")
        return 0, lines

    if state == ALERTS_OPEN:
        lines.append(f"FAIL: release blocked — {detail}")
        lines.append(
            "::error::Release blocked by open code-scanning alert(s). "
            "Close or dismiss them (with a documented rationale) and re-push the tag."
        )
        if bypass:
            lines.append("FAIL: a bypass was supplied, and it is being ignored. The bypass covers a disabled scanner, never a live alert.")
        return 1, lines

    if state == NOT_ENABLED:
        if bypass == BYPASS_VALUE:
            who = actor or "an unrecorded actor"
            lines.append(f"BYPASS: code scanning is not producing results — {detail}")
            lines.append(f"::warning::DELIBERATE SECURITY-GATE BYPASS: {who} set the code-scanning bypass input to '{BYPASS_VALUE}'.")
            lines.append(f"::warning::This release is NOT gated on code-scanning alerts. Requested by: {who}.")
            lines.append("BYPASS: the gate did not run. Re-enable code scanning and stop using this input.")
            return 0, lines
        lines.append(f"FAIL: code scanning is not producing results for this repository — {detail}")
        lines.append("::error::Release blocked: the code-scanning gate could not run, so it did not pass.")
        if bypass:
            lines.append(
                f"FAIL: a bypass value was supplied but it is not the exact required string; it must be literally '{BYPASS_VALUE}'."
            )
        else:
            lines.append("FAIL: re-enable code scanning, or dispatch this workflow with the bypass input set to the exact")
            lines.append(f"FAIL: string '{BYPASS_VALUE}' to record a deliberate, attributed, ungated release.")
        return 1, lines

    lines.append(f"FAIL: the code-scanning gate could not run — {detail}")
    lines.append("::error::Release blocked: the code-scanning gate could not read the API, so it did not pass.")
    lines.append("FAIL: this state is not bypassable — every cause is a fixable misconfiguration, not a disabled feature.")
    if bypass:
        lines.append("FAIL: a bypass was supplied, and it is being ignored. Fix the access problem instead.")
    return 1, lines


def evaluate(args: argparse.Namespace) -> tuple[str, str]:
    """Produce ``(state, detail)`` for the run described by ``args``."""
    if args.alerts_file is not None:
        with open(args.alerts_file, "rb") as handle:
            body = handle.read().decode("utf-8", errors="replace")
        state, detail = classify_response(Response(args.http_status, body))
    else:
        path = f"repos/{args.repo}/code-scanning/alerts?state=open&per_page={PER_PAGE}"
        state, detail = _with_retries(
            lambda: classify_response(gh_get(path, args.timeout)),
            args.attempts,
            args.retry_delay,
        )
    if state != CLEAN:
        return state, detail

    # Positive control: a zero is only clean if something was actually scanned.
    if args.analyses_file is not None:
        with open(args.analyses_file, "rb") as handle:
            body = handle.read().decode("utf-8", errors="replace")
        control_state, control_detail = classify_analyses_response(Response(args.analyses_http_status, body))
    elif args.alerts_file is not None:
        # Simulated alerts response with no simulated analyses response: keep
        # the alerts verdict rather than inventing a control result.
        return state, detail
    else:
        path = f"repos/{args.repo}/code-scanning/analyses?per_page=1"
        control_state, control_detail = _with_retries(
            lambda: classify_analyses_response(gh_get(path, args.timeout)),
            args.attempts,
            args.retry_delay,
        )
    if control_state == CLEAN:
        return CLEAN, f"{detail}; positive control: {control_detail}"
    return control_state, f"0 open alerts were reported, but the zero has no positive control: {control_detail}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"owner/name (default: {DEFAULT_REPO})")
    parser.add_argument("--bypass", default="", help=f"must be exactly '{BYPASS_VALUE}'; only clears the not-enabled state")
    parser.add_argument("--bypass-actor", default="", help="who requested the bypass, for the audit line")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per-request timeout in seconds")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS, help="attempts for transient failures only")
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY, help="seconds between attempts")
    parser.add_argument("--alerts-file", help="read a captured alerts body instead of calling gh (for tests/demos)")
    parser.add_argument("--http-status", type=int, default=200, help="status to pair with --alerts-file")
    parser.add_argument("--analyses-file", help="read a captured analyses body instead of calling gh (for tests/demos)")
    parser.add_argument("--analyses-http-status", type=int, default=200, help="status to pair with --analyses-file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        state, detail = evaluate(args)
    except GateError as exc:
        state, detail = UNREADABLE, str(exc)
    except (OSError, UnicodeDecodeError) as exc:
        state, detail = UNREADABLE, f"could not read the payload: {exc}"

    code, lines = report(state, detail, args.bypass, args.bypass_actor)
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
