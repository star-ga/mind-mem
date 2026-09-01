#!/usr/bin/env python3
"""Trajectory Memory — task execution trace storage and recall.

Provides trajectory block parsing, validation, storage, and similarity
matching for case-based reasoning. Agents learn from past task outcomes
without fine-tuning.

Block format:
    [TRAJ-20260221-001]
    Task: Deploy v1.0.6 to production
    Date: 2026-02-21
    Duration: 45min
    Tools: git, pytest, docker
    Outcome: SUCCESS
    Reward: 1.0
    Lessons:
      - Always run pytest before tagging
      - Never skip smoke tests on staging
    Steps:
      1. git checkout main && git pull
      2. pytest tests/ -x
      3. docker build -t mind-mem:v1.0.6 .
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Any, Mapping

# Trajectory block ID pattern: TRAJ-YYYYMMDD-NNN
_TRAJ_ID_RE = re.compile(r"^TRAJ-(\d{8})-(\d{3,})$")

# Valid outcome values
_VALID_OUTCOMES = {"SUCCESS", "FAILURE", "PARTIAL", "ABORTED"}

#: v4 feature flag gating BOTH halves of the wiring — the ``report_outcome``
#: capture hook and the ``similar_trajectories`` MCP tool. Default-OFF; with
#: it off no trajectory is ever written and the tool refuses without touching
#: the filesystem.
TRAJECTORY_FLAG = "trajectory"

#: Sub-directory of the workspace holding one ``TRAJ-*.md`` file per capture.
#: A SIDECAR, deliberately not part of the governed corpus: nothing here is
#: recalled by ``recall()``, and promoting a lesson into the corpus still goes
#: through ``propose_update`` -> HITL like every other block.
TRAJECTORY_DIR = "trajectories"

#: The kernel this module is tuned by.
KERNEL_FILENAME = "trajectory.mind"

#: Longest scalar field written into a captured block. Provenance strings
#: arrive from an MCP caller, so they are bounded and flattened at the write
#: boundary (see :func:`_scalar`) rather than trusted.
MAX_FIELD_LEN = 500

#: Verdict vocabulary of ``report_outcome`` -> the trajectory outcome
#: vocabulary declared in ``mind/trajectory.mind [outcome] values``.
#: ``neutral`` is PARTIAL: the work happened, the blocks neither earned nor
#: lost credit for it. It is not ABORTED, which means the work stopped.
_OUTCOME_TO_TRAJECTORY = {
    "success": "SUCCESS",
    "failure": "FAILURE",
    "neutral": "PARTIAL",
}

#: Any whitespace run, including the newlines that would otherwise let a
#: caller-supplied ``evidence`` string forge a second ``Key: value`` line or a
#: whole ``[TRAJ-...]`` header inside a block it does not own.
_WHITESPACE_RUN = re.compile(r"\s+")


#: Knob -> (kernel section, coercion). The defaults are the values the
#: shipped ``mind/trajectory.mind`` declares, so a workspace with no kernel
#: of its own behaves exactly as the shipped one does.
_KNOBS: dict[str, tuple[str, type]] = {
    "recall_limit": ("recall", int),
    "recency_halflife": ("recall", int),
    "outcome_weight": ("recall", float),
    "tool_overlap_boost": ("recall", float),
    "min_duration": ("capture", int),
    "min_tool_calls": ("capture", int),
    "default_reward_success": ("outcome", float),
    "default_reward_partial": ("outcome", float),
    "default_reward_failure": ("outcome", float),
    "default_reward_aborted": ("outcome", float),
}

_DEFAULTS: dict[str, Any] = {
    "recall_limit": 5,
    "recency_halflife": 30,
    "outcome_weight": 0.3,
    "tool_overlap_boost": 1.5,
    "min_duration": 60,
    "min_tool_calls": 3,
    "default_reward_success": 1.0,
    "default_reward_partial": 0.5,
    "default_reward_failure": 0.0,
    "default_reward_aborted": 0.0,
}


def kernel_path(workspace: str | None = None) -> str:
    """Resolve the ``trajectory.mind`` this workspace is tuned by.

    Workspace-local first (``<ws>/mind/trajectory.mind``, via the canonical
    :func:`mind_mem.mind_ffi.get_mind_dir` resolver every other kernel
    consumer uses), then the repo-root kernel that ships with the source.

    Two levels up, not one: ``__file__`` is ``src/mind_mem/trajectory.py``,
    so a single ``".."`` lands in ``src/`` and looked for
    ``src/mind/trajectory.mind``, which has never existed — the kernels live
    at the REPO ROOT. Every knob in this module silently fell back to its
    default, which is exactly why nobody noticed: the shipped kernel's values
    ARE the defaults, so the bug was invisible until a workspace overrode one.
    """
    if workspace:
        from .mind_ffi import get_mind_dir

        candidate = os.path.join(get_mind_dir(workspace), KERNEL_FILENAME)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(os.path.dirname(__file__), "..", "..", "mind", KERNEL_FILENAME)


def _load_config(workspace: str | None = None) -> dict:
    """Load ``trajectory.mind`` knobs, falling back to the shipped defaults.

    Parsing goes through :func:`mind_mem.mind_ffi.load_kernel_config` — the
    one ``.mind`` INI reader the package already has — rather than a second
    private ``ConfigParser``. A knob whose value will not coerce keeps its
    default instead of raising: a typo in a kernel must not take recall down.
    """
    from .mind_ffi import load_kernel_config

    config = dict(_DEFAULTS)
    sections = load_kernel_config(kernel_path(workspace))
    for knob, (section, coerce) in _KNOBS.items():
        if knob not in sections.get(section, {}):
            continue
        try:
            config[knob] = coerce(sections[section][knob])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return config


def generate_id(workspace: str | None = None, on: date | None = None) -> str:
    """Generate next trajectory block ID.

    Format: TRAJ-YYYYMMDD-NNN where NNN is auto-incremented.

    Args:
        workspace: Workspace whose ``trajectories/`` dir is scanned for the
            highest sequence already used on *on*.
        on: The date the id is stamped with. INJECTED by the capture path
            from the outcome's own ``recorded_at``, so a replay of the same
            report lands on the same day it was first recorded rather than
            on whatever day the replay happens. ``None`` reads the local
            clock, preserving the original behaviour for direct callers.
    """
    day = (on or date.today()).strftime("%Y%m%d")
    prefix = f"TRAJ-{day}-"

    # Find existing trajectories for that day
    max_seq = 0
    if workspace:
        traj_dir = os.path.join(workspace, TRAJECTORY_DIR)
        if os.path.isdir(traj_dir):
            for fname in os.listdir(traj_dir):
                if fname.startswith("TRAJ-") and fname.endswith(".md"):
                    block_id = fname.replace(".md", "")
                    m = _TRAJ_ID_RE.match(block_id)
                    if m and m.group(1) == day:
                        max_seq = max(max_seq, int(m.group(2)))

    return f"{prefix}{max_seq + 1:03d}"


def validate_block(block: dict) -> list[str]:
    """Validate a trajectory block. Returns list of error strings (empty = valid)."""
    errors = []

    # Required fields
    for field in ("Task", "Date", "Outcome"):
        if not block.get(field):
            errors.append(f"Missing required field: {field}")

    # Validate ID format
    block_id = block.get("_id", "")
    if block_id and not _TRAJ_ID_RE.match(block_id):
        errors.append(f"Invalid trajectory ID format: {block_id}")

    # Validate outcome
    outcome = block.get("Outcome", "").upper()
    if outcome and outcome not in _VALID_OUTCOMES:
        errors.append(f"Invalid outcome '{outcome}', must be one of: {', '.join(sorted(_VALID_OUTCOMES))}")

    # Validate reward range
    reward = block.get("Reward")
    if reward is not None:
        try:
            r = float(reward)
            if r < 0.0 or r > 1.0:
                errors.append(f"Reward {r} out of range [0.0, 1.0]")
        except (ValueError, TypeError):
            errors.append(f"Reward must be a number, got: {reward}")

    # Validate date format
    date_str = block.get("Date", "")
    if date_str:
        try:
            datetime.strptime(str(date_str), "%Y-%m-%d")
        except ValueError:
            errors.append(f"Invalid date format '{date_str}', expected YYYY-MM-DD")

    return errors


def parse_trajectory_md(text: str) -> dict | None:
    """Parse a trajectory block from Markdown text.

    Expected format:
        [TRAJ-20260221-001]
        Task: Deploy v1.0.6
        Date: 2026-02-21
        ...
    """
    lines = text.strip().splitlines()
    if not lines:
        return None

    # Find block header
    header_match = re.match(r"^\[?(TRAJ-\d{8}-\d{3,})\]?$", lines[0].strip())
    if not header_match:
        return None

    block: dict[str, Any] = {"_id": header_match.group(1)}
    current_list_key: str | None = None
    current_list: list[str] = []

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue

        # List item (indented with - or numbered)
        if re.match(r"^\s+[-\d]", line):
            # The bullet marker must come off. The prior pattern demanded a
            # "." or ")" after the marker, which a numbered step has and a
            # "- " bullet does not, so every Lessons entry came back with its
            # own dash glued to the front -- and the format -> parse roundtrip
            # was not the identity it is documented to be. The two markers are
            # now alternatives, not one pattern that happens to fit numbers.
            item = re.sub(r"^\s+(?:[-*]|\d+[.)])\s*", "", line).strip()
            if item:
                current_list.append(item)
            continue

        # Flush previous list
        if current_list_key and current_list:
            block[current_list_key] = current_list
            current_list = []
            current_list_key = None

        # Key: Value line
        kv_match = re.match(r"^(\w+)\s*:\s*(.*)$", stripped)
        if kv_match:
            key = kv_match.group(1)
            value = kv_match.group(2).strip()
            if value:
                block[key] = value
            else:
                # Empty value = start of list section
                current_list_key = key
                current_list = []

    # Flush final list
    if current_list_key and current_list:
        block[current_list_key] = current_list

    return block


def format_trajectory_md(block: dict) -> str:
    """Format a trajectory block as Markdown."""
    lines = [f"[{block['_id']}]"]

    # Simple fields first. ``Status`` and ``Outcome_Id`` are written by the
    # capture path: the first is what ``admit_corpus`` filters the store on,
    # the second is what makes a replayed ``report_outcome`` recognisable as
    # one already captured.
    for key in ("Task", "Date", "Status", "Duration", "Tools", "Outcome", "Reward", "Context", "Error", "Outcome_Id"):
        if key in block and not isinstance(block[key], list):
            lines.append(f"{key}: {block[key]}")

    # List fields
    for key in ("Lessons", "Steps"):
        if key in block and isinstance(block[key], list):
            lines.append(f"{key}:")
            for i, item in enumerate(block[key], 1):
                if key == "Steps":
                    lines.append(f"  {i}. {item}")
                else:
                    lines.append(f"  - {item}")

    return "\n".join(lines) + "\n"


def compute_similarity(
    traj_a: dict,
    traj_b: dict,
    reference_date: date | None = None,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Compute similarity between two trajectory blocks.

    Uses task text overlap + tool overlap + outcome matching, then applies
    exponential recency decay based on the age of ``traj_b`` (the candidate).

    Args:
        traj_a: Query trajectory (the current task).
        traj_b: Candidate trajectory being compared.
        reference_date: Date to measure age from. Defaults to ``date.today()``.
            :func:`similar_trajectories` always injects it, so the ranking
            loop reads no clock; the default is for direct callers only.
        config: Pre-resolved kernel knobs. Injected by the ranking loop so
            the kernel is read once per query rather than once per candidate
            — and so the scoring function is a pure function of its
            arguments, with no filesystem read of its own.

    Returns:
        Score in [0.0, 1.0].
    """
    config = _load_config() if config is None else config
    score = 0.0
    weight_sum = 0.0

    # Task text overlap (Jaccard on words)
    task_a = set(str(traj_a.get("Task", "")).lower().split())
    task_b = set(str(traj_b.get("Task", "")).lower().split())
    if task_a or task_b:
        jaccard = len(task_a & task_b) / max(len(task_a | task_b), 1)
        score += jaccard * 0.5
        weight_sum += 0.5

    # Tool overlap
    tools_a = set(str(traj_a.get("Tools", "")).lower().replace(",", " ").split())
    tools_b = set(str(traj_b.get("Tools", "")).lower().replace(",", " ").split())
    if tools_a or tools_b:
        tool_jaccard = len(tools_a & tools_b) / max(len(tools_a | tools_b), 1)
        tool_boost = config.get("tool_overlap_boost", 1.5)
        score += tool_jaccard * 0.3 * tool_boost
        weight_sum += 0.3

    # Outcome match
    outcome_w = config.get("outcome_weight", 0.3)
    if traj_a.get("Outcome") and traj_b.get("Outcome"):
        outcome_match = 1.0 if traj_a["Outcome"] == traj_b["Outcome"] else 0.0
        score += outcome_match * outcome_w
        weight_sum += outcome_w

    raw_score = min(score / max(weight_sum, 0.01), 1.0)

    # Recency decay — exponential half-life on traj_b's age
    decay = 1.0
    date_str = traj_b.get("Date", "")
    if date_str:
        try:
            traj_date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
            ref = reference_date if reference_date is not None else date.today()
            age_days = max((ref - traj_date).days, 0)
            halflife = max(config.get("recency_halflife", 30), 1)
            decay = 0.5 ** (age_days / halflife)
        except (ValueError, TypeError, ZeroDivisionError):
            decay = 1.0

    return min(raw_score * decay, 1.0)


# ---------------------------------------------------------------------------
# Store — a workspace sidecar, admission-filtered on read
# ---------------------------------------------------------------------------


def trajectory_dir(workspace: str) -> str:
    """The directory holding this workspace's ``TRAJ-*.md`` files."""
    return os.path.join(workspace, TRAJECTORY_DIR)


def _scalar(value: Any, limit: int = MAX_FIELD_LEN) -> str:
    """Flatten *value* into one bounded, single-line field value.

    Provenance strings (``task_id``, ``evidence``, ``tool_id``) come from
    whoever called ``report_outcome``. A newline in one of them would end the
    field's line and let the rest be re-read as a second ``Key: value`` pair —
    or as a whole ``[TRAJ-...]`` header — inside a block the caller does not
    own. Every whitespace run therefore collapses to a single space before the
    value is written, and the result is truncated rather than trusted.
    """
    text = _WHITESPACE_RUN.sub(" ", str(value)).strip()
    return text[:limit]


def write_trajectory(workspace: str, block: Mapping[str, Any]) -> str:
    """Write one validated trajectory block into ``<ws>/trajectories/``.

    Returns the path written. Raises :class:`ValueError` when the block does
    not validate — the store is small and hand-inspectable, and a malformed
    block in it is worse than a refused capture.

    This is NOT a corpus write. The governed ``propose_update`` -> HITL gate
    owns ``memory/``; this sidecar is the same class of artifact as the
    calibration database that ``report_outcome`` already appends to, and
    nothing here is ever served by ``recall()``.
    """
    errors = validate_block(dict(block))
    if errors:
        raise ValueError("; ".join(errors))
    block_id = str(block["_id"])
    if not _TRAJ_ID_RE.match(block_id):
        raise ValueError(f"Invalid trajectory ID format: {block_id}")

    directory = trajectory_dir(workspace)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{block_id}.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(format_trajectory_md(dict(block)))
    return path


def load_trajectories(workspace: str) -> list[dict]:
    """Every servable trajectory block in *workspace*, ordered by id.

    **Admission-filtered.** The parsed blocks go through
    :func:`mind_mem.admissibility.admit_corpus` before any caller sees them,
    exactly like every other block-reading leg in the package. Selecting a
    status is not filtering on it: a file carrying ``Status: quarantined`` or
    ``Status: pending`` — or a status nobody has named, which the allow-list
    withholds fail-closed — is dropped here and can never reach the MCP
    surface.
    """
    from .admissibility import admit_corpus

    directory = trajectory_dir(workspace)
    if not os.path.isdir(directory):
        return []

    blocks: list[dict] = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".md") or not _TRAJ_ID_RE.match(fname[:-3]):
            continue
        try:
            with open(os.path.join(directory, fname), "r", encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        parsed = parse_trajectory_md(text)
        if parsed is not None and parsed.get("_id") == fname[:-3]:
            blocks.append(parsed)

    return admit_corpus(blocks)


# ---------------------------------------------------------------------------
# Capture — driven by report_outcome, flag-gated at the call site
# ---------------------------------------------------------------------------


def outcome_to_block(
    result: Mapping[str, Any],
    *,
    on: date,
    block_id: str,
) -> dict[str, Any]:
    """Project one ``report_outcome`` result into a trajectory block.

    The reward comes from the kernel's ``[outcome] default_reward_*`` knobs
    via *config* resolution at the call site, so a workspace that values a
    partial outcome differently gets its own number.
    """
    verdict = str(result.get("outcome", "")).strip().lower()
    outcome = _OUTCOME_TO_TRAJECTORY.get(verdict, "PARTIAL")
    task = _scalar(result.get("task_id") or "") or f"outcome {result.get('outcome_id', '')[:12]}"

    block: dict[str, Any] = {
        "_id": block_id,
        "Task": task,
        "Date": on.isoformat(),
        # Captured trajectories are minted ACTIVE and say so. The status is
        # written, not implied: an unstated status is servable by package
        # convention, so leaving it off would make the admit_corpus call on
        # the read side decorative for exactly the blocks we mint.
        "Status": "active",
        "Outcome": outcome,
        "Outcome_Id": _scalar(result.get("outcome_id") or "", 64),
    }
    tools = _scalar(result.get("tool_id") or "")
    if tools:
        block["Tools"] = tools
    context = _scalar(result.get("session_id") or "")
    if context:
        block["Context"] = context
    evidence = _scalar(result.get("evidence") or "")
    if evidence:
        block["Lessons"] = [evidence]
    return block


def capture_from_outcome(workspace: str, result: Mapping[str, Any]) -> str | None:
    """Capture a ``report_outcome`` result as a trajectory. Never raises.

    Returns the path written, or ``None`` when nothing was captured. Nothing
    is captured when the report was a REPLAY (``idempotent``): the outcome id
    is the SHA-256 of the canonical payload, so a replay records no new
    evidence and must not mint a second trajectory for the same event.

    The date is taken from the outcome's own ``recorded_at`` — the *stored*
    stamp, which a replay preserves — never from the clock, so a capture is a
    function of the report it describes.

    Failure here is contained on purpose: recording an outcome is the
    caller's actual request, and a full trajectories/ disk must not turn a
    successful outcome report into an error.
    """
    if result.get("idempotent"):
        return None

    try:
        on = _recorded_date(result.get("recorded_at"))
        if on is None:
            return None
        config = _load_config(workspace)
        block = outcome_to_block(result, on=on, block_id=generate_id(workspace, on=on))
        block["Reward"] = _reward_for(block["Outcome"], config)
        return write_trajectory(workspace, block)
    except (OSError, ValueError) as exc:  # pragma: no cover - degradation path
        from .observability import get_logger

        get_logger("trajectory").warning("trajectory_capture_failed", error=str(exc))
        return None


def _recorded_date(stamp: Any) -> date | None:
    """The UTC calendar date of an outcome's ``recorded_at`` stamp."""
    if not isinstance(stamp, str) or len(stamp) < 10:
        return None
    try:
        return date.fromisoformat(stamp[:10])
    except ValueError:
        return None


def _reward_for(outcome: str, config: Mapping[str, Any]) -> float:
    """Kernel-configured reward for a trajectory outcome."""
    return float(config.get(f"default_reward_{outcome.lower()}", 0.0))


# ---------------------------------------------------------------------------
# Recall — deterministic given (store, kernel, scoring_instant)
# ---------------------------------------------------------------------------


def similar_trajectories(
    workspace: str,
    task: str,
    *,
    tools: str = "",
    outcome: str = "",
    limit: int | None = None,
    scoring_instant: date | str | None = None,
) -> list[dict[str, Any]]:
    """Rank this workspace's captured trajectories against *task*.

    Case-based reasoning, not retrieval: the answer is "here is what happened
    the last few times something like this was attempted", newest first among
    equally-similar matches because the recency half-life discounts the rest.

    **Deterministic given (store, kernel, scoring_instant).** The instant is
    resolved ONCE here, at the boundary, exactly as ``recall()`` does it — the
    ranking loop below is then a pure function of its inputs. Pass
    ``scoring_instant`` and the whole call is clock-free; omit it and exactly
    one clock read happens, before any scoring.

    Args:
        workspace: Workspace root holding ``trajectories/``.
        task: The task being attempted now.
        tools: Comma- or space-separated tools in play, for the overlap term.
        outcome: Bias toward trajectories that ended this way.
        limit: Max results; defaults to the kernel's ``recall_limit``.
        scoring_instant: UTC date the recency decay measures age from.

    Returns:
        A list of ranked dicts, highest score first, ties broken by block id
        so the order never depends on directory iteration.
    """
    from .scoring_instant import format_scoring_instant, resolve_scoring_instant

    instant = resolve_scoring_instant(scoring_instant)
    config = _load_config(workspace)
    cap = config["recall_limit"] if limit is None else limit
    if cap <= 0:
        return []

    query = {"Task": task, "Tools": tools, "Outcome": outcome}
    ranked = [
        {
            "id": candidate.get("_id", ""),
            "score": round(compute_similarity(query, candidate, reference_date=instant, config=config), 6),
            "task": candidate.get("Task", ""),
            "date": candidate.get("Date", ""),
            "outcome": candidate.get("Outcome", ""),
            "reward": _reward_value(candidate.get("Reward")),
            "tools": candidate.get("Tools", ""),
            "lessons": _as_list(candidate.get("Lessons")),
            "scoring_instant": format_scoring_instant(instant),
        }
        for candidate in load_trajectories(workspace)
    ]
    ranked.sort(key=lambda row: (-row["score"], row["id"]))
    return ranked[:cap]


def _reward_value(raw: Any) -> float | None:
    """A block's ``Reward`` as a number, or ``None`` when it has none.

    The store is Markdown, so every scalar comes back as a string; a JSON
    consumer ranking on reward should not have to re-parse it. An
    unparseable value reads as absent rather than as zero — zero is a real
    reward (a FAILURE scores it) and must not double as "unknown".
    """
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[str]:
    """Normalise a block field that may be a scalar, a list, or absent."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


__all__ = [
    "KERNEL_FILENAME",
    "MAX_FIELD_LEN",
    "TRAJECTORY_DIR",
    "TRAJECTORY_FLAG",
    "capture_from_outcome",
    "compute_similarity",
    "format_trajectory_md",
    "generate_id",
    "kernel_path",
    "load_trajectories",
    "outcome_to_block",
    "parse_trajectory_md",
    "similar_trajectories",
    "trajectory_dir",
    "validate_block",
    "write_trajectory",
]
