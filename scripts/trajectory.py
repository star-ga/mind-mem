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
import sys
from configparser import ConfigParser
from datetime import date, datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Trajectory block ID pattern: TRAJ-YYYYMMDD-NNN
_TRAJ_ID_RE = re.compile(r"^TRAJ-(\d{8})-(\d{3,})$")

# Valid outcome values
_VALID_OUTCOMES = {"SUCCESS", "FAILURE", "PARTIAL", "ABORTED"}


def _load_config() -> dict:
    """Load trajectory.mind config with defaults."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "mind", "trajectory.mind"
    )
    defaults = {
        "recall_limit": 5,
        "recency_halflife": 30,
        "outcome_weight": 0.3,
        "tool_overlap_boost": 1.5,
        "min_duration": 60,
        "min_tool_calls": 3,
    }
    if os.path.isfile(config_path):
        cp = ConfigParser()
        cp.read(config_path)
        if cp.has_section("recall"):
            for k in ("recall_limit", "recency_halflife"):
                if cp.has_option("recall", k):
                    defaults[k] = int(cp.get("recall", k))
            for k in ("outcome_weight", "tool_overlap_boost"):
                if cp.has_option("recall", k):
                    defaults[k] = float(cp.get("recall", k))
        if cp.has_section("capture"):
            for k in ("min_duration", "min_tool_calls"):
                if cp.has_option("capture", k):
                    defaults[k] = int(cp.get("capture", k))
    return defaults


def generate_id(workspace: str | None = None) -> str:
    """Generate next trajectory block ID.

    Format: TRAJ-YYYYMMDD-NNN where NNN is auto-incremented.
    """
    today = date.today().strftime("%Y%m%d")
    prefix = f"TRAJ-{today}-"

    # Find existing trajectories for today
    max_seq = 0
    if workspace:
        traj_dir = os.path.join(workspace, "trajectories")
        if os.path.isdir(traj_dir):
            for fname in os.listdir(traj_dir):
                if fname.startswith("TRAJ-") and fname.endswith(".md"):
                    block_id = fname.replace(".md", "")
                    m = _TRAJ_ID_RE.match(block_id)
                    if m and m.group(1) == today:
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
        errors.append(
            f"Invalid outcome '{outcome}', must be one of: {', '.join(sorted(_VALID_OUTCOMES))}"
        )

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
            item = re.sub(r"^\s+[-\d]+[.)]\s*", "", line).strip()
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

    # Simple fields first
    for key in ("Task", "Date", "Duration", "Tools", "Outcome", "Reward", "Context", "Error"):
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


def compute_similarity(traj_a: dict, traj_b: dict) -> float:
    """Compute similarity between two trajectory blocks.

    Uses task text overlap + tool overlap + outcome matching.
    Returns score in [0.0, 1.0].
    """
    config = _load_config()
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

    return min(score / max(weight_sum, 0.01), 1.0)
