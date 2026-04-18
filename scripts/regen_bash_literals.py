#!/usr/bin/env python3
"""Regenerate src/mind_mem/_task_status_literals.sh from enums.py.

Run this whenever TaskStatus gains or loses members so that the bash
validator (validate.sh) stays in sync with the canonical Python enum.

Usage:
    python3 scripts/regen_bash_literals.py

Or via make:
    make regen-bash-literals

The generated file is committed to the repo so CI does not need to run
this script during normal test runs.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from mind_mem.enums import TaskStatus  # noqa: E402


def main() -> None:
    values = [s.value for s in TaskStatus]
    pipe_joined = "|".join(values)

    out_path = REPO_ROOT / "src" / "mind_mem" / "_task_status_literals.sh"

    content = textwrap.dedent(
        f"""\
        # AUTO-GENERATED — do not edit by hand.
        # Regenerate with:  python3 scripts/regen_bash_literals.py
        # Source:           src/mind_mem/enums.py (TaskStatus)
        #
        # Provides TASK_STATUS_RE — a pipe-joined alternation for use in
        # bash grep -E patterns.  Must stay in sync with TaskStatus in
        # enums.py; run `make regen-bash-literals` after any enum change.

        TASK_STATUS_RE="{pipe_joined}"
        TASK_STATUS_VALUES=({' '.join(f'"{v}"' for v in values)})
        """
    )

    out_path.write_text(content, encoding="utf-8")
    print(f"Written {out_path.relative_to(REPO_ROOT)}")
    print(f"  TASK_STATUS_RE=\"{pipe_joined}\"")
    print(f"  TASK_STATUS_VALUES=({len(values)} values)")


if __name__ == "__main__":
    main()
