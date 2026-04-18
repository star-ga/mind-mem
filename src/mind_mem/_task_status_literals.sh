# AUTO-GENERATED — do not edit by hand.
# Regenerate with:  python3 scripts/regen_bash_literals.py
# Source:           src/mind_mem/enums.py (TaskStatus)
#
# Provides TASK_STATUS_RE — a pipe-joined alternation for use in
# bash grep -E patterns.  Must stay in sync with TaskStatus in
# enums.py; run `make regen-bash-literals` after any enum change.

TASK_STATUS_RE="todo|doing|blocked|done|canceled"
TASK_STATUS_VALUES=("todo" "doing" "blocked" "done" "canceled")
