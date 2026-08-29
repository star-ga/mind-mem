"""mind-mem benchmark harnesses — scalar metrics over the live corpus.

The ``eval_*`` + ``longmemeval_suite`` modules add a pluggable,
self-asserting retrieval-eval harness (see ``eval_adapter`` for the
contract and the honesty rationale).

The ``repo_task_*`` modules mine this repository's own git history into
machine-checkable agent tasks -- the substrate for the with-memory versus
without-memory comparison (see ``repo_task_mining`` for the selection rule).

The ``ab_*`` modules are that comparison: two arms differing only in
whether recalled context is prefixed to the prompt, graded by pytest and
summarised with an exact paired test (see ``ab_harness`` for the design
and ``ab_seed`` for the pre-cutoff seeding guarantee).
"""

from .ab_agent import AgentRequest, AgentResult, get_agent
from .ab_arms import ARM_CONTROL, ARM_MEMORY, Budget, assert_arms_equal, assert_control_isolated, build_prompt
from .ab_cli import main as run_memory_ab
from .ab_grade import Verdict, grade
from .ab_harness import digest as ab_digest
from .ab_harness import run_suite as run_memory_ab_suite
from .ab_seed import SeedLeakError, SeedReport, assert_no_leak, seed_workspace
from .ab_stats import PairedSummary, mcnemar_exact, summarise
from .ab_task import Task, load_task_set, select_tasks
from .eval_adapter import EvalAdapter, PipelineProbe, SessionDoc, config_sha256
from .eval_adapters import Bm25BaselineAdapter, MindMemAdapter, get_adapter
from .eval_scorer import K_VALUES, QuestionScore, aggregate, score_question
from .repo_task_cli import SCHEMA_VERSION, derive_task_statement
from .repo_task_cli import run as generate_repo_tasks
from .repo_task_mining import Candidate, MiningStats, select_candidates
from .repo_task_validation import Validation, validate

__all__ = [
    "ARM_CONTROL",
    "ARM_MEMORY",
    "AgentRequest",
    "AgentResult",
    "Budget",
    "PairedSummary",
    "SeedLeakError",
    "SeedReport",
    "Task",
    "Verdict",
    "ab_digest",
    "assert_arms_equal",
    "assert_control_isolated",
    "assert_no_leak",
    "build_prompt",
    "get_agent",
    "grade",
    "load_task_set",
    "mcnemar_exact",
    "run_memory_ab",
    "run_memory_ab_suite",
    "seed_workspace",
    "select_tasks",
    "summarise",
    "EvalAdapter",
    "PipelineProbe",
    "SessionDoc",
    "config_sha256",
    "Bm25BaselineAdapter",
    "MindMemAdapter",
    "get_adapter",
    "K_VALUES",
    "QuestionScore",
    "aggregate",
    "score_question",
    "Candidate",
    "MiningStats",
    "SCHEMA_VERSION",
    "Validation",
    "derive_task_statement",
    "generate_repo_tasks",
    "select_candidates",
    "validate",
]
