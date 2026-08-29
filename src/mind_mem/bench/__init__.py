"""mind-mem benchmark harnesses — scalar metrics over the live corpus.

The ``eval_*`` + ``longmemeval_suite`` modules add a pluggable,
self-asserting retrieval-eval harness (see ``eval_adapter`` for the
contract and the honesty rationale).

The ``repo_task_*`` modules mine this repository's own git history into
machine-checkable agent tasks -- the substrate for the with-memory versus
without-memory comparison (see ``repo_task_mining`` for the selection rule).
"""

from .eval_adapter import EvalAdapter, PipelineProbe, SessionDoc, config_sha256
from .eval_adapters import Bm25BaselineAdapter, MindMemAdapter, get_adapter
from .eval_scorer import K_VALUES, QuestionScore, aggregate, score_question
from .repo_task_cli import SCHEMA_VERSION, derive_task_statement
from .repo_task_cli import run as generate_repo_tasks
from .repo_task_mining import Candidate, MiningStats, select_candidates
from .repo_task_validation import Validation, validate

__all__ = [
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
