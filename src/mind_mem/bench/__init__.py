"""mind-mem benchmark harnesses — scalar metrics over the live corpus.

The ``eval_*`` + ``longmemeval_suite`` modules add a pluggable,
self-asserting retrieval-eval harness (see ``eval_adapter`` for the
contract and the honesty rationale).
"""

from .eval_adapter import EvalAdapter, PipelineProbe, SessionDoc, config_sha256
from .eval_adapters import Bm25BaselineAdapter, MindMemAdapter, get_adapter
from .eval_scorer import K_VALUES, QuestionScore, aggregate, score_question

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
]
