"""
Evaluation harness and judge proof benchmarks.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentinelsentry.eval.benchmark import BenchmarkRunner

__all__ = ["BenchmarkRunner"]


def __getattr__(name: str) -> Any:
    if name == "BenchmarkRunner":
        from sentinelsentry.eval.benchmark import BenchmarkRunner
        return BenchmarkRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
