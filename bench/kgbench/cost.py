"""What a bench run cost: wall time, and the provider calls, dollars and waiting behind it.

Lift only means something next to its price (GRA-87), so every results file carries a ``Cost``.
A system that never reaches a provider (BM25, dense retrieval, the offline pipeline) has zero LLM
usage by construction; a real-stack run reads it from the LLM clients through ``llm_usage``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from graph_rag.wiring import LlmUsage


@dataclass(frozen=True)
class Cost:
    wall_seconds: float = 0.0
    llm: LlmUsage = field(default_factory=LlmUsage)

    def to_dict(self) -> dict[str, float]:
        return {
            "wall_seconds": round(self.wall_seconds, 3),
            "llm_calls": self.llm.calls,
            "llm_cache_hits": self.llm.cache_hits,
            "llm_seconds": round(self.llm.seconds, 3),
            "llm_cost_usd": round(self.llm.cost_usd, 6),
        }

    def format_line(self) -> str:
        return (
            f"{self.wall_seconds:.2f}s wall, {self.llm.calls} provider calls "
            f"({self.llm.cache_hits} from cache), ${self.llm.cost_usd:.4f}"
        )


class Meter:
    """``with Meter(components) as m: ...`` then ``m.cost``. ``components`` may be None."""

    def __init__(self, components=None) -> None:
        self._usage = components.llm_usage if components is not None else LlmUsage
        self.cost = Cost()

    def __enter__(self) -> Meter:
        self._before = self._usage()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.cost = Cost(time.perf_counter() - self._start, self._usage() - self._before)
