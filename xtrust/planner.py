"""SLO-aware planning utilities for configuring the search engine."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .index import XTRustIndex


@dataclass
class PlanRequest:
    query_tokens: int
    target_recall: float
    latency_budget_ms: Optional[float] = None
    epsilon: float = 0.0
    delta: float = 0.0


@dataclass
class PlanResult:
    nprobe: int
    t_prime: int
    residual_bits: int
    max_probes: int
    certificate_mode: str
    expected_latency_ms: float
    expected_recall: float
    meets_latency: bool


class PlanOptimizer:
    """Simple heuristic-driven planner used by the research prototype."""

    def __init__(self, index: XTRustIndex):
        self.index = index
        self.cluster_count = max(1, len(index.envelopes))
        self.bits = max(1, int(math.log2(next(iter(index.envelopes.values())).levels)))

    def plan(self, request: PlanRequest) -> PlanResult:
        base_probe = max(4, int(math.sqrt(self.cluster_count)))
        quality_factor = max(0.5, min(1.5, request.target_recall / 0.6))
        nprobe = int(min(self.cluster_count, max(base_probe, base_probe * quality_factor)))
        if request.latency_budget_ms is not None:
            latency_limited = max(1, int(request.latency_budget_ms / max(request.query_tokens, 1)))
            nprobe = min(nprobe, latency_limited)
        t_prime = max(4, request.query_tokens * 2)
        residual_bits = self.bits
        max_probes = nprobe * max(1, request.query_tokens)
        certificate_mode = "probabilistic" if request.delta > 0 else "deterministic"
        expected_latency = float(nprobe * max(1, request.query_tokens) * 1.5)
        expected_recall = float(min(0.99, 0.55 + 0.3 * math.log1p(nprobe)))
        meets_latency = request.latency_budget_ms is None or expected_latency <= request.latency_budget_ms
        return PlanResult(
            nprobe=nprobe,
            t_prime=t_prime,
            residual_bits=residual_bits,
            max_probes=max_probes,
            certificate_mode=certificate_mode,
            expected_latency_ms=expected_latency,
            expected_recall=expected_recall,
            meets_latency=meets_latency,
        )

