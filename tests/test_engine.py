from __future__ import annotations

import random

from xtrust.engine import XTRustEngine
from xtrust.planner import PlanOptimizer, PlanRequest


def build_small_engine() -> XTRustEngine:
    rng = random.Random(42)
    docs = {}
    for doc_idx in range(10):
        tokens = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(6)]
        docs[f"doc-{doc_idx}"] = tokens
    return XTRustEngine.from_documents(docs, num_clusters=6, residual_bits=3, seed=0)


def random_query(seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(8)] for _ in range(3)]


def test_search_returns_ranked_documents():
    engine = build_small_engine()
    query = random_query(1)
    result = engine.search(query, k=3, epsilon=0.05)
    assert len(result.documents) == 3
    lowers = [doc.lower for doc in result.documents]
    assert lowers == sorted(lowers, reverse=True)
    assert result.certificate.mode == "deterministic"
    assert "probes" in result.diagnostics


def test_probability_certificate_converges():
    engine = build_small_engine()
    query = random_query(2)
    result = engine.search(query, k=2, delta=0.1, epsilon=0.2)
    assert result.certificate.mode == "probabilistic"
    assert result.certificate.gap <= 0.2 or not result.certificate.satisfied


def test_planner_outputs_consistent_plan():
    engine = build_small_engine()
    planner = PlanOptimizer(engine.index)
    plan = planner.plan(PlanRequest(query_tokens=3, target_recall=0.85, latency_budget_ms=20.0))
    assert plan.nprobe > 0
    assert plan.max_probes >= plan.nprobe
    assert isinstance(plan.expected_latency_ms, float)

