from __future__ import annotations

import random

from xtrust.engine import XTRustEngine
from xtrust.planner import PlanOptimizer, PlanRequest
from xtrust.vector_utils import normalize_rows, relu_dot


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


def exact_scores(engine: XTRustEngine, query: list[list[float]]) -> dict[str, float]:
    norm_query = normalize_rows(query)
    scores: dict[str, float] = {}
    for doc_id in engine.index.documents:
        decoded = []
        for cluster_tokens in engine.index.cluster_tokens.values():
            for token in cluster_tokens:
                if token.doc_id == doc_id:
                    decoded.append((token.token_index, engine.index.decode_token(token)))
        decoded.sort(key=lambda item: item[0])
        totals = []
        for q in norm_query:
            best = 0.0
            for _, vector in decoded:
                best = max(best, relu_dot(q, vector))
            totals.append(best)
        scores[doc_id] = sum(totals) / len(totals)
    return scores


def test_search_returns_ranked_documents():
    engine = build_small_engine()
    query = random_query(1)
    result = engine.search(query, k=3, epsilon=0.05)
    assert len(result.documents) == 3
    scores = [doc.score for doc in result.documents]
    assert scores == sorted(scores, reverse=True)
    assert result.certificate.mode == "deterministic"
    assert "probes" in result.diagnostics


def test_probability_certificate_converges():
    engine = build_small_engine()
    query = random_query(2)
    result = engine.search(query, k=2, delta=0.1, epsilon=0.2)
    assert result.certificate.mode == "probabilistic"
    assert result.certificate.gap <= 0.2 or not result.certificate.satisfied


def test_bounds_cover_exact_scores():
    engine = build_small_engine()
    query = random_query(3)
    result = engine.search(query, k=4, epsilon=0.1)
    truth = exact_scores(engine, query)
    for doc in result.documents:
        actual = truth[doc.doc_id]
        assert doc.lower - 1e-6 <= actual <= doc.upper + 1e-6
        assert actual <= doc.prob_upper + 1e-6
    returned_scores = [truth[doc.doc_id] for doc in result.documents]
    ranked_truth = sorted(truth.values(), reverse=True)
    threshold = ranked_truth[3]
    assert min(returned_scores) >= threshold - 1e-3


def test_planner_outputs_consistent_plan():
    engine = build_small_engine()
    planner = PlanOptimizer(engine.index)
    plan = planner.plan(PlanRequest(query_tokens=3, target_recall=0.85, latency_budget_ms=20.0))
    assert plan.nprobe > 0
    assert plan.max_probes >= plan.nprobe
    assert isinstance(plan.expected_latency_ms, float)

