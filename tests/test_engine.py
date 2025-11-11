from __future__ import annotations

import random

from xtrust.engine import XTRustEngine


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
    results, diagnostics = engine.search(query, k=3)
    assert len(results) == 3
    assert diagnostics["probes"] > 0
    lowers = [doc.lower for doc in results]
    assert lowers == sorted(lowers, reverse=True)


def test_probability_certificate_converges():
    engine = build_small_engine()
    query = random_query(2)
    results, _ = engine.search(query, k=2, delta=0.1, epsilon=0.05)
    for doc in results:
        assert doc.prob_upper + 1e-6 >= doc.lower
