from __future__ import annotations

import random

from xtrust.audit import audit_index
from xtrust.index import IndexBuilder


def _build_index() -> IndexBuilder:
    rng = random.Random(4)
    builder = IndexBuilder(dimension=4, num_clusters=3, residual_bits=2, seed=0)
    for doc_idx in range(4):
        tokens = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(5)]
        builder.add_document(f"doc-{doc_idx}", tokens)
    return builder


def test_audit_passes_for_consistent_index():
    index = _build_index().build()
    report = audit_index(index, sample_queries=8)
    assert report.succeeded
    assert report.stats["upper_bound_checks"] > 0


def test_audit_detects_signature_issue():
    index = _build_index().build()
    entry = index.documents["doc-0"]
    entry.cluster_signature.bits = bytearray(len(entry.cluster_signature.bits))
    entry.cluster_signature.universe.clear()
    report = audit_index(index, sample_queries=4)
    assert not report.succeeded

