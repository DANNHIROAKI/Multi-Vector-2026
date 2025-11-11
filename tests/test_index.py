from __future__ import annotations

import random

from xtrust.audit import audit_index
from xtrust.index import IndexBuilder
from xtrust.engine import XTRustEngine


def _build_docs(seed: int = 0) -> dict[str, list[list[float]]]:
    rng = random.Random(seed)
    return {
        "a": [[rng.gauss(0, 1) for _ in range(6)] for _ in range(4)],
        "b": [[rng.gauss(0, 1) for _ in range(6)] for _ in range(5)],
        "c": [[rng.gauss(0, 1) for _ in range(6)] for _ in range(3)],
    }


def test_index_builder_constructs_envelopes():
    docs = _build_docs()
    builder = IndexBuilder(dimension=6, num_clusters=4, residual_bits=3, seed=0)
    for doc_id, tokens in docs.items():
        builder.add_document(doc_id, tokens)
    index = builder.build()
    assert index.dimension == 6
    assert len(index.envelopes) == 4
    doc_entry = index.documents["a"]
    assert doc_entry.length == len(doc_entry.tokens)
    report = audit_index(index)
    assert report.succeeded


def test_streaming_promote_shadow_updates_version():
    docs = _build_docs()
    engine = XTRustEngine.from_documents(docs, num_clusters=4, residual_bits=3, seed=0)
    streaming = engine.with_streaming()
    original_version = streaming.active.version
    streaming.append_document("d", docs["a"])
    streaming.remove_document("b")
    streaming.stage_shadow()
    streaming.promote_shadow()
    assert streaming.active.version == original_version + 1
    assert "d" in streaming.active.documents
    assert "b" not in streaming.active.documents

