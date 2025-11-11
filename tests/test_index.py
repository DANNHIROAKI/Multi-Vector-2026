from __future__ import annotations

import random

from xtrust.index import IndexBuilder


def test_index_builder_constructs_envelopes():
    rng = random.Random(0)
    docs = {
        "a": [[rng.gauss(0, 1) for _ in range(6)] for _ in range(4)],
        "b": [[rng.gauss(0, 1) for _ in range(6)] for _ in range(5)],
    }
    builder = IndexBuilder(dimension=6, num_clusters=4, residual_bits=3, seed=0)
    for doc_id, tokens in docs.items():
        builder.add_document(doc_id, tokens)
    index = builder.build()
    assert index.dimension == 6
    assert len(index.envelopes) == 4
    doc_entry = index.documents["a"]
    assert doc_entry.length == 4
    assert doc_entry.cluster_signature.contains(next(iter(doc_entry.cluster_set)))
