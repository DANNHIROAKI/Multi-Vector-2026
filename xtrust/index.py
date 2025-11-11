"""Index construction for the XTRust late interaction engine (pure Python)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .quantization import ClusterEnvelope, ResidualQuantizer
from .signature import BloomSignature
from .vector_utils import normalize_rows


@dataclass
class EncodedToken:
    doc_id: str
    cluster_id: int
    code: List[int]


@dataclass
class DocumentEntry:
    doc_id: str
    cluster_signature: BloomSignature
    cluster_set: set[int]
    length: int


@dataclass
class XTRustIndex:
    dimension: int
    envelopes: Dict[int, ClusterEnvelope]
    cluster_tokens: Dict[int, List[EncodedToken]]
    documents: Dict[str, DocumentEntry]

    def iter_cluster_tokens(self, cluster_id: int) -> List[EncodedToken]:
        return self.cluster_tokens.get(cluster_id, [])


class IndexBuilder:
    def __init__(self, dimension: int, num_clusters: int = 16, residual_bits: int = 4, seed: int | None = None):
        self.dimension = dimension
        self.num_clusters = num_clusters
        self.residual_bits = residual_bits
        self.seed = seed or 0
        self._documents: Dict[str, List[List[float]]] = {}

    def add_document(self, doc_id: str, tokens: List[List[float]]) -> None:
        if any(len(row) != self.dimension for row in tokens):
            raise ValueError("Token matrix has inconsistent dimensionality")
        self._documents[doc_id] = normalize_rows(tokens)

    def build(self) -> XTRustIndex:
        if not self._documents:
            raise ValueError("No documents added to the index")

        token_matrix: List[List[float]] = []
        token_docs: List[str] = []
        doc_offsets: Dict[str, List[int]] = {}
        for doc_id, tokens in self._documents.items():
            indices: List[int] = []
            for token in tokens:
                indices.append(len(token_matrix))
                token_matrix.append(token)
                token_docs.append(doc_id)
            doc_offsets[doc_id] = indices

        quantizer = ResidualQuantizer(self.num_clusters, self.residual_bits, seed=self.seed)
        centroids, buckets, envelopes, assignments = quantizer.fit(token_matrix)

        cluster_tokens: Dict[int, List[EncodedToken]] = {cid: [] for cid in range(self.num_clusters)}
        doc_entries: Dict[str, DocumentEntry] = {}

        for token_idx, doc_id in enumerate(token_docs):
            cluster_id = assignments[token_idx]
            envelope = envelopes[cluster_id]
            residual = [token_matrix[token_idx][d] - envelope.centroid[d] for d in range(self.dimension)]
            codes: List[int] = []
            for d in range(self.dimension):
                candidates = envelope.residual_codebook[d]
                code = min(range(len(candidates)), key=lambda c: abs(candidates[c] - residual[d]))
                codes.append(code)
            cluster_tokens[cluster_id].append(EncodedToken(doc_id=doc_id, cluster_id=cluster_id, code=codes))

        for doc_id, indices in doc_offsets.items():
            cluster_set: set[int] = set(assignments[i] for i in indices)
            signature = BloomSignature.create(size=len(cluster_set))
            signature.update(cluster_set)
            doc_entries[doc_id] = DocumentEntry(
                doc_id=doc_id,
                cluster_signature=signature,
                cluster_set=cluster_set,
                length=len(indices),
            )

        return XTRustIndex(
            dimension=self.dimension,
            envelopes=envelopes,
            cluster_tokens=cluster_tokens,
            documents=doc_entries,
        )
