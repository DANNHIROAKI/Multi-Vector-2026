"""Index construction and streaming maintenance for the XTRust engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional

from .quantization import ClusterEnvelope, ResidualQuantizer
from .signature import BloomSignature
from .vector_utils import l2_norm, normalize_rows


@dataclass
class EncodedToken:
    """Quantised representation of a document token."""

    doc_id: str
    cluster_id: int
    code: List[int]
    token_index: int


@dataclass
class DocumentEntry:
    """Metadata tracked for each document in the index."""

    doc_id: str
    cluster_signature: BloomSignature
    cluster_set: set[int]
    length: int
    tokens: List[List[float]]
    active: bool = True

    def clone(self) -> "DocumentEntry":
        signature = BloomSignature(
            num_bits=self.cluster_signature.num_bits,
            num_hashes=self.cluster_signature.num_hashes,
            bits=bytearray(self.cluster_signature.bits),
            universe=set(self.cluster_signature.universe),
        )
        return DocumentEntry(
            doc_id=self.doc_id,
            cluster_signature=signature,
            cluster_set=set(self.cluster_set),
            length=self.length,
            tokens=[row[:] for row in self.tokens],
            active=self.active,
        )


@dataclass
class XTRustIndex:
    """Immutable view over the indexed corpus."""

    dimension: int
    envelopes: Dict[int, ClusterEnvelope]
    cluster_tokens: Dict[int, List[EncodedToken]]
    documents: Dict[str, DocumentEntry]
    version: int = 0

    def iter_cluster_tokens(self, cluster_id: int) -> Iterable[EncodedToken]:
        return self.cluster_tokens.get(cluster_id, [])

    def decode_token(self, token: EncodedToken) -> List[float]:
        envelope = self.envelopes[token.cluster_id]
        vector = envelope.decode(token.code)
        norm = l2_norm(vector)
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def get_document(self, doc_id: str) -> Optional[DocumentEntry]:
        return self.documents.get(doc_id)

    def materialise_documents(self) -> Dict[str, List[List[float]]]:
        return {doc_id: entry.tokens[:] for doc_id, entry in self.documents.items() if entry.active}

    def with_updates(
        self,
        added: MutableMapping[str, List[List[float]]],
        removed: Iterable[str] | None = None,
        num_clusters: Optional[int] = None,
        residual_bits: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "XTRustIndex":
        """Return a new index rebuilt with the specified updates applied."""

        documents = self.materialise_documents()
        if removed:
            for doc_id in removed:
                documents.pop(doc_id, None)
        for doc_id, tokens in added.items():
            documents[doc_id] = tokens
        if not documents:
            raise ValueError("Cannot build index without documents")

        sample_doc = next(iter(documents.values()))
        dimension = len(sample_doc[0])
        builder = IndexBuilder(
            dimension=dimension,
            num_clusters=num_clusters or len(self.envelopes),
            residual_bits=residual_bits or next(iter(self.envelopes.values())).levels,
            seed=seed or 0,
        )
        for doc_id, tokens in documents.items():
            builder.add_document(doc_id, tokens)
        rebuilt = builder.build()
        rebuilt.version = self.version + 1
        return rebuilt


class IndexBuilder:
    """Helper responsible for constructing :class:`XTRustIndex` instances."""

    def __init__(self, dimension: int, num_clusters: int = 16, residual_bits: int = 4, seed: int | None = None):
        self.dimension = dimension
        self.num_clusters = num_clusters
        self.residual_bits = residual_bits
        self.seed = seed or 0
        self._documents: Dict[str, List[List[float]]] = {}

    def add_document(self, doc_id: str, tokens: List[List[float]]) -> None:
        if not tokens:
            raise ValueError("Document must contain at least one token")
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
        centroids, envelopes, assignments = quantizer.fit(token_matrix)

        cluster_tokens: Dict[int, List[EncodedToken]] = {cid: [] for cid in range(self.num_clusters)}
        doc_entries: Dict[str, DocumentEntry] = {}

        for token_idx, doc_id in enumerate(token_docs):
            cluster_id = assignments[token_idx]
            envelope = envelopes[cluster_id]
            residual = [token_matrix[token_idx][d] - envelope.centroid[d] for d in range(self.dimension)]
            code = envelope.encode(residual, update_stats=False)
            cluster_tokens[cluster_id].append(
                EncodedToken(doc_id=doc_id, cluster_id=cluster_id, code=code, token_index=token_idx)
            )

        for doc_id, indices in doc_offsets.items():
            cluster_set: set[int] = set(assignments[i] for i in indices)
            signature = BloomSignature.create(size=max(len(cluster_set), 1))
            signature.update(cluster_set)
            doc_entries[doc_id] = DocumentEntry(
                doc_id=doc_id,
                cluster_signature=signature,
                cluster_set=cluster_set,
                length=len(indices),
                tokens=[row[:] for row in self._documents[doc_id]],
            )

        return XTRustIndex(
            dimension=self.dimension,
            envelopes=envelopes,
            cluster_tokens=cluster_tokens,
            documents=doc_entries,
            version=0,
        )


class StreamingIndex:
    """Utility that manages append-only updates using shadow rebuilds."""

    def __init__(self, base_index: XTRustIndex):
        self._active = base_index
        self._pending_add: Dict[str, List[List[float]]] = {}
        self._pending_remove: set[str] = set()
        self._shadow: Optional[XTRustIndex] = None

    @property
    def active(self) -> XTRustIndex:
        return self._active

    def append_document(self, doc_id: str, tokens: List[List[float]]) -> None:
        self._pending_add[doc_id] = normalize_rows(tokens)
        self._pending_remove.discard(doc_id)

    def remove_document(self, doc_id: str) -> None:
        if doc_id in self._pending_add:
            del self._pending_add[doc_id]
        self._pending_remove.add(doc_id)

    def stage_shadow(
        self,
        num_clusters: Optional[int] = None,
        residual_bits: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not self._pending_add and not self._pending_remove:
            return
        self._shadow = self._active.with_updates(
            added=self._pending_add,
            removed=self._pending_remove,
            num_clusters=num_clusters,
            residual_bits=residual_bits,
            seed=seed,
        )

    def promote_shadow(self) -> None:
        if self._shadow is None:
            self.stage_shadow()
        if self._shadow is None:
            return
        self._active = self._shadow
        self._shadow = None
        self._pending_add.clear()
        self._pending_remove.clear()

