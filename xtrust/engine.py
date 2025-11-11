"""Query execution engine for the pure Python XTRust prototype."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .index import EncodedToken, IndexBuilder, XTRustIndex
from .quantization import ClusterEnvelope
from .vector_utils import add, dot, ints, mean, normalize_rows, relu_dot, zeros


@dataclass
class DocumentBounds:
    doc_id: str
    score: float
    lower: float
    upper: float
    prob_upper: float


class XTRustEngine:
    def __init__(self, index: XTRustIndex):
        self.index = index

    @classmethod
    def from_documents(
        cls,
        documents: Dict[str, List[List[float]]],
        num_clusters: int = 16,
        residual_bits: int = 4,
        seed: int | None = None,
    ) -> "XTRustEngine":
        dimension = len(next(iter(documents.values()))[0])
        builder = IndexBuilder(dimension=dimension, num_clusters=num_clusters, residual_bits=residual_bits, seed=seed)
        for doc_id, tokens in documents.items():
            builder.add_document(doc_id, tokens)
        return cls(builder.build())

    def search(
        self,
        query_tokens: Sequence[Sequence[float]],
        k: int = 10,
        delta: float = 0.0,
        epsilon: float = 0.0,
        max_probes: Optional[int] = None,
    ) -> Tuple[List[DocumentBounds], Dict[str, float]]:
        query_tokens = normalize_rows(query_tokens)
        num_tokens = len(query_tokens)
        visited: List[set[int]] = [set() for _ in range(num_tokens)]
        heaps: List[List[Tuple[float, int]]] = []
        for q_idx, query in enumerate(query_tokens):
            heap: List[Tuple[float, int]] = []
            for cid, envelope in self.index.envelopes.items():
                bound = envelope.upper_bound(query)
                heapq.heappush(heap, (-bound, cid))
            heaps.append(heap)

        doc_obs: Dict[str, List[float]] = {}
        doc_clusters: Dict[str, List[int]] = {}
        doc_bounds: Dict[str, DocumentBounds] = {}
        candidate_docs: set[str] = set()
        probe_count = 0
        total_events = max(1, len(self.index.documents) * num_tokens)

        def decode_token(envelope: ClusterEnvelope, token: EncodedToken) -> List[float]:
            residual = [envelope.residual_codebook[d][code] for d, code in enumerate(token.code)]
            return add(envelope.centroid, residual)

        def compute_bounds(doc_id: str) -> None:
            obs = doc_obs.setdefault(doc_id, zeros(num_tokens))
            clusters = doc_clusters.setdefault(doc_id, ints(num_tokens))
            lower_terms = zeros(num_tokens)
            upper_terms = zeros(num_tokens)
            prob_upper_terms = zeros(num_tokens)
            for q_idx in range(num_tokens):
                query = query_tokens[q_idx]
                cluster_id = clusters[q_idx]
                envelope = self.index.envelopes.get(cluster_id) if cluster_id >= 0 else None
                obs_val = obs[q_idx]
                if envelope is None:
                    upper_terms[q_idx] = self._max_unvisited(doc_id, q_idx, query_tokens, visited, delta, total_events, probabilistic=False)
                    prob_upper_terms[q_idx] = self._max_unvisited(doc_id, q_idx, query_tokens, visited, delta, total_events, probabilistic=True)
                    continue
                error = sum(abs(q) * err for q, err in zip(query, envelope.residual_error))
                lower_terms[q_idx] = max(obs_val - error, 0.0)
                upper_terms[q_idx] = max(
                    obs_val,
                    self._max_unvisited(doc_id, q_idx, query_tokens, visited, delta, total_events, probabilistic=False),
                )
                prob_upper_terms[q_idx] = max(
                    obs_val,
                    self._max_unvisited(doc_id, q_idx, query_tokens, visited, delta, total_events, probabilistic=True),
                )
            lower = mean(lower_terms)
            upper = mean(upper_terms)
            prob_upper = mean(prob_upper_terms)
            score = mean(max(obs_val, upper_val) for obs_val, upper_val in zip(obs, upper_terms))
            doc_bounds[doc_id] = DocumentBounds(doc_id=doc_id, score=score, lower=lower, upper=upper, prob_upper=prob_upper)

        while True:
            if max_probes is not None and probe_count >= max_probes:
                break
            next_choice = self._next_cluster(heaps, visited)
            if next_choice is None:
                break
            q_idx, cluster_id = next_choice
            visited[q_idx].add(cluster_id)
            envelope = self.index.envelopes[cluster_id]
            query = query_tokens[q_idx]
            for token in self.index.iter_cluster_tokens(cluster_id):
                doc_id = token.doc_id
                vector = decode_token(envelope, token)
                score = relu_dot(query, vector)
                obs = doc_obs.setdefault(doc_id, zeros(num_tokens))
                clusters = doc_clusters.setdefault(doc_id, ints(num_tokens))
                if score > obs[q_idx]:
                    obs[q_idx] = score
                    clusters[q_idx] = cluster_id
                candidate_docs.add(doc_id)
            for doc_id in candidate_docs:
                compute_bounds(doc_id)
            probe_count += 1
            if self._should_stop(doc_bounds, heaps, visited, query_tokens, delta, epsilon, total_events, k):
                break

        diagnostics = {
            "probes": probe_count,
            "num_candidates": len(candidate_docs),
            "global_upper": self._global_upper(heaps, visited, query_tokens, delta, total_events, probabilistic=False),
        }
        ranked = sorted(doc_bounds.values(), key=lambda d: d.lower, reverse=True)
        return ranked[:k], diagnostics

    def _max_unvisited(
        self,
        doc_id: str,
        query_idx: int,
        query_tokens: Sequence[Sequence[float]],
        visited: List[set[int]],
        delta: float,
        total_events: int,
        probabilistic: bool,
    ) -> float:
        doc_entry = self.index.documents.get(doc_id)
        if doc_entry is None:
            return 0.0
        best = 0.0
        query = query_tokens[query_idx]
        for cluster_id in doc_entry.cluster_set:
            if cluster_id in visited[query_idx]:
                continue
            envelope = self.index.envelopes[cluster_id]
            if probabilistic and delta > 0:
                bound = envelope.probabilistic_upper_bound(query, delta, total_events)
            else:
                bound = envelope.upper_bound(query)
            if bound > best:
                best = bound
        return best

    def _next_cluster(self, heaps: List[List[Tuple[float, int]]], visited: List[set[int]]) -> Optional[Tuple[int, int]]:
        best_score = -1.0
        best_choice: Optional[Tuple[int, int]] = None
        for q_idx, heap in enumerate(heaps):
            while heap and heap[0][1] in visited[q_idx]:
                heapq.heappop(heap)
            if not heap:
                continue
            score = -heap[0][0]
            if score > best_score:
                best_score = score
                best_choice = (q_idx, heapq.heappop(heap)[1])
        return best_choice

    def _global_upper(
        self,
        heaps: List[List[Tuple[float, int]]],
        visited: List[set[int]],
        query_tokens: Sequence[Sequence[float]],
        delta: float,
        total_events: int,
        probabilistic: bool,
    ) -> float:
        totals: List[float] = []
        for q_idx, heap in enumerate(heaps):
            best = 0.0
            for neg_score, cluster_id in heap:
                if cluster_id in visited[q_idx]:
                    continue
                if probabilistic and delta > 0:
                    bound = self.index.envelopes[cluster_id].probabilistic_upper_bound(query_tokens[q_idx], delta, total_events)
                else:
                    bound = -neg_score
                if bound > best:
                    best = bound
                    break
            totals.append(best)
        return mean(totals)

    def _should_stop(
        self,
        doc_bounds: Dict[str, DocumentBounds],
        heaps: List[List[Tuple[float, int]]],
        visited: List[set[int]],
        query_tokens: Sequence[Sequence[float]],
        delta: float,
        epsilon: float,
        total_events: int,
        k: int,
    ) -> bool:
        if len(doc_bounds) < k:
            return False
        sorted_docs = sorted(doc_bounds.values(), key=lambda d: d.lower, reverse=True)
        top_k = sorted_docs[:k]
        rest = sorted_docs[k:]
        min_lower = min(doc.lower for doc in top_k)
        if delta > 0:
            rest_prob = [doc.prob_upper for doc in rest]
            rest_prob.append(self._global_upper(heaps, visited, query_tokens, delta, total_events, probabilistic=True))
            if rest_prob and min_lower >= max(rest_prob) - epsilon:
                return True
        rest_det = [doc.upper for doc in rest]
        rest_det.append(self._global_upper(heaps, visited, query_tokens, delta, total_events, probabilistic=False))
        if rest_det and min_lower >= max(rest_det) - epsilon:
            return True
        return False
