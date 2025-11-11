"""Residual quantisation utilities implemented in pure Python."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .vector_utils import (
    abs_list,
    add,
    clamp,
    dot,
    ints,
    linspace,
    l2_norm,
    mean,
    normalize_rows,
    subtract,
)


@dataclass
class ClusterEnvelope:
    centroid: List[float]
    linf_radius: List[float]
    l2_radius: float
    residual_codebook: List[List[float]]
    residual_error: List[float]
    residual_samples: List[float]

    def upper_bound(self, query: Sequence[float]) -> float:
        linf = dot(query, self.centroid) + sum(abs(q) * r for q, r in zip(query, self.linf_radius))
        l2 = dot(query, self.centroid) + self.l2_radius
        return max(0.0, min(1.0, min(linf, l2)))

    def probabilistic_upper_bound(self, query: Sequence[float], delta: float, total_events: int) -> float:
        if delta <= 0 or total_events <= 0 or not self.residual_samples:
            return self.upper_bound(query)
        from math import log, sqrt

        adjusted_delta = max(min(delta / total_events, 0.25), 1e-9)
        n = len(self.residual_samples)
        epsilon = sqrt(log(2.0 / adjusted_delta) / (2.0 * max(n, 1)))
        quantile = min(1.0, 1.0 - epsilon)
        index = int(max(0, min(n - 1, round(quantile * (n - 1)))))
        residual = self.residual_samples[index]
        bound = dot(query, self.centroid) + sum(abs(q) * residual for q in query)
        return max(0.0, min(1.0, bound))


class ResidualQuantizer:
    def __init__(self, num_clusters: int, residual_bits: int, max_iter: int = 25, seed: int | None = None):
        self.num_clusters = num_clusters
        self.residual_bits = residual_bits
        self.max_iter = max_iter
        self.seed = seed or 0

    def _kmeans(self, matrix: List[List[float]]) -> Tuple[List[List[float]], List[int]]:
        import random

        n = len(matrix)
        dim = len(matrix[0])
        rng = random.Random(self.seed)
        if n <= self.num_clusters:
            centroids = [matrix[i][:] for i in range(n)]
            while len(centroids) < self.num_clusters:
                centroids.append([0.0] * dim)
            assignments = list(range(n))
            return centroids, assignments
        indices = rng.sample(range(n), self.num_clusters)
        centroids = [matrix[i][:] for i in indices]
        assignments = [0] * n
        for _ in range(self.max_iter):
            for idx, vector in enumerate(matrix):
                best = None
                best_idx = 0
                for cid, centroid in enumerate(centroids):
                    dist = sum((v - c) ** 2 for v, c in zip(vector, centroid))
                    if best is None or dist < best:
                        best = dist
                        best_idx = cid
                assignments[idx] = best_idx
            new_centroids = [[0.0] * dim for _ in range(self.num_clusters)]
            counts = [0] * self.num_clusters
            for vector, cid in zip(matrix, assignments):
                counts[cid] += 1
                new_centroids[cid] = [nc + v for nc, v in zip(new_centroids[cid], vector)]
            for cid in range(self.num_clusters):
                if counts[cid] == 0:
                    new_centroids[cid] = matrix[rng.randrange(n)][:]
                else:
                    new_centroids[cid] = [v / counts[cid] for v in new_centroids[cid]]
            if all(
                all(abs(a - b) < 1e-4 for a, b in zip(old, new))
                for old, new in zip(centroids, new_centroids)
            ):
                centroids = new_centroids
                break
            centroids = new_centroids
        return centroids, assignments

    def fit(
        self, tokens: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[int]], Dict[int, ClusterEnvelope], List[int]]:
        centroids, assignments = self._kmeans(tokens)
        centroids = normalize_rows(centroids)
        buckets: List[List[int]] = [[] for _ in range(self.num_clusters)]
        for idx, cid in enumerate(assignments):
            buckets[cid].append(idx)
        dim = len(tokens[0])
        levels = 2 ** self.residual_bits
        envelopes: Dict[int, ClusterEnvelope] = {}

        for cid, members in enumerate(buckets):
            if not members:
                centroid = centroids[cid][:]
                linf_radius = [0.0] * dim
                codebook = [[0.0] * levels for _ in range(dim)]
                envelopes[cid] = ClusterEnvelope(
                    centroid=centroid,
                    linf_radius=linf_radius,
                    l2_radius=0.0,
                    residual_codebook=codebook,
                    residual_error=[0.0] * dim,
                    residual_samples=[0.0],
                )
                continue
            cluster_vectors = [tokens[i] for i in members]
            centroid = centroids[cid][:]
            residuals = [subtract(vec, centroid) for vec in cluster_vectors]
            linf_radius = [max(abs(residual[d]) for residual in residuals) for d in range(dim)]
            l2_radius = max(l2_norm(residual) for residual in residuals)
            min_vals = [min(residual[d] for residual in residuals) for d in range(dim)]
            max_vals = [max(residual[d] for residual in residuals) for d in range(dim)]
            codebook = [[0.0] * levels for _ in range(dim)]
            for d in range(dim):
                if max_vals[d] == min_vals[d]:
                    codebook[d] = [min_vals[d]] * levels
                else:
                    codebook[d] = linspace(min_vals[d], max_vals[d], levels)
            approx = [[0.0] * dim for _ in residuals]
            for ridx, residual in enumerate(residuals):
                for d in range(dim):
                    candidates = codebook[d]
                    best_code = min(range(levels), key=lambda c: abs(candidates[c] - residual[d]))
                    approx[ridx][d] = candidates[best_code]
            errors = [max(abs(residuals[i][d] - approx[i][d]) for i in range(len(residuals))) for d in range(dim)]
            samples = sorted(abs(value) for residual in residuals for value in residual)
            envelopes[cid] = ClusterEnvelope(
                centroid=centroid,
                linf_radius=linf_radius,
                l2_radius=l2_radius,
                residual_codebook=codebook,
                residual_error=errors,
                residual_samples=samples,
            )
        return centroids, buckets, envelopes, assignments
