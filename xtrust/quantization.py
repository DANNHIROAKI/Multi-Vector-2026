"""Residual quantisation and safety envelope utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import log, sqrt
from typing import Dict, List, Sequence, Tuple

from .vector_utils import clamp, linspace, l2_norm, normalize_rows, subtract


@dataclass
class ClusterEnvelope:
    """Statistics required to derive deterministic and probabilistic bounds."""

    centroid: List[float]
    linf_radius: List[float]
    l2_radius: float
    residual_codebook: List[List[float]]
    residual_error: List[float]
    residual_samples: List[float]
    levels: int
    sample_cap: int = 2048
    _sorted_cache: List[float] = field(default_factory=list, init=False, repr=False)
    _cache_valid: bool = field(default=False, init=False, repr=False)

    @classmethod
    def empty(cls, centroid: List[float], dimension: int, levels: int) -> "ClusterEnvelope":
        return cls(
            centroid=centroid,
            linf_radius=[0.0] * dimension,
            l2_radius=0.0,
            residual_codebook=[[0.0] * levels for _ in range(dimension)],
            residual_error=[0.0] * dimension,
            residual_samples=[0.0],
            levels=levels,
        )

    @classmethod
    def from_residuals(cls, centroid: List[float], residuals: List[List[float]], levels: int) -> "ClusterEnvelope":
        dimension = len(centroid)
        if not residuals:
            return cls.empty(centroid, dimension, levels)
        linf_radius = [max(abs(residual[d]) for residual in residuals) for d in range(dimension)]
        l2_radius = max(l2_norm(residual) for residual in residuals)
        min_vals = [min(residual[d] for residual in residuals) for d in range(dimension)]
        max_vals = [max(residual[d] for residual in residuals) for d in range(dimension)]
        codebook = [[0.0] * levels for _ in range(dimension)]
        for d in range(dimension):
            if max_vals[d] == min_vals[d]:
                codebook[d] = [min_vals[d]] * levels
            else:
                codebook[d] = linspace(min_vals[d], max_vals[d], levels)
        errors = [0.0] * dimension
        samples = []
        for residual in residuals:
            encoded = []
            for d in range(dimension):
                cb = codebook[d]
                idx = min(range(levels), key=lambda i: abs(cb[i] - residual[d]))
                approx = cb[idx]
                errors[d] = max(errors[d], abs(residual[d] - approx))
                encoded.append(idx)
                samples.append(abs(residual[d]))
        envelope = cls(
            centroid=centroid,
            linf_radius=linf_radius,
            l2_radius=l2_radius,
            residual_codebook=codebook,
            residual_error=errors,
            residual_samples=samples if samples else [0.0],
            levels=levels,
        )
        envelope._invalidate_cache()
        return envelope

    def _invalidate_cache(self) -> None:
        self._cache_valid = False

    def encode(self, residual: Sequence[float], *, update_stats: bool = True) -> List[int]:
        codes: List[int] = []
        for d, value in enumerate(residual):
            codebook = self.residual_codebook[d]
            idx = min(range(self.levels), key=lambda i: abs(codebook[i] - value))
            approx = codebook[idx]
            if update_stats:
                self.residual_error[d] = max(self.residual_error[d], abs(value - approx))
                self.linf_radius[d] = max(self.linf_radius[d], abs(value))
            codes.append(idx)
            if update_stats:
                self.residual_samples.append(abs(value))
                if len(self.residual_samples) > self.sample_cap:
                    self.residual_samples.pop(0)
                self._invalidate_cache()
        if update_stats:
            self.l2_radius = max(self.l2_radius, l2_norm(residual))
        return codes

    def decode(self, code: Sequence[int]) -> List[float]:
        return [self.centroid[d] + self.residual_codebook[d][idx] for d, idx in enumerate(code)]

    def upper_bound(self, query: Sequence[float]) -> float:
        linf = sum(q * c for q, c in zip(query, self.centroid)) + sum(abs(q) * r for q, r in zip(query, self.linf_radius))
        l2 = sum(q * c for q, c in zip(query, self.centroid)) + self.l2_radius
        return max(0.0, min(1.0, min(linf, l2)))

    def probabilistic_upper_bound(self, query: Sequence[float], delta: float, total_events: int) -> float:
        if delta <= 0 or total_events <= 0 or not self.residual_samples:
            return self.upper_bound(query)
        adjusted_delta = max(min(delta / total_events, 0.25), 1e-9)
        samples = self._sorted_samples()
        n = len(samples)
        epsilon = sqrt(log(2.0 / adjusted_delta) / (2.0 * max(n, 1)))
        quantile = clamp(1.0 - epsilon, 0.0, 1.0)
        index = int(clamp(quantile * (n - 1), 0, n - 1))
        residual = samples[index]
        bound = sum(q * c for q, c in zip(query, self.centroid)) + residual * sum(abs(q) for q in query)
        return max(0.0, min(1.0, bound))

    def _sorted_samples(self) -> List[float]:
        if not self._cache_valid:
            self._sorted_cache = sorted(self.residual_samples)
            self._cache_valid = True
        return self._sorted_cache


class ResidualQuantizer:
    """Lightweight residual quantiser used by the prototype implementation."""

    def __init__(self, num_clusters: int, residual_bits: int, max_iter: int = 25, seed: int | None = None):
        self.num_clusters = num_clusters
        self.residual_bits = residual_bits
        self.max_iter = max_iter
        self.seed = seed or 0

    def _kmeans(self, matrix: List[List[float]]) -> Tuple[List[List[float]], List[int]]:
        n = len(matrix)
        dim = len(matrix[0])
        rng = random.Random(self.seed)
        if n <= self.num_clusters:
            centroids = [matrix[i][:] for i in range(n)]
            while len(centroids) < self.num_clusters:
                centroids.append([0.0] * dim)
            assignments = [min(i, self.num_clusters - 1) for i in range(n)]
            return centroids, assignments
        indices = rng.sample(range(n), self.num_clusters)
        centroids = [matrix[i][:] for i in indices]
        assignments = [0] * n
        for _ in range(self.max_iter):
            changed = False
            for idx, vector in enumerate(matrix):
                best = None
                best_idx = 0
                for cid, centroid in enumerate(centroids):
                    dist = sum((v - c) ** 2 for v, c in zip(vector, centroid))
                    if best is None or dist < best:
                        best = dist
                        best_idx = cid
                if assignments[idx] != best_idx:
                    changed = True
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
            if not changed:
                centroids = new_centroids
                break
            centroids = new_centroids
        return centroids, assignments

    def fit(self, tokens: List[List[float]]) -> Tuple[List[List[float]], Dict[int, ClusterEnvelope], List[int]]:
        if not tokens:
            raise ValueError("Cannot fit quantiser without tokens")
        tokens = normalize_rows(tokens)
        centroids, assignments = self._kmeans(tokens)
        buckets: Dict[int, List[int]] = {cid: [] for cid in range(self.num_clusters)}
        for idx, cid in enumerate(assignments):
            buckets[cid].append(idx)
        dimension = len(tokens[0])
        levels = 2 ** self.residual_bits
        envelopes: Dict[int, ClusterEnvelope] = {}
        for cid in range(self.num_clusters):
            members = buckets[cid]
            centroid = centroids[cid][:]
            if not members:
                envelopes[cid] = ClusterEnvelope.empty(centroid, dimension, levels)
                continue
            residuals = [subtract(tokens[i], centroid) for i in members]
            envelopes[cid] = ClusterEnvelope.from_residuals(centroid, residuals, levels)
        return centroids, envelopes, assignments

