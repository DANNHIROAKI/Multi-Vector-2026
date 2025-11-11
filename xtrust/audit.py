"""Audit helpers that validate structural invariants of an :class:`XTRustIndex`."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from .index import XTRustIndex
from .vector_utils import l2_norm, normalize_rows, relu_dot, subtract


@dataclass
class AuditIssue:
    """Represents a single finding produced by :func:`audit_index`."""

    severity: str
    message: str
    context: Dict[str, object] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Aggregated result of auditing an index."""

    issues: List[AuditIssue] = field(default_factory=list)
    warnings: List[AuditIssue] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return not self.issues

    def add_issue(self, severity: str, message: str, **context: object) -> None:
        issue = AuditIssue(severity=severity, message=message, context=context)
        if severity == "error":
            self.issues.append(issue)
        else:
            self.warnings.append(issue)


def audit_index(index: XTRustIndex, *, sample_queries: int = 16, seed: int = 0) -> AuditReport:
    """Run a lightweight audit over ``index`` and return a report."""

    report = AuditReport()
    if not index.documents:
        report.add_issue("error", "index contains no documents")
        return report

    doc_cluster_usage: Dict[str, set[int]] = {doc_id: set() for doc_id in index.documents}

    for doc_id, entry in index.documents.items():
        if entry.length != len(entry.tokens):
            report.add_issue("error", "document length mismatch", doc_id=doc_id)
        if entry.length != len(entry.token_map):
            report.add_issue("error", "token map mismatch", doc_id=doc_id)
        for cluster_id in entry.cluster_set:
            if not entry.cluster_signature.contains(cluster_id):
                report.add_issue(
                    "error",
                    "signature missing assigned cluster",
                    doc_id=doc_id,
                    cluster=cluster_id,
                )

    query_vectors = _sample_queries(index, sample_queries, seed)
    violations = 0
    total_checks = 0
    for cluster_id, tokens in index.cluster_tokens.items():
        envelope = index.envelopes[cluster_id]
        for token in tokens:
            entry = index.get_document(token.doc_id)
            if entry is None:
                report.add_issue(
                    "error",
                    "token references missing document",
                    doc_id=token.doc_id,
                    cluster=cluster_id,
                    token_index=token.token_index,
                )
                continue
            if token.token_index not in entry.token_map:
                report.add_issue(
                    "error",
                    "token index not present in document map",
                    doc_id=token.doc_id,
                    cluster=cluster_id,
                    token_index=token.token_index,
                )
                continue
            doc_cluster_usage[token.doc_id].add(cluster_id)
            original = entry.token_vector(token.token_index)
            residual = subtract(original, envelope.centroid)
            if any(abs(value) - radius > 1e-6 for value, radius in zip(residual, envelope.linf_radius)):
                report.add_issue(
                    "error",
                    "linf radius violated",
                    doc_id=token.doc_id,
                    cluster=cluster_id,
                    token_index=token.token_index,
                )
            if l2_norm(residual) - envelope.l2_radius > 1e-6:
                report.add_issue(
                    "error",
                    "l2 radius violated",
                    doc_id=token.doc_id,
                    cluster=cluster_id,
                    token_index=token.token_index,
                )
            approx_residual = [envelope.residual_codebook[d][code] for d, code in enumerate(token.code)]
            if any(
                abs(residual[d] - approx_residual[d]) - envelope.residual_error[d] > 1e-6
                for d in range(len(residual))
            ):
                report.add_issue(
                    "error",
                    "residual error bound violated",
                    doc_id=token.doc_id,
                    cluster=cluster_id,
                    token_index=token.token_index,
                )
        for query in query_vectors:
            actual = 0.0
            for token in tokens:
                entry = index.get_document(token.doc_id)
                if entry is not None and token.token_index in entry.token_map:
                    vector = entry.token_vector(token.token_index)
                else:
                    vector = index.decode_token(token)
                actual = max(actual, relu_dot(query, vector))
            upper = envelope.upper_bound(query)
            if actual - upper > 1e-6:
                violations += 1
            total_checks += 1
    if violations:
        report.add_issue("error", "deterministic upper bound violated", violations=violations, total_checks=total_checks)
    else:
        report.stats["upper_bound_checks"] = float(total_checks)

    for doc_id, clusters in doc_cluster_usage.items():
        entry = index.documents[doc_id]
        missing = entry.cluster_set - clusters
        extra = clusters - entry.cluster_set
        if missing:
            report.add_issue(
                "error",
                "cluster set missing assignments",
                doc_id=doc_id,
                missing=sorted(missing),
            )
        if extra:
            report.add_issue(
                "error",
                "cluster set contains unexpected assignments",
                doc_id=doc_id,
                extra=sorted(extra),
            )

    return report


def _sample_queries(index: XTRustIndex, sample_queries: int, seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    tokens: List[List[float]] = []
    for entry in index.documents.values():
        tokens.extend(entry.tokens)
    if not tokens:
        return [[1.0] + [0.0] * (index.dimension - 1)]
    rng.shuffle(tokens)
    queries = tokens[:sample_queries]
    return normalize_rows(queries)

