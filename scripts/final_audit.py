#!/usr/bin/env python
"""Comprehensive audit runner for the XTRust prototype.

This script rebuilds synthetic corpora, validates deterministic and probabilistic
certificates against exact MaxSim scores, exercises the streaming rebuild
machinery, and ensures the SLO planner recommendations are consistent with the
research methodology requirements.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xtrust.audit import audit_index
from xtrust.engine import SearchResult, XTRustEngine
from xtrust.index import XTRustIndex
from xtrust.planner import PlanOptimizer, PlanRequest
from xtrust.vector_utils import normalize_rows, relu_dot


def generate_corpus(
    *,
    num_docs: int,
    tokens_per_doc: int,
    dimension: int,
    seed: int,
) -> Dict[str, List[List[float]]]:
    rng = random.Random(seed)
    docs: Dict[str, List[List[float]]] = {}
    for doc_idx in range(num_docs):
        tokens = [
            [rng.gauss(0, 1) for _ in range(dimension)]
            for _ in range(tokens_per_doc)
        ]
        docs[f"doc-{doc_idx}"] = normalize_rows(tokens)
    return docs


def maxsim_score(query: List[List[float]], tokens: List[List[float]]) -> float:
    totals: List[float] = []
    for q in query:
        best = 0.0
        for token in tokens:
            best = max(best, relu_dot(q, token))
        totals.append(best)
    return sum(totals) / max(len(query), 1)


def decoded_document_tokens(index: XTRustIndex, doc_id: str) -> List[List[float]]:
    tokens: List[Tuple[int, List[float]]] = []
    for cluster in index.cluster_tokens.values():
        for token in cluster:
            if token.doc_id == doc_id:
                tokens.append((token.token_index, index.decode_token(token)))
    tokens.sort(key=lambda item: item[0])
    return [vector for _, vector in tokens]


def exact_ranking(index: XTRustIndex, query: List[List[float]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for doc_id, entry in index.documents.items():
        if not entry.active:
            continue
        tokens = decoded_document_tokens(index, doc_id)
        scores[doc_id] = maxsim_score(query, tokens)
    return scores


def assert_bounds_consistent(result: SearchResult, truth: Dict[str, float], tolerance: float) -> List[str]:
    failures: List[str] = []
    for doc in result.documents:
        actual = truth.get(doc.doc_id)
        if actual is None:
            failures.append(f"Missing truth for {doc.doc_id}")
            continue
        if actual > doc.upper + tolerance:
            failures.append(
                f"upper bound too low for {doc.doc_id}: actual={actual:.6f} upper={doc.upper:.6f}"
            )
        if actual < doc.lower - tolerance:
            failures.append(
                f"lower bound too high for {doc.doc_id}: actual={actual:.6f} lower={doc.lower:.6f}"
            )
        if doc.prob_upper + tolerance < doc.upper:
            failures.append(
                f"probabilistic bound tighter than deterministic for {doc.doc_id}: "
                f"upper={doc.upper:.6f} prob_upper={doc.prob_upper:.6f}"
            )
    ranked_truth = sorted(truth.items(), key=lambda item: item[1], reverse=True)
    returned = [doc.doc_id for doc in result.documents]
    top_truth = [doc_id for doc_id, _ in ranked_truth[: len(result.documents)]]
    if set(returned) != set(top_truth):
        failures.append(
            "returned documents do not match exact top-k: "
            f"returned={returned} truth_top={top_truth}"
        )
    return failures


def run_streaming_audit(engine: XTRustEngine, seed: int) -> List[str]:
    rng = random.Random(seed)
    streaming = engine.with_streaming()
    append_tokens = [
        [rng.gauss(0, 1) for _ in range(engine.index.dimension)]
        for _ in range(6)
    ]
    streaming.append_document("doc-new", append_tokens)
    to_remove = next(iter(engine.index.documents))
    streaming.remove_document(to_remove)
    streaming.stage_shadow()
    streaming.promote_shadow()
    report = audit_index(streaming.active)
    failures: List[str] = []
    if not report.succeeded:
        failures.append(
            "streaming rebuild failed audit: "
            + "; ".join(issue.message for issue in report.issues)
        )
    if report.warnings:
        failures.append(
            "streaming rebuild produced warnings: "
            + "; ".join(issue.message for issue in report.warnings)
        )
    if streaming.active.version <= engine.index.version:
        failures.append("streaming index version did not advance after promote")
    return failures


def evaluate_planner(engine: XTRustEngine, query_tokens: int, epsilon: float, delta: float) -> List[str]:
    planner = PlanOptimizer(engine.index)
    request = PlanRequest(
        query_tokens=query_tokens,
        target_recall=0.9,
        latency_budget_ms=30.0,
        epsilon=epsilon,
        delta=delta,
    )
    plan = planner.plan(request)
    failures: List[str] = []
    if plan.nprobe <= 0 or plan.max_probes < plan.nprobe:
        failures.append("planner produced invalid probe counts")
    if plan.certificate_mode == "deterministic" and delta > 0:
        failures.append("planner ignored probabilistic request")
    if plan.certificate_mode == "probabilistic" and delta == 0:
        failures.append("planner incorrectly escalated to probabilistic mode")
    if request.latency_budget_ms is not None and not plan.meets_latency:
        failures.append("planner failed to meet latency budget")
    if not (0.5 <= plan.expected_recall <= 1.0):
        failures.append("planner recall estimate out of range")
    return failures


def run_audit(args: argparse.Namespace) -> int:
    docs = generate_corpus(
        num_docs=args.docs,
        tokens_per_doc=args.tokens,
        dimension=args.dimension,
        seed=args.seed,
    )
    engine = XTRustEngine.from_documents(
        docs,
        num_clusters=args.clusters,
        residual_bits=args.bits,
        seed=args.seed,
    )
    audit = audit_index(engine.index, sample_queries=args.audit_queries, seed=args.seed)
    failures: List[str] = []
    if not audit.succeeded:
        failures.append(
            "index audit failed: " + "; ".join(issue.message for issue in audit.issues)
        )
    if audit.warnings:
        failures.append(
            "index audit produced warnings: "
            + "; ".join(issue.message for issue in audit.warnings)
        )

    query = normalize_rows(generate_corpus(
        num_docs=1,
        tokens_per_doc=args.query_tokens,
        dimension=args.dimension,
        seed=args.seed + 1,
    )["doc-0"])

    exact = exact_ranking(engine.index, query)

    sorted_truth = sorted(exact.values(), reverse=True)
    min_top = sorted_truth[min(args.top_k, len(sorted_truth)) - 1]
    max_rest = sorted_truth[args.top_k] if len(sorted_truth) > args.top_k else float("-inf")
    actual_gap = max_rest - min_top if max_rest != float("-inf") else float("-inf")

    det_result = engine.search(query, k=args.top_k, epsilon=args.epsilon)
    failures.extend(assert_bounds_consistent(det_result, exact, tolerance=1e-4))
    if not det_result.certificate.satisfied and actual_gap <= args.epsilon + 1e-6:
        failures.append("deterministic certificate failed to close despite admissible gap")

    prob_result = engine.search(
        query,
        k=args.top_k,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    failures.extend(assert_bounds_consistent(prob_result, exact, tolerance=1e-4))
    if prob_result.certificate.mode != "probabilistic":
        failures.append("probabilistic search did not report probabilistic mode")
    if prob_result.certificate.satisfied and prob_result.certificate.gap > args.epsilon + 1e-6:
        failures.append("probabilistic certificate gap exceeds epsilon despite satisfaction")

    failures.extend(run_streaming_audit(engine, args.seed + 2))
    failures.extend(evaluate_planner(engine, args.query_tokens, args.epsilon, args.delta))

    if failures:
        print("XTRust comprehensive audit FAILED", file=sys.stderr)
        for line in failures:
            print(f" - {line}", file=sys.stderr)
        return 1

    print("XTRust comprehensive audit PASSED")
    print(
        f"Documents={args.docs} tokens/doc={args.tokens} query_tokens={args.query_tokens} "
        f"clusters={args.clusters} residual_bits={args.bits}"
    )
    print(
        f"Deterministic gap={det_result.certificate.gap:.6f} Probabilistic gap="
        f"{prob_result.certificate.gap:.6f}"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a comprehensive XTRust audit")
    parser.add_argument("--docs", type=int, default=24, help="Number of documents in the synthetic corpus")
    parser.add_argument("--tokens", type=int, default=8, help="Tokens per document")
    parser.add_argument("--dimension", type=int, default=8, help="Vector dimensionality")
    parser.add_argument("--query-tokens", type=int, default=4, help="Query token count")
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters for quantisation")
    parser.add_argument("--bits", type=int, default=4, help="Residual quantisation bit-width")
    parser.add_argument("--top-k", type=int, default=5, help="Result cut-off for validation")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Epsilon budget for certificates")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta budget for probabilistic certificates")
    parser.add_argument("--audit-queries", type=int, default=32, help="Number of sample queries for envelope auditing")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_audit(args)


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    sys.exit(main())
