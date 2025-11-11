import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xtrust import PlanOptimizer, PlanRequest, XTRustEngine, audit_index


def _generate_corpus(num_docs: int, tokens_per_doc: int, dimension: int, seed: int) -> Dict[str, List[List[float]]]:
    rng = random.Random(seed)
    corpus: Dict[str, List[List[float]]] = {}
    for doc_idx in range(num_docs):
        matrix = [[rng.gauss(0, 1) for _ in range(dimension)] for _ in range(tokens_per_doc)]
        corpus[f"doc-{doc_idx}"] = matrix
    return corpus


def _generate_query(tokens: int, dimension: int, seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(dimension)] for _ in range(tokens)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate the XTRust engine on synthetic data.")
    parser.add_argument("--docs", type=int, default=128, help="Number of synthetic documents to index")
    parser.add_argument("--tokens", type=int, default=8, help="Tokens per document")
    parser.add_argument("--dimension", type=int, default=16, help="Embedding dimensionality")
    parser.add_argument("--query-tokens", type=int, default=4, help="Number of tokens in the query")
    parser.add_argument("--clusters", type=int, default=16, help="Number of residual clusters")
    parser.add_argument("--bits", type=int, default=4, help="Residual quantisation bits")
    parser.add_argument("--delta", type=float, default=0.0, help="Target failure probability for certificates")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Tolerance for certificate gap")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    corpus = _generate_corpus(args.docs, args.tokens, args.dimension, args.seed)
    engine = XTRustEngine.from_documents(corpus, num_clusters=args.clusters, residual_bits=args.bits, seed=args.seed)

    planner = PlanOptimizer(engine.index)
    plan = planner.plan(
        PlanRequest(
            query_tokens=args.query_tokens,
            target_recall=0.9,
            latency_budget_ms=None,
            epsilon=args.epsilon,
            delta=args.delta,
        )
    )

    query = _generate_query(args.query_tokens, args.dimension, args.seed + 1)
    result = engine.search(
        query,
        k=5,
        delta=args.delta,
        epsilon=args.epsilon,
        max_probes=plan.max_probes,
    )

    audit = audit_index(engine.index)

    print("Planner suggestion:")
    print(json.dumps(plan.__dict__, indent=2))
    print("\nDiagnostics:")
    print(json.dumps(result.diagnostics, indent=2))
    print("\nCertificate:")
    print(json.dumps(result.certificate.__dict__, indent=2))
    print("\nAudit:")
    if audit.succeeded:
        print("  All invariants satisfied")
    else:
        for issue in audit.issues:
            print(f"  {issue.severity.upper()}: {issue.message} -> {issue.context}")

    print("\nResults:")
    for idx, doc in enumerate(result.documents):
        print(
            f"{idx+1:02d}. {doc.doc_id}\tLB={doc.lower:.4f}\tUB={doc.upper:.4f}\tUB^(δ)={doc.prob_upper:.4f}\tScore={doc.score:.4f}"
        )


if __name__ == "__main__":
    main()

