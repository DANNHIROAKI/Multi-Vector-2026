# Multi-Vector-2026

A research-oriented prototype of the XTRust late-interaction retrieval engine. The project now includes
comprehensive tooling that mirrors the system design described in the accompanying research summary:

- `xtrust/` – the Python package containing the query engine, residual quantisation logic, audit helpers, and
  the SLO-aware planner.
- `scripts/run_demo.py` – generates a synthetic corpus, uses the planner to select execution parameters, runs a
  query, prints the diagnostics/certificate, and audits the resulting index.
- `scripts/final_audit.py` – performs an exhaustive verification pass that recomputes exact MaxSim scores, validates
  deterministic/probabilistic certificates, exercises streaming rebuilds, and checks planner decisions against the
  research methodology.
- `tests/` – regression tests covering index construction, streaming rebuilds, the audit surface, the planner, and
  certificate-aware query execution.

## Getting started

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the demonstration experiment (feel free to tweak the knobs):

```bash
python scripts/run_demo.py --docs 64 --tokens 8 --dimension 16 --query-tokens 4 --clusters 12 --bits 4 --delta 0.1 --epsilon 0.05
```

Run the comprehensive compliance audit:

```bash
python scripts/final_audit.py --docs 48 --tokens 8 --dimension 16 --query-tokens 4 --clusters 12 --bits 4 --epsilon 0.2 --delta 0.1
```

Execute the unit tests:

```bash
pytest
```

## Key modules

- `xtrust.engine` exposes `XTRustEngine.search`, returning full ranking information together with deterministic or
  probabilistic certificates and per-query diagnostics.
- `xtrust.index` provides builders and a `StreamingIndex` helper that supports append/remove workloads via shadow
  rebuilds with versioning.
- `xtrust.quantization` implements residual quantisation envelopes capable of delivering deterministic and high
  confidence bounds.
- `xtrust.audit` allows the generated index to be audited for envelope safety and signature integrity.
- `xtrust.planner` offers a lightweight SLO-aware heuristic that maps declarative quality/latency requests to engine
  parameters.

## Development notes

The implementation is intentionally pure Python to maximise readability. The focus is on correctness, invariants, and
clear structuring that mirrors the systems paper. Performance is therefore secondary, but the APIs are designed so that
lower-level accelerators could be swapped in without changing the higher-level logic.

