# Multi-Vector-2026

Prototype implementation of the XTRust late interaction retrieval engine described in the accompanying research summary. The repository contains:

- `xtrust/`: core Python package with index construction, quantisation envelopes, probabilistic certificates, and the query engine.
- `scripts/run_demo.py`: demonstration script generating a synthetic corpus and running the engine end-to-end.
- `tests/`: automated tests covering index construction and query execution with certificates.

## Getting started

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the demonstration experiment:

```bash
python scripts/run_demo.py --docs 64 --tokens 8 --dimension 16 --query-tokens 4 --clusters 12 --bits 4
```

Execute the unit tests:

```bash
pytest
```
