# parsimonious-rs

A performance-focused Rust port of the Python PEG parser **Parsimonious**, with:

- A Rust crate: `parsimonious`
- Python bindings (PyO3) providing a `parsimonious` Python package

## Development

Rust:

```bash
cargo test
cargo bench -p parsimonious
```

Python (from a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[tests]

maturin develop
pytest -q
```

## Goals

- Match Parsimonious behavior as closely as possible.
- Be substantially faster than the pure-Python implementation for real grammars.

