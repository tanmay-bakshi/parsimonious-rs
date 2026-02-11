#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source .venv/bin/activate
maturin develop --release -m crates/parsimonious-py/Cargo.toml -q
python bench/python/compare_with_upstream.py "$@"
