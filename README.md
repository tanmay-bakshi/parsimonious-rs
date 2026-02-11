# parsimonious-rs

A performance-focused Rust port of the Python PEG parser **Parsimonious**, with:

- A Rust crate: `parsimonious`
- Python bindings (PyO3) providing a `parsimonious` Python package

## Status

- Python parity test suite migrated from upstream and passing: `84 passed, 2 skipped`.
- Core parsing engine for Python bindings implemented in Rust with packrat caching.
- Core native Rust crate (`crates/parsimonious`) now provides a usable expression API.

## Repository Layout

- `crates/parsimonious`: native Rust expression API and parser.
- `crates/parsimonious-py`: PyO3 extension module.
- `python/parsimonious`: Python package surface and grammar compiler glue.
- `tests/python/parsimonious_tests`: migrated upstream behavior tests.
- `bench/python`: cross-implementation Python benchmark scripts.

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

Performance benchmark builds should use:

```bash
maturin develop --release -m crates/parsimonious-py/Cargo.toml
```

## Python API Notes

- Compatibility mode:
  - `Grammar.parse(...)`
  - `Grammar.match(...)`
  - `Expression.parse(...)`
  - `Expression.match(...)`
- High-throughput mode:
  - `Expression.parse_end(...)` returns an integer end offset.
  - `Expression.match_end(...)` returns `int | None`.

The high-throughput mode avoids Python parse-tree materialization and is the
current speed path in Python benchmarks.

## Rust API Example

```rust
use parsimonious::Expression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expr = Expression::sequence(vec![
        Expression::literal("hi"),
        Expression::literal(" "),
        Expression::one_of(vec![
            Expression::literal("there"),
            Expression::literal("world"),
        ]),
    ])
    .with_name("greeting");

    let node = expr.parse("hi world")?;
    assert_eq!(node.end, 8);
    Ok(())
}
```

## Python Benchmarks

Run (release build + comparison):

```bash
bench/python/run_release_compare.sh --repeats 2
```

Sample output from this repository on CPython 3.14:

| Case | Impl | API | Parse (s) | Throughput (KB/s) | Speedup vs upstream |
|---|---:|---:|---:|---:|---:|
| arithmetic | upstream-python | parse | 0.021786 | 524 | 1.00x |
| arithmetic | parsimonious-rs | parse | 0.006181 | 1,848 | 3.52x |
| arithmetic | parsimonious-rs | parse_end | 0.003223 | 3,544 | 6.76x |
| json_like | upstream-python | parse | 0.016736 | 1,812 | 1.00x |
| json_like | parsimonious-rs | parse | 0.004394 | 6,902 | 3.81x |
| json_like | parsimonious-rs | parse_end | 0.002516 | 12,053 | 6.65x |
| log_lines | upstream-python | parse | 0.006948 | 16,866 | 1.00x |
| log_lines | parsimonious-rs | parse | 0.002841 | 41,246 | 2.45x |
| log_lines | parsimonious-rs | parse_end | 0.001891 | 61,972 | 3.67x |
| tiny_literal | upstream-python | parse | 0.000010 | 3,206 | 1.00x |
| tiny_literal | parsimonious-rs | parse | 0.000003 | 8,956 | 2.79x |
| tiny_literal | parsimonious-rs | parse_end | 0.000002 | 18,010 | 5.62x |

Interpretation:

- Drop-in `parse()` behavior is faster than upstream across all benchmark
  regimes in this suite, including tiny-input per-call overhead.
- The compiled end-offset API (`parse_end`) provides larger speedups when parse
  tree construction is not needed.
