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
bench/python/run_release_compare.sh --repeats 5
```

Sample output from this repository on CPython 3.14:

| Case | Impl | API | Parse (s) | Throughput (KB/s) | Speedup vs upstream |
|---|---:|---:|---:|---:|---:|
| arithmetic | upstream-python | parse | 0.021295 | 536 | 1.00x |
| arithmetic | parsimonious-rs | parse | 0.012912 | 885 | 1.65x |
| arithmetic | parsimonious-rs | parse_end | 0.006724 | 1,699 | 3.17x |
| json_like | upstream-python | parse | 0.016373 | 1,852 | 1.00x |
| json_like | parsimonious-rs | parse | 0.009412 | 3,222 | 1.74x |
| json_like | parsimonious-rs | parse_end | 0.005833 | 5,199 | 2.81x |
| log_lines | upstream-python | parse | 0.006745 | 17,374 | 1.00x |
| log_lines | parsimonious-rs | parse | 0.003913 | 29,948 | 1.72x |
| log_lines | parsimonious-rs | parse_end | 0.001895 | 61,843 | 3.56x |

Interpretation:

- Drop-in `parse()` behavior is now faster than upstream on these benchmark
  cases when compiled in release mode.
- The compiled end-offset API (`parse_end`) provides larger speedups when parse
  tree construction is not needed.
