"""Benchmark parsimonious-rs against the upstream Python Parsimonious package.

This script runs each implementation in its own subprocess so that both can use
the `parsimonious` module name without import collisions.

For meaningful comparisons, build/install the Rust extension in release mode:
`maturin develop --release -m crates/parsimonious-py/Cargo.toml`.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from timeit import repeat
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Benchmark workload definition.

    :ivar name: Case name.
    :ivar grammar: Grammar definition.
    :ivar text: Text to parse.
    :ivar iterations: Number of parse calls per timing sample.
    """

    name: str
    grammar: str
    text: str
    iterations: int


def _json_case() -> BenchmarkCase:
    """Build a JSON-like parsing benchmark case.

    :returns: A benchmark case.
    """

    father = """{
        "id" : 1,
        "married" : true,
        "name" : "Larry Lopez",
        "sons" : null,
        "daughters" : [
          {
            "age" : 26,
            "name" : "Sandra"
            },
          {
            "age" : 25,
            "name" : "Margaret"
            },
          {
            "age" : 6,
            "name" : "Mary"
            }
          ]
        }"""
    more_fathers = ",".join([father] * 80)
    payload = '{"fathers" : [' + more_fathers + "]}"
    grammar = r"""
        value = space (string / number / object / array / true_false_null) space

        object = "{" members "}"
        members = (pair ("," pair)*)?
        pair = string ":" value
        array = "[" elements "]"
        elements = (value ("," value)*)?
        true_false_null = "true" / "false" / "null"

        string = space "\"" chars "\"" space
        chars = ~"[^\"]*"
        number = (int frac exp) / (int exp) / (int frac) / int
        int = "-"? ((digit1to9 digits) / digit)
        frac = "." digits
        exp = e digits
        digits = digit+
        e = "e+" / "e-" / "e" / "E+" / "E-" / "E"

        digit1to9 = ~"[1-9]"
        digit = ~"[0-9]"
        space = ~"\s*"
    """
    return BenchmarkCase(
        name="json_like",
        grammar=grammar,
        text=payload,
        iterations=24,
    )


def _log_case() -> BenchmarkCase:
    """Build a log file parsing benchmark case.

    :returns: A benchmark case.
    """

    grammar = r"""
        file = line+
        line = ts " " level " " module ":" " " message "\n"
        ts = ~"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
        level = "INFO" / "WARN" / "ERROR"
        module = ~"[a-z_][a-z_0-9]*"
        message = ~"[^\n]*"
    """
    line = "2025-09-24T12:34:56Z INFO parser_core: matched rule packet_header in 182us\n"
    text = line * 1600
    return BenchmarkCase(
        name="log_lines",
        grammar=grammar,
        text=text,
        iterations=60,
    )


def _arith_case() -> BenchmarkCase:
    """Build an arithmetic-expression parsing benchmark case.

    :returns: A benchmark case.
    """

    grammar = r"""
        expr = ws sum ws
        sum = product (ws ("+" / "-") ws product)*
        product = atom (ws ("*" / "/") ws atom)*
        atom = number / "(" ws sum ws ")"
        number = ~"[0-9]+"
        ws = ~"\s*"
    """
    term = "(17 + 4*8 - 3/9 + 24*(8+5) - (11-7)*(3+2))"
    text = " + ".join([term] * 260)
    return BenchmarkCase(
        name="arithmetic",
        grammar=grammar,
        text=text,
        iterations=110,
    )


def get_cases() -> dict[str, BenchmarkCase]:
    """Return benchmark cases keyed by case name.

    :returns: Mapping of benchmark cases.
    """

    cases = [_json_case(), _log_case(), _arith_case()]
    return {case.name: case for case in cases}


def _worker_import_paths(repo_root: Path, implementation: str) -> list[str]:
    """Build import paths for a worker process.

    :param repo_root: Path to the `parsimonious-rs` repository.
    :param implementation: `rust` or `python`.
    :returns: Python path entries for the worker.
    :raises RuntimeError: If required source directories do not exist.
    """

    if implementation == "rust":
        rust_source = repo_root / "python"
        if rust_source.is_dir() is False:
            raise RuntimeError(f"Missing Rust binding package directory: {rust_source}")
        return [str(rust_source)]

    if implementation == "python":
        upstream_repo = repo_root.parent / "parsimonious"
        if upstream_repo.is_dir() is False:
            raise RuntimeError(
                "Upstream Parsimonious source directory was not found at "
                f"{upstream_repo}"
            )
        return [str(upstream_repo)]

    raise RuntimeError(f"Unsupported implementation: {implementation}")


def run_worker(
    repo_root: Path,
    implementation: str,
    mode: str,
    case_name: str,
    repeats: int,
) -> dict[str, Any]:
    """Run one benchmark worker and return its parsed JSON payload.

    :param repo_root: Path to the `parsimonious-rs` repository.
    :param implementation: `rust` or `python`.
    :param mode: `parse` or `parse_end`.
    :param case_name: Case to execute.
    :param repeats: Number of repeat samples in the worker.
    :returns: Worker metrics.
    :raises RuntimeError: If the worker process fails.
    """

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--implementation",
        implementation,
        "--mode",
        mode,
        "--case",
        case_name,
        "--repeats",
        str(repeats),
    ]

    env = os.environ.copy()
    # Build PYTHONPATH deterministically. Do not inherit empty separators.
    worker_paths = _worker_import_paths(repo_root, implementation)
    inherited_path = env.get("PYTHONPATH", "")
    if len(inherited_path) > 0:
        path_value = os.pathsep.join(worker_paths + [inherited_path])
    else:
        path_value = os.pathsep.join(worker_paths)
    env["PYTHONPATH"] = path_value

    completed = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Benchmark worker failed for "
            f"{implementation}/{mode}/{case_name}: {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def _worker_mode(
    implementation: str,
    mode: str,
    case_name: str,
    repeats: int,
) -> None:
    """Execute one benchmark case inside an isolated worker process.

    :param implementation: `rust` or `python`.
    :param mode: `parse` or `parse_end`.
    :param case_name: Benchmark case name.
    :param repeats: Number of repeat samples.
    :raises RuntimeError: If benchmark setup/validation fails.
    """

    from parsimonious.grammar import Grammar

    cases = get_cases()
    if case_name not in cases:
        raise RuntimeError(f"Unknown benchmark case: {case_name}")
    if implementation == "python" and mode == "parse_end":
        raise RuntimeError("Upstream implementation does not expose parse_end.")
    case = cases[case_name]

    compile_seconds = min(repeat(lambda: Grammar(case.grammar), repeat=repeats, number=1))
    grammar = Grammar(case.grammar)

    if mode == "parse":
        parsed = grammar.parse(case.text)
        if parsed.end != len(case.text):
            raise RuntimeError("Benchmark input was not fully parsed.")
        parse_total_seconds = min(
            repeat(lambda: grammar.parse(case.text), repeat=repeats, number=case.iterations)
        )
        parse_seconds = parse_total_seconds / case.iterations
    elif mode == "parse_end":
        default_rule = grammar.default_rule
        if default_rule is None:
            raise RuntimeError("Grammar has no default rule for parse_end benchmark.")
        parsed_end = default_rule.parse_end(case.text)
        if parsed_end != len(case.text):
            raise RuntimeError("Benchmark input was not fully parsed.")
        parse_total_seconds = min(
            repeat(
                lambda: default_rule.parse_end(case.text),
                repeat=repeats,
                number=case.iterations,
            )
        )
        parse_seconds = parse_total_seconds / case.iterations
    else:
        raise RuntimeError(f"Unknown benchmark mode: {mode}")

    payload_size_kb = len(case.text.encode("utf-8")) / 1024.0
    kb_per_sec = payload_size_kb / parse_seconds

    payload = {
        "implementation": implementation,
        "mode": mode,
        "case": case.name,
        "compile_seconds": compile_seconds,
        "parse_seconds": parse_seconds,
        "payload_size_kb": payload_size_kb,
        "kb_per_sec": kb_per_sec,
    }
    print(json.dumps(payload))


def _format_seconds(seconds: float) -> str:
    """Format seconds for table output.

    :param seconds: Seconds value.
    :returns: Formatted string.
    """

    return f"{seconds:.6f}"


def _format_speed(speed: float) -> str:
    """Format throughput for table output.

    :param speed: Throughput in KB/s.
    :returns: Formatted string.
    """

    return f"{speed:,.0f}"


def run_orchestrator(repeats: int) -> None:
    """Run benchmark comparisons and print results.

    :param repeats: Number of repeat samples for each implementation/case.
    """

    repo_root = Path(__file__).resolve().parents[2]
    cases = get_cases()
    ordered_cases = sorted(cases.keys())

    print("# Python Benchmark Comparison")
    print("")
    print(f"Interpreter: {sys.executable}")
    print(f"Repeats per sample: {repeats}")
    print("Expected extension build: release (`maturin develop --release ...`)")
    print("")
    print(
        "| Case | Impl | API | Compile (s) | Parse (s) | Throughput (KB/s) | Speedup vs upstream |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")

    for case_name in ordered_cases:
        upstream = run_worker(repo_root, "python", "parse", case_name, repeats)
        rust_parse = run_worker(repo_root, "rust", "parse", case_name, repeats)
        rust_fast = run_worker(repo_root, "rust", "parse_end", case_name, repeats)

        speedup_parse = upstream["parse_seconds"] / rust_parse["parse_seconds"]
        speedup_fast = upstream["parse_seconds"] / rust_fast["parse_seconds"]
        print(
            "| {case} | upstream-python | parse | {compile} | {parse} | {speed} | 1.00x |".format(
                case=case_name,
                compile=_format_seconds(upstream["compile_seconds"]),
                parse=_format_seconds(upstream["parse_seconds"]),
                speed=_format_speed(upstream["kb_per_sec"]),
            )
        )
        print(
            "| {case} | parsimonious-rs | parse | {compile} | {parse} | {speed} | {speedup:.2f}x |".format(
                case=case_name,
                compile=_format_seconds(rust_parse["compile_seconds"]),
                parse=_format_seconds(rust_parse["parse_seconds"]),
                speed=_format_speed(rust_parse["kb_per_sec"]),
                speedup=speedup_parse,
            )
        )
        print(
            "| {case} | parsimonious-rs | parse_end | {compile} | {parse} | {speed} | {speedup:.2f}x |".format(
                case=case_name,
                compile=_format_seconds(rust_fast["compile_seconds"]),
                parse=_format_seconds(rust_fast["parse_seconds"]),
                speed=_format_speed(rust_fast["kb_per_sec"]),
                speedup=speedup_fast,
            )
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :returns: Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Benchmark parsimonious-rs against upstream Parsimonious."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeat samples (minimum value is 1).",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run in worker mode (internal use).",
    )
    parser.add_argument(
        "--implementation",
        choices=["rust", "python"],
        default="rust",
        help="Implementation to benchmark in worker mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["parse", "parse_end"],
        default="parse",
        help="Benchmark mode in worker mode.",
    )
    parser.add_argument(
        "--case",
        default="json_like",
        help="Benchmark case name to run in worker mode.",
    )
    return parser.parse_args()


def main() -> None:
    """Program entrypoint."""

    args = parse_args()
    if args.repeats < 1:
        raise RuntimeError("--repeats must be at least 1.")

    if args.worker:
        _worker_mode(args.implementation, args.mode, args.case, args.repeats)
        return
    run_orchestrator(args.repeats)


if __name__ == "__main__":
    main()
