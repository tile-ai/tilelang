from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover
    tabulate = None  # type: ignore


@dataclass(frozen=True)
class PerfResult:
    name: str
    latency: float


@dataclass(frozen=True)
class BenchCase:
    rel_path: str
    function: str
    display_name: str

    @property
    def id(self) -> str:
        return f"{self.rel_path}::{self.display_name}"


@dataclass(frozen=True)
class BenchFile:
    path: Path
    rel_path: str
    cases: tuple[BenchCase, ...]


@dataclass(frozen=True)
class ParsedResults:
    results: dict[str, float]
    cases: dict[str, list[str]]
    result_cases: dict[str, str]


_RESULTS: list[PerfResult] = []

_RESULTS_JSON_PREFIX = "__TILELANG_PERF_RESULTS_JSON__="


def _display_case_name(function_name: str) -> str:
    return function_name[len("regression_") :] if function_name.startswith("regression_") else function_name


def _parse_cases(bench_file: Path, rel_path: str) -> tuple[BenchCase, ...]:
    try:
        tree = ast.parse(bench_file.read_text())
    except (OSError, SyntaxError):
        return ()

    cases: list[BenchCase] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("regression_"):
            cases.append(
                BenchCase(
                    rel_path=rel_path,
                    function=node.name,
                    display_name=_display_case_name(node.name),
                )
            )
    return tuple(sorted(cases, key=lambda case: case.function))


def _parse_results(output: str, rel_path: str | None = None) -> ParsedResults:
    # Prefer a single JSON marker line if present.
    for line in reversed(output.splitlines()):
        if line.startswith(_RESULTS_JSON_PREFIX):
            payload = line[len(_RESULTS_JSON_PREFIX) :].strip()
            items = json.loads(payload)

            if isinstance(items, dict) and "results" in items:
                results = {str(k): float(v) for k, v in items.get("results", {}).items()}
                cases = {str(k): [str(v) for v in values] for k, values in items.get("cases", {}).items()}
                result_cases = {str(k): str(v) for k, v in items.get("result_cases", {}).items()}
                return ParsedResults(results=results, cases=cases, result_cases=result_cases)

            if isinstance(items, dict):
                return ParsedResults(
                    results={str(k): float(v) for k, v in items.items()},
                    cases={},
                    result_cases={},
                )

            data: dict[str, float] = {}
            cases: dict[str, list[str]] = {}
            result_cases: dict[str, str] = {}
            for item in items:
                name = str(item["name"]).strip()
                latency = float(item["latency"])
                data[name] = latency

                case = item.get("case")
                if case is not None and rel_path is not None:
                    case_id = f"{rel_path}::{str(case).strip()}"
                    cases.setdefault(case_id, []).append(name)
                    result_cases[name] = case_id
            return ParsedResults(results=data, cases=cases, result_cases=result_cases)

    # Backward-compatible text parsing (best-effort).
    data = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, val = line.partition(":")
        name = name.strip()
        val = val.strip()
        if not name:
            continue
        try:
            data[name] = float(val)
        except ValueError:
            # Ignore unrelated prints/logs.
            continue
    return ParsedResults(results=data, cases={}, result_cases={})


def _examples_root() -> Path:
    return Path(__file__).resolve().parents[2] / "examples"


def _discover_bench_files(examples_root: Path) -> list[Path]:
    patterns = ("regression_*.py",)
    files: list[Path] = []
    for pat in patterns:
        files.extend(examples_root.rglob(pat))
    # Avoid picking up things like __pycache__ etc.
    return sorted({p for p in files if p.is_file() and p.name != "__init__.py"})


def discover_benchmarks(examples_root: str | os.PathLike[str] | None = None) -> list[BenchFile]:
    root = (Path(examples_root) if examples_root is not None else _examples_root()).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Examples root not found: {root}")

    benchmarks: list[BenchFile] = []
    for path in _discover_bench_files(root):
        rel_path = path.relative_to(root).as_posix()
        benchmarks.append(BenchFile(path=path, rel_path=rel_path, cases=_parse_cases(path, rel_path)))
    return benchmarks


def _matches_filter(value: str, pattern: str) -> bool:
    value = value.lower()
    pattern = pattern.lower()
    return fnmatch.fnmatch(value, pattern) or pattern in value


def _file_matches(bench_file: BenchFile, pattern: str) -> bool:
    candidates = (
        bench_file.rel_path,
        str(Path(bench_file.rel_path).with_suffix("")).replace(os.sep, "/"),
        Path(bench_file.rel_path).name,
        Path(bench_file.rel_path).stem,
    )
    return any(_matches_filter(candidate, pattern) for candidate in candidates)


def _case_matches(case: BenchCase, pattern: str) -> bool:
    candidates = (case.id, case.function, case.display_name)
    return any(_matches_filter(candidate, pattern) for candidate in candidates)


def select_benchmarks(
    benchmarks: Sequence[BenchFile], filters: Sequence[str] | None = None
) -> list[tuple[BenchFile, tuple[BenchCase, ...] | None]]:
    """Select benchmark files and optional per-file cases.

    A None case tuple means the full file should run. A non-empty case tuple
    means only those regression functions should run in that file.
    """

    patterns = [pattern for pattern in (filters or []) if pattern]
    if not patterns:
        return [(bench_file, None) for bench_file in benchmarks]

    selected: list[tuple[BenchFile, tuple[BenchCase, ...] | None]] = []
    for bench_file in benchmarks:
        if any(_file_matches(bench_file, pattern) for pattern in patterns):
            selected.append((bench_file, None))
            continue

        matched_cases = tuple(case for case in bench_file.cases if any(_case_matches(case, pattern) for pattern in patterns))
        if matched_cases:
            selected.append((bench_file, matched_cases))

    return selected


def list_benchmarks(examples_root: str | os.PathLike[str] | None = None, filters: Sequence[str] | None = None) -> None:
    selected = select_benchmarks(discover_benchmarks(examples_root), filters)
    for bench_file, cases in selected:
        listed_cases = bench_file.cases if cases is None else cases
        if not listed_cases:
            print(bench_file.rel_path)
            continue
        for case in listed_cases:
            print(case.id)


def regression_all(
    examples_root: str | os.PathLike[str] | None = None,
    filters: Sequence[str] | None = None,
) -> dict[str, float]:
    """Run example benchmark drivers and print a consolidated table."""

    _RESULTS.clear()

    benchmarks = discover_benchmarks(examples_root)
    if not benchmarks:
        root = Path(examples_root) if examples_root is not None else _examples_root()
        raise RuntimeError(f"No drivers found under: {root}")

    selected = select_benchmarks(benchmarks, filters)
    if not selected:
        raise RuntimeError(f"No regression benchmarks matched filters: {', '.join(filters or [])}")

    merged: dict[str, float] = {}
    case_results: dict[str, list[str]] = {}
    result_cases: dict[str, str] = {}
    failures: list[str] = []

    total = len(selected)
    print(f"\n{'=' * 60}")
    print("  TileLang Performance Regression Suite")
    print(f"  Found {len(benchmarks)} test file(s), selected {total}")
    print(f"{'=' * 60}")
    for idx, (bench_file, cases) in enumerate(selected, 1):
        case_names = [case.display_name for case in cases] if cases is not None else []
        print(f"\n{'-' * 60}")
        print(f"[{idx}/{total}] {bench_file.rel_path}")
        if case_names:
            print(f"  cases: {', '.join(case_names)}")
        print(f"{'-' * 60}")

        child_env = {
            **os.environ,
            "TL_PERF_REGRESSION_FORMAT": "json",
        }
        if cases is not None:
            child_env["TL_PERF_REGRESSION_ONLY"] = json.dumps(case_names)
        else:
            child_env.pop("TL_PERF_REGRESSION_ONLY", None)

        proc = subprocess.Popen(
            [sys.executable, str(bench_file.path)],
            cwd=str(bench_file.path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env,
        )

        stdout_lines: list[str] = []
        # Stream stdout in real-time.
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            # Do not print the JSON result line.
            if not line.startswith(_RESULTS_JSON_PREFIX):
                print(line, end="", flush=True)

        proc.wait()
        stdout_content = "".join(stdout_lines)
        stderr_content = proc.stderr.read() if proc.stderr else ""

        if proc.returncode != 0:
            failures.append(f"{bench_file.rel_path}\nSTDOUT:\n{stdout_content}\nSTDERR:\n{stderr_content}")
            print("  `- FAILED")
            continue

        parsed = _parse_results(stdout_content, rel_path=bench_file.rel_path)
        num_tests = len(parsed.results)
        for k, v in parsed.results.items():
            if k not in merged:
                merged[k] = v
                _RESULTS.append(PerfResult(name=k, latency=v))
        for case_id, result_names in parsed.cases.items():
            case_results.setdefault(case_id, [])
            for result_name in result_names:
                if result_name not in case_results[case_id]:
                    case_results[case_id].append(result_name)
        result_cases.update(parsed.result_cases)

        print(f"  `- Completed ({num_tests} tests)")

    # Print summary.
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    passed = total - len(failures)
    print(f"  Passed: {passed}/{total} files")
    if failures:
        print(f"  Failed: {len(failures)}/{total} files")
    print(f"  Total tests: {len(merged)}")
    print()

    if failures and not merged:
        raise RuntimeError("All benchmark drivers failed:\n\n" + "\n\n".join(failures))
    if failures:
        # Do not hard-fail if we have some results; surface the errors for debugging.
        print(f"{'-' * 60}")
        print("  Failed benchmarks (partial results):")
        print(f"{'-' * 60}")
        for msg in failures:
            print("  ---")
            for line in msg.splitlines():
                print(f"  {line}")
        print()

    fmt = os.environ.get("TL_PERF_REGRESSION_FORMAT", "text").strip().lower()
    if fmt == "rich-json":
        print(
            _RESULTS_JSON_PREFIX
            + json.dumps(
                {"results": merged, "cases": case_results, "result_cases": result_cases},
                separators=(",", ":"),
            )
        )
        return merged
    if fmt == "json":
        print(_RESULTS_JSON_PREFIX + json.dumps(merged, separators=(",", ":")))
        return merged

    print(f"{'-' * 60}")
    print("  Results")
    print(f"{'-' * 60}")
    rows = [[k, merged[k]] for k in sorted(merged.keys())]
    headers = ["Name", "Latency (ms)"]
    if tabulate is None:
        print(f"| {headers[0]} | {headers[1]} |")
        print("|---|---|")
        for name, latency in rows:
            print(f"| {name} | {latency} |")
    else:
        print(tabulate(rows, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))
    return merged


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TileLang example performance regression benchmarks.")
    parser.add_argument("patterns", nargs="*", help="Optional file/case filters, e.g. gemm flash_attention")
    parser.add_argument("-k", "--filter", action="append", default=[], help="Additional file/case filter")
    parser.add_argument("--examples-root", default=None, help="Examples directory to scan")
    parser.add_argument("--list", action="store_true", help="List selected benchmark cases without running")
    parser.add_argument(
        "--format",
        choices=("text", "json", "rich-json"),
        default=os.environ.get("TL_PERF_REGRESSION_FORMAT", "text").strip().lower(),
        help="Output format",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    filters = [*args.patterns, *args.filter]
    if args.list:
        list_benchmarks(args.examples_root, filters)
        return

    os.environ["TL_PERF_REGRESSION_FORMAT"] = args.format
    regression_all(args.examples_root, filters)


if __name__ == "__main__":
    main()
