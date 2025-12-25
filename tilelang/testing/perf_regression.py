from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from collections.abc import Sequence
import warnings

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover
    tabulate = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


@dataclass(frozen=True)
class PerfResult:
    name: str
    latency: float


_RESULTS: list[PerfResult] = []

_MAX_RETRY_NUM = 5

_RESULTS_JSON_PREFIX = "__TILELANG_PERF_RESULTS_JSON__="


def _results_to_jsonable() -> list[dict[str, float | str]]:
    return [{"name": r.name, "latency": r.latency} for r in _RESULTS]


def _emit_results() -> None:
    """Emit results for parent collectors.

    Default output remains the historical text format. Set
    `TL_PERF_REGRESSION_FORMAT=json` to emit a single JSON marker line which is
    robust against extra prints from benchmark code.
    """
    fmt = os.environ.get("TL_PERF_REGRESSION_FORMAT", "text").strip().lower()
    if fmt == "json":
        print(_RESULTS_JSON_PREFIX + json.dumps(_results_to_jsonable(), separators=(",", ":")))
        return
    # Fallback (human-readable): one result per line.
    for r in _RESULTS:
        print(f"{r.name}: {r.latency}")


def _reset_results() -> None:
    _RESULTS.clear()


def process_func(func: Callable[..., float], name: str | None = None, /, **kwargs: Any) -> None:
    """Execute a single perf function and record its latency.

    `func` is expected to return a positive latency scalar (seconds or ms; we
    treat it as an opaque number, only ratios matter for regression).
    """
    result_name = getattr(func, "__module__", "<unknown>") if name is None else name
    if result_name.startswith("regression_"):
        result_name = result_name[len("regression_") :]
    latency = float(func(**kwargs))
    _iter = 0
    while latency <= 0.0 and _iter < _MAX_RETRY_NUM:
        latency = float(func(**kwargs))
        _iter += 1
    if latency <= 0.0:
        warnings.warn(f"{result_name} has latency {latency} <= 0. Please verify the profiling results.", RuntimeWarning, 1)
        return
    _RESULTS.append(PerfResult(name=result_name, latency=latency))


def regression(prefixes: Sequence[str] = ("regression_",)) -> None:
    """Run entrypoints in the caller module and print a markdown table.

    This is invoked by many example scripts.
    """

    caller_globals = inspect.currentframe().f_back.f_globals  # type: ignore[union-attr]

    _reset_results()
    functions: list[tuple[str, Callable[[], Any]]] = []
    for k, v in list(caller_globals.items()):
        if not callable(v):
            continue
        if any(k.startswith(p) for p in prefixes):
            functions.append((k, v))

    for _, fn in sorted(functions, key=lambda kv: kv[0]):
        fn()

    _emit_results()


def _parse_table(output: str) -> dict[str, float]:
    # Prefer a single JSON marker line if present.
    for line in reversed(output.splitlines()):
        if line.startswith(_RESULTS_JSON_PREFIX):
            payload = line[len(_RESULTS_JSON_PREFIX) :].strip()
            items = json.loads(payload)
            data: dict[str, float] = {}
            for item in items:
                name = str(item["name"]).strip()
                latency = float(item["latency"])
                data[name] = latency
            return data

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
    return data


def _examples_root() -> Path:
    # repo_root/tilelang/testing/perf_regression.py -> repo_root
    return Path(__file__).resolve().parents[2] / "examples"


def _discover_bench_files(examples_root: Path) -> list[Path]:
    patterns = ("regression_*.py",)
    files: list[Path] = []
    for pat in patterns:
        files.extend(examples_root.rglob(pat))
    # Avoid picking up things like __pycache__ etc.
    return sorted({p for p in files if p.is_file() and p.name != "__init__.py"})


def regression_all(examples_root: str | os.PathLike[str] | None = None) -> None:
    """Run all example benchmark drivers and print a consolidated table.

    Intended usage (CI): `python -c "import tilelang.testing.perf_regression as pr; pr.regression_all()"`
    """

    root = Path(examples_root) if examples_root is not None else _examples_root()
    if not root.exists():
        raise FileNotFoundError(f"Examples root not found: {root}")

    bench_files = _discover_bench_files(root)
    if not bench_files:
        raise RuntimeError(f"No drivers found under: {root}")

    _reset_results()
    merged: dict[str, float] = {}
    failures: list[str] = []

    for bench_file in tqdm(bench_files, desc="Running regression tests ..."):
        proc = subprocess.run(
            [sys.executable, str(bench_file)],
            cwd=str(bench_file.parent),
            capture_output=True,
            text=True,
            env={
                **os.environ,
                # Keep child processes from picking up user-site or random paths.
                "PYTHONNOUSERSITE": "1",
                # Ask child to emit a single JSON marker line for robust parsing.
                "TL_PERF_REGRESSION_FORMAT": "json",
            },
        )
        if proc.returncode != 0:
            failures.append(f"{bench_file.relative_to(root)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
            continue

        parsed = _parse_table(proc.stdout)
        for k, v in parsed.items():
            # First writer wins to keep stable behavior if duplicates happen.
            if k not in merged:
                merged[k] = v
                _RESULTS.append(PerfResult(name=k, latency=v))

    if failures and not merged:
        raise RuntimeError("All benchmark drivers failed:\n\n" + "\n\n".join(failures))
    if failures:
        # Don't hard-fail if we have some results; surface the errors for debugging.
        print("# Some benchmark drivers failed (partial results)")
        for msg in failures:
            print("# ---")
            for line in msg.splitlines():
                print(f"# {line}")

    fmt = os.environ.get("TL_PERF_REGRESSION_FORMAT", "text").strip().lower()
    if fmt == "json":
        print(_RESULTS_JSON_PREFIX + json.dumps(merged, separators=(",", ":")))
        return

    rows = [[k, merged[k]] for k in sorted(merged.keys())]
    headers = ["File", "Latency"]
    if tabulate is None:
        print(f"| {headers[0]} | {headers[1]} |")
        print("|---|---|")
        for name, latency in rows:
            print(f"| {name} | {latency} |")
    else:
        print(tabulate(rows, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))
