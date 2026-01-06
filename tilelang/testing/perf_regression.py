from __future__ import annotations

import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable
from collections.abc import Sequence
import warnings


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


def regression(prefixes: Sequence[str] = ("regression_",), verbose: bool = True) -> None:
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

    sorted_functions = sorted(functions, key=lambda kv: kv[0])
    total = len(sorted_functions)

    for idx, (name, fn) in enumerate(sorted_functions, 1):
        if verbose:
            # Strip 'regression_' prefix for cleaner display
            display_name = name[len("regression_") :] if name.startswith("regression_") else name
            print(f"  ├─ [{idx}/{total}] {display_name}", end="", flush=True)
        start_time = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppress logging warnings during benchmark execution
            prev_level = logging.root.level
            logging.disable(logging.WARNING)
            try:
                fn()
            finally:
                logging.disable(logging.NOTSET)
                logging.root.setLevel(prev_level)
        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f" ({elapsed:.2f}s)", flush=True)

    _emit_results()
