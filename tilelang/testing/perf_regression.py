from __future__ import annotations

import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable
from collections.abc import Sequence
import warnings
import fnmatch


@dataclass(frozen=True)
class PerfResult:
    name: str
    latency: float
    case: str | None = None


_RESULTS: list[PerfResult] = []
_CURRENT_CASE: str | None = None

_MAX_RETRY_NUM = 5

_RESULTS_JSON_PREFIX = "__TILELANG_PERF_RESULTS_JSON__="


def _results_to_jsonable() -> list[dict[str, float | str]]:
    items: list[dict[str, float | str]] = []
    for r in _RESULTS:
        item: dict[str, float | str] = {"name": r.name, "latency": r.latency}
        if r.case is not None:
            item["case"] = r.case
        items.append(item)
    return items


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


def _parse_only_filters() -> list[str]:
    value = os.environ.get("TL_PERF_REGRESSION_ONLY", "").strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in value.split(",")]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _matches_filter(value: str, pattern: str) -> bool:
    value = value.lower()
    pattern = pattern.lower()
    return fnmatch.fnmatch(value, pattern) or pattern in value


def _should_run_case(name: str, display_name: str, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    candidates = (name, display_name)
    return any(_matches_filter(candidate, pattern) for pattern in filters for candidate in candidates)


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
    _RESULTS.append(PerfResult(name=result_name, latency=latency, case=_CURRENT_CASE))


def regression(prefixes: Sequence[str] = ("regression_",), verbose: bool = True) -> None:
    """Run entrypoints in the caller module and print a markdown table.

    This is invoked by many example scripts.
    """

    caller_globals = inspect.currentframe().f_back.f_globals  # type: ignore[union-attr]

    global _CURRENT_CASE

    _reset_results()
    functions: list[tuple[str, Callable[[], Any]]] = []
    for k, v in list(caller_globals.items()):
        if not callable(v):
            continue
        if any(k.startswith(p) for p in prefixes):
            functions.append((k, v))

    sorted_functions = sorted(functions, key=lambda kv: kv[0])
    only_filters = _parse_only_filters()
    sorted_functions = [
        (name, fn)
        for name, fn in sorted_functions
        if _should_run_case(name, name[len("regression_") :] if name.startswith("regression_") else name, only_filters)
    ]
    total = len(sorted_functions)

    for idx, (name, fn) in enumerate(sorted_functions, 1):
        _CURRENT_CASE = name[len("regression_") :] if name.startswith("regression_") else name
        if verbose:
            # Strip 'regression_' prefix for cleaner display
            print(f"  ├─ [{idx}/{total}] {_CURRENT_CASE}", end="", flush=True)
        start_time = time.perf_counter()
        try:
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
        finally:
            _CURRENT_CASE = None

    _emit_results()
