from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover
    tabulate = None  # type: ignore


@dataclass(frozen=True)
class PerfResult:
    name: str
    latency: float


_RESULTS: list[PerfResult] = []

_RESULTS_JSON_PREFIX = "__TILELANG_PERF_RESULTS_JSON__="


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

    merged: dict[str, float] = {}
    failures: list[str] = []

    total = len(bench_files)
    print(f"\n{'═' * 60}")
    print("  TileLang Performance Regression Suite")
    print(f"  Found {total} test file(s)")
    print(f"{'═' * 60}")
    for idx, bench_file in enumerate(bench_files, 1):
        rel_path = bench_file.relative_to(root)
        print(f"\n{'─' * 60}")
        print(f"[{idx}/{total}] 📂 {rel_path}")
        print(f"{'─' * 60}")

        proc = subprocess.Popen(
            [sys.executable, str(bench_file)],
            cwd=str(bench_file.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                **os.environ,
                "PYTHONNOUSERSITE": "1",
                "TL_PERF_REGRESSION_FORMAT": "json",
            },
        )

        stdout_lines: list[str] = []
        # Stream stdout in real-time
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            # Don't print the JSON result line
            if not line.startswith(_RESULTS_JSON_PREFIX):
                print(line, end="", flush=True)

        proc.wait()
        stdout_content = "".join(stdout_lines)
        stderr_content = proc.stderr.read() if proc.stderr else ""

        if proc.returncode != 0:
            failures.append(f"{rel_path}\nSTDOUT:\n{stdout_content}\nSTDERR:\n{stderr_content}")
            print("  └─ ❌ FAILED")
            continue

        parsed = _parse_table(stdout_content)
        num_tests = len(parsed)
        for k, v in parsed.items():
            if k not in merged:
                merged[k] = v
                _RESULTS.append(PerfResult(name=k, latency=v))

        print(f"  └─ ✅ Completed ({num_tests} tests)")

    # Print summary
    print(f"\n{'═' * 60}")
    print("  Summary")
    print(f"{'═' * 60}")
    passed = total - len(failures)
    print(f"  ✅ Passed: {passed}/{total} files")
    if failures:
        print(f"  ❌ Failed: {len(failures)}/{total} files")
    print(f"  📊 Total tests: {len(merged)}")
    print()

    if failures and not merged:
        raise RuntimeError("All benchmark drivers failed:\n\n" + "\n\n".join(failures))
    if failures:
        # Don't hard-fail if we have some results; surface the errors for debugging.
        print(f"{'─' * 60}")
        print("  Failed benchmarks (partial results):")
        print(f"{'─' * 60}")
        for msg in failures:
            print("  ---")
            for line in msg.splitlines():
                print(f"  {line}")
        print()

    fmt = os.environ.get("TL_PERF_REGRESSION_FORMAT", "text").strip().lower()
    if fmt == "json":
        print(_RESULTS_JSON_PREFIX + json.dumps(merged, separators=(",", ":")))
        return

    print(f"{'─' * 60}")
    print("  Results")
    print(f"{'─' * 60}")
    rows = [[k, merged[k]] for k in sorted(merged.keys())]
    headers = ["Name", "Latency (ms)"]
    if tabulate is None:
        print(f"| {headers[0]} | {headers[1]} |")
        print("|---|---|")
        for name, latency in rows:
            print(f"| {name} | {latency} |")
    else:
        print(tabulate(rows, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))


if __name__ == "__main__":
    regression_all()
