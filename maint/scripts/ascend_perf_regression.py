"""Ascend perf-regression driver with a supervised persistent worker.

Run via `maint/scripts/run_perf_regression_ascend.sh`, which overlays this driver and
the Ascend examples onto each tested ref, or invoke this file directly.

Why this file uses a worker
---------------------------
Many Ascend examples can fail *hard* on the baseline ref — a C++ `LOG(FATAL)`/`abort`,
an NPU/bisheng crash, an import-time failure, or a hang — none of which a Python
`try/except` can catch. A single such failure must not discard every result for that ref.

Starting one process per entry provides isolation, but repeatedly pays Python import,
Ascend context initialization, allocator setup, and in-process JIT cache costs. This
driver therefore starts one persistent worker and sends entries to it one at a time. The
worker reuses imports and the NPU context while it is healthy. Results are returned after
every entry; if the worker crashes or an entry times out, the parent preserves completed
results, restarts the worker for the next entry, and continues.

This logic is intentionally kept in an Ascend-only maintenance driver. The shared CUDA
regression modules remain unchanged; this file emits the JSON-marker contract consumed
by `run_perf_regression_ascend.sh`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples", "ascend")

# Must match the marker `run_perf_regression_ascend.sh` greps for.
_MARKER = "__TILELANG_PERF_RESULTS_JSON__="
_ENV_ENTRY = "TL_PERF_REGRESSION_ENTRY"  # child: run only this single entry
_ENV_WORKER = "TL_PERF_REGRESSION_WORKER"  # persistent child: read entries from stdin
_ENV_TIMEOUT = "TL_PERF_REGRESSION_TIMEOUT"  # parent: per-entry timeout (seconds)
_DEFAULT_ENTRY_TIMEOUT = 900.0
_WORKER_SHUTDOWN_TIMEOUT = 60.0
_WORKER_TERMINATE_TIMEOUT = 10.0

# (entry name, example path relative to this file, run_regression_perf kwargs).
# One row == one worker command and one independently reported result. The path may live
# in a subfolder (e.g. flash_attention/); the worker puts its directory on sys.path so
# intra-folder imports (``from core import …``) resolve without dotted-module gymnastics.
_ENTRIES: list[tuple[str, str, dict]] = [
    ("ascend_gemm_bf16", "example_gemm.py", {"dtype": "bfloat16"}),
    ("ascend_gemm_fp32", "example_gemm.py", {"dtype": "float32"}),
    ("ascend_gemm_fp32_hf32", "example_gemm.py", {"dtype": "float32", "hf32": "nearest_even"}),
    ("ascend_gemm_splitk", "example_gemm_splitk.py", {}),
    (
        "ascend_gemm_splitk_deterministic",
        "example_gemm_splitk.py",
        {"deterministic": True},
    ),
    ("ascend_gemm_l0", "example_gemm_l0.py", {}),
    ("ascend_gemm_mix_manual", "example_gemm_mix_manual.py", {}),
    ("ascend_gemm_mixedkernel", "example_gemm_mixedkernel.py", {}),
    ("ascend_gemm_bypass_l2", "example_gemm_bypass_l2.py", {}),
    ("ascend_gemm_various_shapes", "example_gemm_various_shapes.py", {}),
    ("ascend_blockscaled_gemm", "example_blockscaled_gemm.py", {}),
    ("ascend_mha", "flash_attention/example_mha.py", {}),
    ("ascend_gqa", "flash_attention/example_gqa.py", {}),
    ("ascend_gqa_bwd", "flash_attention/example_gqa_bwd.py", {}),
    ("ascend_rmsnorm", "example_rmsnorm.py", {}),
    ("ascend_buffer_version_annotation", "example_buffer_version_annotation.py", {}),
    ("ascend_atomic_add", "example_atomic.py", {}),
    ("ascend_simdvf_topk_gate", "example_simdvf_topk_gate.py", {}),
    ("ascend_simdvf_vecadd", "example_simdvf_vecadd.py", {}),
    ("ascend_simdvf_vecadd_lower", "example_simdvf_vecadd_lower.py", {}),
    ("ascend_simtvf_vecadd", "example_simtvf_vecadd.py", {}),
    ("ascend_simtvf_vecadd_mutex", "example_simtvf_vecadd_mutex.py", {}),
    ("ascend_simdvf_per_token_cast_to_fp8", "example_simdvf_per_token_cast_to_fp8.py", {}),
    ("ascend_simtvf_per_token_cast_to_fp8", "example_simtvf_per_token_cast_to_fp8.py", {}),
    ("ascend_crosslevel_multibuffer", "example_crosslevel_multibuffer.py", {}),
    ("ascend_compress", "example_compress.py", {}),
]

_EXAMPLE_MODULES: dict[str, object] = {}


def _load_example(rel_path: str):
    """Import an example by file path, with its own directory on sys.path[0].

    Loading by path (rather than dotted module name) lets a subfolder example
    (e.g. ``flash_attention/example_mha.py``) use plain ``from core import …``
    against a sibling ``core.py`` — the same way it resolves when run directly.
    """
    cached = _EXAMPLE_MODULES.get(rel_path)
    if cached is not None:
        return cached

    file_path = os.path.join(_EXAMPLES_DIR, rel_path)
    example_dir = os.path.dirname(file_path)
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)
    module_path = os.path.splitext(rel_path)[0].replace(os.sep, "_").replace("/", "_")
    mod_name = "_regression_example_" + module_path
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EXAMPLE_MODULES[rel_path] = module
    return module


def _run_child(entry_name: str) -> int:
    """Import + run exactly one entry, emit its JSON marker, and return a status.

    This supports both the persistent worker and the backwards-compatible direct child
    mode selected with ``TL_PERF_REGRESSION_ENTRY``.
    """
    spec = next((e for e in _ENTRIES if e[0] == entry_name), None)
    if spec is None:
        print(f"  ⚠️  unknown perf regression entry: {entry_name}", flush=True)
        print(_MARKER + json.dumps([]), flush=True)
        return 1

    _, rel_path, kwargs = spec
    results: list[dict] = []
    try:
        module = _load_example(rel_path)
        # process_func records into the shared _RESULTS and retries non-positive latency.
        import tilelang.testing as tilelang_testing
        from tilelang.testing import perf_regression as pr

        pr._reset_results()
        tilelang_testing.process_func(module.run_regression_perf, entry_name, **kwargs)
        results = [{"name": r.name, "latency": r.latency} for r in pr._RESULTS]
    except Exception as e:  # noqa: BLE001 - catchable failures still skip cleanly
        print(f"  ⚠️  {entry_name} failed, skipping: {type(e).__name__}: {e}", flush=True)

    print(_MARKER + json.dumps(results, separators=(",", ":")), flush=True)
    return 0


@dataclass
class _WorkerProcess:
    process: subprocess.Popen[str]
    lines: queue.Queue[str | None]
    reader: threading.Thread


_WorkerResultStatus = Literal["result", "timeout", "crashed"]


def _start_worker() -> _WorkerProcess:
    """Start a worker and continuously drain its merged stdout/stderr."""
    worker_env = {
        **os.environ,
        _ENV_WORKER: "1",
        "PYTHONUNBUFFERED": "1",
        "TL_PERF_REGRESSION_FORMAT": "json",
    }
    worker_env.pop(_ENV_ENTRY, None)
    process = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)],
        env=worker_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    lines: queue.Queue[str | None] = queue.Queue()

    def _read_stdout() -> None:
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=_read_stdout, name="ascend-perf-worker-output", daemon=True)
    reader.start()
    return _WorkerProcess(process=process, lines=lines, reader=reader)


def _send_worker_entry(worker: _WorkerProcess, entry_name: str) -> bool:
    """Send one entry to a live worker, returning False if it already exited."""
    if worker.process.poll() is not None or worker.process.stdin is None:
        return False
    try:
        worker.process.stdin.write(json.dumps(entry_name) + "\n")
        worker.process.stdin.flush()
    except (BrokenPipeError, OSError, ValueError):
        return False
    return True


def _wait_for_worker_result(
    worker: _WorkerProcess,
    timeout: float,
) -> tuple[_WorkerResultStatus, list[dict], list[str]]:
    """Wait for one result marker while retaining a diagnostic output tail."""
    deadline = time.monotonic() + max(timeout, 0.0)
    tail: deque[str] = deque(maxlen=20)

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout", [], list(tail)
        try:
            line = worker.lines.get(timeout=remaining)
        except queue.Empty:
            return "timeout", [], list(tail)

        if line is None:
            return "crashed", [], list(tail)

        clean_line = line.rstrip("\r\n")
        tail.append(clean_line)
        if clean_line.startswith(_MARKER):
            return "result", _parse_marker(clean_line), list(tail)


def _stop_worker(worker: _WorkerProcess, *, graceful: bool) -> None:
    """Close an idle worker cleanly, or terminate a failed/timed-out worker."""
    process = worker.process
    if graceful and process.poll() is None and process.stdin is not None:
        try:
            process.stdin.close()
            process.wait(timeout=_WORKER_SHUTDOWN_TIMEOUT)
        except (BrokenPipeError, OSError):
            pass
        except subprocess.TimeoutExpired:
            graceful = False

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=_WORKER_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    worker.reader.join(timeout=1.0)
    if process.stdout is not None:
        process.stdout.close()


def _run_worker() -> int:
    """Read JSON-encoded entry names from stdin and execute them sequentially."""
    for command in sys.stdin:
        try:
            entry_name = json.loads(command)
        except json.JSONDecodeError:
            entry_name = None
        if not isinstance(entry_name, str) or not entry_name:
            print("  ⚠️  invalid persistent-worker command", flush=True)
            print(_MARKER + "[]", flush=True)
            continue
        _run_child(entry_name)
    return 0


def _run_parent() -> int:
    """Parent mode: supervise one persistent worker and merge surviving results."""
    try:
        timeout = float(os.environ.get(_ENV_TIMEOUT, _DEFAULT_ENTRY_TIMEOUT))
    except ValueError:
        timeout = _DEFAULT_ENTRY_TIMEOUT

    total = len(_ENTRIES)
    merged: list[dict] = []
    worker: _WorkerProcess | None = None
    print(f"\n{'=' * 60}")
    print("  Ascend Performance Regression Suite (persistent worker)")
    print(f"  {total} entries, {timeout:.0f}s per-entry timeout, restart on failure")
    print(f"{'=' * 60}")

    try:
        for idx, (entry_name, _path, _kwargs) in enumerate(_ENTRIES, 1):
            print(f"\n  ├─ [{idx}/{total}] {entry_name}", end="", flush=True)
            start = time.perf_counter()

            sent = False
            start_error: OSError | None = None
            for _attempt in range(2):
                if worker is None:
                    try:
                        worker = _start_worker()
                    except OSError as err:
                        start_error = err
                        break
                if _send_worker_entry(worker, entry_name):
                    sent = True
                    break
                _stop_worker(worker, graceful=False)
                worker = None

            if not sent:
                detail = f": {start_error}" if start_error is not None else ""
                print(f"\n  ⚠️  failed to start persistent worker{detail}; skipping {entry_name}", flush=True)
                continue

            remaining = timeout - (time.perf_counter() - start)
            status, items, output_tail = _wait_for_worker_result(worker, remaining)
            elapsed = time.perf_counter() - start

            if status == "timeout":
                tail = "\n".join(output_tail)
                suffix = f" Last output:\n{tail}" if tail else ""
                print(f"\n  ⚠️  {entry_name} timed out after {timeout:.0f}s, skipping.{suffix}", flush=True)
                _stop_worker(worker, graceful=False)
                worker = None
                continue

            if status == "crashed":
                returncode = worker.process.poll()
                if returncode is None:
                    try:
                        returncode = worker.process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        returncode = None
                tail = "\n".join(output_tail)
                suffix = f" Last output:\n{tail}" if tail else ""
                print(f"\n  ⚠️  {entry_name} crashed (exit {returncode}), skipping.{suffix}", flush=True)
                _stop_worker(worker, graceful=False)
                worker = None
                continue

            if not items:
                tail = "\n".join(output_tail)
                suffix = f" Last output:\n{tail}" if tail else ""
                print(f"\n  ⚠️  {entry_name} produced no results, skipping.{suffix}", flush=True)
                continue

            merged.extend(items)
            print(f" ({elapsed:.2f}s)", flush=True)
    finally:
        if worker is not None:
            _stop_worker(worker, graceful=True)

    print(f"\n{'=' * 60}")
    print(f"  Collected {len(merged)}/{total} entries")
    print(f"{'=' * 60}\n")
    # The single mapping marker consumed by run_perf_regression_ascend.sh.
    result_map = {str(item["name"]): float(item["latency"]) for item in merged}
    print(_MARKER + json.dumps(result_map, separators=(",", ":")), flush=True)
    return 0


def _parse_marker(output: str) -> list[dict]:
    for line in reversed(output.splitlines()):
        if line.startswith(_MARKER):
            try:
                return json.loads(line[len(_MARKER) :].strip())
            except json.JSONDecodeError:
                return []
    return []


if __name__ == "__main__":
    entry = os.environ.get(_ENV_ENTRY)
    if os.environ.get(_ENV_WORKER) == "1":
        sys.exit(_run_worker())
    sys.exit(_run_child(entry) if entry else _run_parent())
