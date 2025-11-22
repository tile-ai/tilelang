import os
import re
import sys
import inspect
import time
import traceback
import contextlib
import warnings
from tabulate import tabulate
import matplotlib.pyplot as plt
import importlib.util

__all__ = ["main", "process_func"]
_RECORDS = []


@contextlib.contextmanager
def suppress_output():
    # Context manager that redirects stdout/stderr to os.devnull (supports fileno)
    devnull = open(os.devnull, "w")
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        devnull.close()


def process_func(func, *args, repeat=10, warmup=3, **kwargs):
    # Run a target function multiple times and measure average latency.
    try:
        with suppress_output():
            for _ in range(warmup):
                func(*args, **kwargs)
    except Exception:
        pass

    times = []
    fail_count = 0
    for _ in range(repeat):
        start = time.time()
        try:
            with suppress_output():
                func(*args, **kwargs)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        except Exception:
            fail_count += 1
            traceback.print_exc(file=sys.stderr)

    if times:
        avg_latency = sum(times) / len(times)
        if fail_count == 0:
            _RECORDS.append((f"{func.__module__}", avg_latency))
        else:
            warnings.warn(
                f"benchmark for {func.__module__} failed {fail_count} times in {repeat} repeats",
                RuntimeWarning,
                stacklevel=2,
            )
            _RECORDS.append((f"{func.__module__}", avg_latency))
    else:
        warnings.warn(
            f"benchmark for {func.__module__} failed in all repeats (no valid run)",
            RuntimeWarning,
            stacklevel=2,
        )


def analyze_records(records, out_dir):
    # Analyze the data and draw a chart
    records.sort(key=lambda x: x[1])
    headers = ["Functions", "Avg Latency (ms)"]
    print(
        tabulate(_RECORDS, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))

    names = [r[0] for r in records]
    lats = [r[1] for r in records]
    plt.figure(figsize=(max(len(names) * 2.2, 6), 6))
    plt.bar(names, lats)
    plt.xlabel("Latency (ms)")
    plt.title("Benchmark Results")
    out_path = os.path.join(out_dir, "bench_result.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved Bar chart to {out_path}")


def main():
    # Entry point — automatically run all bench_* functions in caller file.
    test_file = inspect.getsourcefile(sys._getframe(1))
    out_dir = os.path.dirname(test_file)
    module = {}
    with open(test_file) as f:
        exec(f.read(), module)

    for name, func in module.items():
        if name.startswith("bench_") and callable(func):
            func()

    analyze_records(_RECORDS, out_dir)


def bench_all():
    # Do benchmark for all bench_* functions in examples

    # Load a Python file as a real module (preserves sys.path, __file__, imports)
    def _load_module(full_path):
        module_name = os.path.splitext(os.path.basename(full_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_root = os.path.abspath(os.path.join(current_dir, "../../examples"))

    bench_funcs = []
    added_roots = set()

    for root, _, files in os.walk(examples_root):
        for file_name in files:
            if re.match(r"^bench_.*\.py$", file_name):
                full_path = os.path.join(root, file_name)
                if root not in added_roots:
                    sys.path.insert(0, root)
                    added_roots.add(root)
                mod = _load_module(full_path)
                for name in dir(mod):
                    if name.startswith("bench_"):
                        func = getattr(mod, name)
                        if callable(func):
                            bench_funcs.append(func)
    for func in bench_funcs:
        func()

    print(tabulate(_RECORDS, tablefmt="github", stralign="left", numalign="decimal"))
