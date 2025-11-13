import os
import sys
import inspect
import time
import traceback
import contextlib
import warnings
import matplotlib.pyplot as plt

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

def analyze_records(records):
    # Analyze the data and draw a chart
    records.sort(key = lambda x: x[1])
    name_col_width = max(len(r[0]) for r in records)
    safe_width = name_col_width + 20
    print("=" * safe_width)
    print(f"{'Function':<{name_col_width}} | Avg Latency (ms)")
    print("-" * safe_width)
    for name, lat in records:
        print(f"{name:<{name_col_width}} | {lat:>10.4f}")
    print("=" * safe_width)

    names = [r[0] for r in records]
    lats = [r[1] for r in records]
    plt.figure(figsize=(max(len(names) * 2.2, 6), 6))
    plt.bar(names, lats)
    plt.xlabel("Latency (ms)")
    plt.title("Benchmark Results")

    test_file = inspect.getsourcefile(sys._getframe(2))
    out_dir = os.path.dirname(test_file)
    out_path = os.path.join(out_dir, "bench_result.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved Bar chart to {out_path}")


def main():
    # Entry point — automatically run all bench_* functions in caller file.
    test_file = inspect.getsourcefile(sys._getframe(1))
    module = {}
    with open(test_file) as f:
        exec(f.read(), module)

    for name, func in module.items():
        if name.startswith("bench_") and callable(func):
            func()
    
    analyze_records(_RECORDS)