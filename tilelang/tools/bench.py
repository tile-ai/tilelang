import os
import sys
import inspect
import time
import traceback
import contextlib
import warnings
import matplotlib.pyplot as plt

__all__ = ["main", "process_func"]


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
            return (f"{func.__module__}.{func.__name__}", avg_latency)
        else:
            warnings.warn(
                f"benchmark for {func.__module__}.{func.__name__} failed {fail_count} times in {repeat} repeats",
                RuntimeWarning,
                stacklevel=2,
            )
            return (f"{func.__module__}.{func.__name__}", avg_latency)
    else:
        warnings.warn(
            f"benchmark for {func.__module__}.{func.__name__} failed in all repeats (no valid run)",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

def analyze_records(records):
    # Analyze the data and draw a picture
    valid = [r for r in records if r is not None]

    name_col_width = max(len(r[0]) for r in valid)
    print("=" * (name_col_width + 20))
    head = f"{'Function':{name_col_width}s} | Avg Latency (ms)"
    print(f"{head:^(name_col_width + 20)}")
    print("-" * (name_col_width + 20))
    for name, lat in valid:
        print(f"{name:<name_col_width} | {lat:>10.4f}")
    print("=" * (name_col_width + 20))

    names = [r[0] for r in valid]
    lats = [r[1] for r in valid]
    plt.figure(figsize=(max(6, len(names) * 0.5), 6))
    plt.barh(names, lats)
    plt.xlabel("Latency (ms)")
    plt.title("Benchmark Results")

    test_file = inspect.getsourcefile(sys._getframe(2))
    out_dir = os.path.dirname(test_file)
    out_path = os.path.join(out_dir, "bench_result.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[Saved] Bar chart -> {out_path}")


def main():
    # Entry point — automatically run all bench_* functions in caller file.
    test_file = inspect.getsourcefile(sys._getframe(1))
    module = {}
    with open(test_file) as f:
        exec(f.read(), module)

    records = []
    for name, func in module.items():
        if name.startswith("bench_") and callable(func):
            records.append(func())
    
    analyze_records(records)