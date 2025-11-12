import os
import sys
import inspect
import time
import traceback
import contextlib

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
            print(f"{func.__module__}.{func.__name__}  {avg_latency:.2f} ms")
        else:
            print(
                f"{func.__module__}.{func.__name__}  {avg_latency:.2f} ms  (fail {fail_count}/{repeat})"
            )
    else:
        print(f"{func.__module__}.{func.__name__}  FAILED (no valid run)")


def main():
    # Entry point — automatically run all bench_* functions in caller file.
    test_file = inspect.getsourcefile(sys._getframe(1))
    module = {}
    with open(test_file) as f:
        exec(f.read(), module)

    for name, func in module.items():
        if name.startswith("bench_") and callable(func):
            func()
