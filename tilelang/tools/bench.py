import os
import re
import sys
import inspect
import traceback
import contextlib
import warnings
from tabulate import tabulate
import matplotlib.pyplot as plt
import importlib.util
import multiprocessing as mp

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


def process_func(func, *args, name=None, **kwargs):
    import torch
    latency = None
    try:
        with suppress_output():
            latency = func(*args, **kwargs)
            torch.cuda.synchronize()
    except Exception:
        pass

    if name is None:
        name = func.__module__
    if latency is not None:
        _RECORDS.append((f"{name}", latency))
        print(f"{name}", latency)
    else:
        warnings.warn(
            f"benchmark for {name} failed",
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


def _load_module(full_path):
    module_name = os.path.splitext(os.path.basename(full_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bench_worker(file_path, func_name, queue):
    import torch
    a = torch.randn(1, device="cuda")
    b = torch.randn(1, device="cuda")
    (a + b).sum().item()
    torch.cuda.synchronize()
    local_records = []
    global _RECORDS
    _RECORDS = local_records
    try:
        mod = _load_module(file_path)
        func = getattr(mod, func_name)
        func()
    except Exception:
        traceback.print_exc()
    finally:
        queue.put(local_records)


def main():
    # Entry point — automatically run all bench_* functions in caller file.
    mp.set_start_method("spawn", force=True)
    test_file = inspect.getsourcefile(sys._getframe(1))
    out_dir = os.path.dirname(test_file)
    module = {}
    with open(test_file) as f:
        exec(f.read(), module)

    bench_funcs = []
    for name, func in module.items():
        if name.startswith("bench_") and callable(func):
            bench_funcs.append((test_file, name))

    queue = mp.Queue()

    for file_path, func_name in bench_funcs:
        p = mp.Process(target=_bench_worker, args=(file_path, func_name, queue))
        p.start()
        p.join()

        if p.exitcode == 0:
            try:
                child_records = queue.get_nowait()
            except Exception:
                child_records = []
            _RECORDS.extend(child_records)
        else:
            print(f"[SKIP] {file_path}:{func_name} crashed, skipping this benchmark.")

        print(len(_RECORDS))

    analyze_records(_RECORDS, out_dir)


def bench_all():
    # Do benchmark for all bench_* functions in examples
    mp.set_start_method("spawn", force=True)
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
                            bench_funcs.append((full_path, name))

    queue = mp.Queue()

    for file_path, func_name in bench_funcs:
        p = mp.Process(target=_bench_worker, args=(file_path, func_name, queue))
        p.start()
        p.join()

        if p.exitcode == 0:
            try:
                child_records = queue.get_nowait()
            except Exception:
                child_records = []
            _RECORDS.extend(child_records)
        else:
            print(f"[SKIP] {file_path}:{func_name} crashed, skipping this benchmark.")

        print(len(_RECORDS))

    if _RECORDS:
        print(tabulate(_RECORDS, tablefmt="github", stralign="left", numalign="decimal"))
    else:
        print("[WARN] no benchmark records collected.")
