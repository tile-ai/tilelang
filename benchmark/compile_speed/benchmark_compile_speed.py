"""Cold parallel-compilation benchmark: a realistic inference kernel zoo.

Compiles a diverse zoo of ~real transformer inference kernels (GEMM projections,
GQA flash-attention, RMSNorm, SwiGLU, softmax; see ``kernel_zoo.py``) from a
*cold* cache and reports the wall-clock before/after this PR's two levers:

* **disk-save contention** — the per-kernel disk save no longer holds the global
  ``KernelCache`` lock, so workers finishing near-simultaneously stop serializing
  their multi-file save behind one mutex.
* **worker count** — ``par_compile`` now defaults to ``min(len(funcs),
  available_cpus)`` instead of the ``ThreadPoolExecutor`` default ``min(32, cpu+4)``.

The lock change is library code and cannot be toggled by a normal call, so the
``baseline`` run reconstructs the old behavior (re-wrapping the save in a global
lock, capped at 32 workers) for a self-contained before/after. Every kernel is a
distinct cold cache miss (throwaway ``TILELANG_CACHE_DIR``, disjoint shapes), so
each actually runs nvcc.

Usage::

    python benchmark_compile_speed.py            # default zoo (~126 kernels)
    python benchmark_compile_speed.py --scale 6  # larger zoo for many-core boxes
"""

import argparse
import contextlib
import os
import shutil
import tempfile
import threading
import time

import tilelang
from tilelang.cache.kernel_cache import KernelCache
from tilelang.utils.device import get_available_cpu_count

from kernel_zoo import build_zoo


@contextlib.contextmanager
def _serialized_save(enabled):
    """Re-serialize the kernel disk-save behind one lock (the pre-PR behavior)."""
    if not enabled:
        yield
        return
    original = KernelCache._save_kernel_to_disk
    guard = threading.Lock()

    def locked_save(self, *args, **kwargs):
        with guard:
            return original(self, *args, **kwargs)

    KernelCache._save_kernel_to_disk = locked_save
    try:
        yield
    finally:
        KernelCache._save_kernel_to_disk = original


def compile_zoo(scale, workers, salt, lock_save):
    """Cold-compile the zoo at `workers`; return (num_kernels, wall_seconds)."""
    cache_dir = tempfile.mkdtemp(prefix="tl_compile_bench_")
    prev = os.environ.get("TILELANG_CACHE_DIR")
    os.environ["TILELANG_CACHE_DIR"] = cache_dir
    try:
        funcs = [pf for _, pf in build_zoo(scale=scale, salt=salt)]
        with _serialized_save(lock_save):
            start = time.time()
            tilelang.par_compile(funcs, target="cuda", num_workers=workers)
            return len(funcs), time.time() - start
    finally:
        if prev is None:
            os.environ.pop("TILELANG_CACHE_DIR", None)
        else:
            os.environ["TILELANG_CACHE_DIR"] = prev
        shutil.rmtree(cache_dir, ignore_errors=True)


def main():
    cpu = get_available_cpu_count()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=int, default=3, help="zoo replication factor (~44 kernels per unit)")
    args = parser.parse_args()

    n = len(build_zoo(scale=args.scale))
    print(f"# {cpu} CPUs, {n} distinct cold kernels (scale={args.scale})")

    compile_zoo(1, min(8, cpu), salt=9, lock_save=False)  # warm up import/PCH

    _, base = compile_zoo(args.scale, min(32, cpu), salt=1, lock_save=True)
    _, cur = compile_zoo(args.scale, cpu, salt=2, lock_save=False)
    print(f"baseline (global lock, {min(32, cpu)} workers)  {base:7.2f}s")
    print(f"current  (no lock,     {cpu} workers)  {cur:7.2f}s")
    print(f"speedup  {base / cur:.2f}x")


if __name__ == "__main__":
    main()
