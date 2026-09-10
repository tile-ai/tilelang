"""Ascend GM float32 atomic add, max, and min examples."""

import tilelang
import tilelang.ascend.language as T

from tilelang.profiler import do_bench


def atomic_add_gm_float(N, num_blocks=8, threads=256):
    """Atomically add 1.0 into a GM float counter from each thread.

    Total expected = num_blocks * threads * 1.0.
    """

    @T.prim_func
    def main(
        counter: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(num_blocks), T.SimtVF(threads=threads):
            for _ in T.Parallel(threads):
                T.atomic_add(counter[0], 1.0)

    return main


def atomic_max_gm_float(N, num_blocks=8, threads=256):
    """Atomically reduce unique per-thread values into a GM float maximum."""

    @T.prim_func
    def main(
        counter: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(num_blocks) as bx, T.SimtVF(threads=threads):
            for tx in T.Parallel(threads):
                T.atomic_max(counter[0], T.cast(bx * threads + tx, T.float32))

    return main


def atomic_min_gm_float(N, num_blocks=8, threads=256):
    """Atomically reduce unique per-thread values into a GM float minimum."""

    @T.prim_func
    def main(
        counter: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(num_blocks) as bx, T.SimtVF(threads=threads):
            for tx in T.Parallel(threads):
                T.atomic_min(counter[0], T.cast(bx * threads + tx, T.float32))

    return main


def run_regression_perf(num_blocks=64, threads=2048):
    import torch

    device = torch.device("npu")
    bench_kernel = tilelang.compile(atomic_add_gm_float(1, num_blocks, threads))
    counter = torch.zeros(1, dtype=torch.float32, device=device)

    def run():
        bench_kernel(counter)

    run()
    torch.npu.synchronize()
    latency_ms = do_bench(run, backend="msprof", _n_warmup=10, _n_repeat=50)
    print(f"  {latency_ms:.3f} ms/iter")
    return latency_ms


if __name__ == "__main__":
    import torch

    device = torch.device("npu")

    num_blocks, threads = 8, 256
    num_updates = num_blocks * threads
    cases = [
        ("atomic_add", atomic_add_gm_float, 0.0, float(num_updates)),
        ("atomic_max", atomic_max_gm_float, -float("inf"), float(num_updates - 1)),
        ("atomic_min", atomic_min_gm_float, float("inf"), 0.0),
    ]
    for name, program_factory, initial, expected in cases:
        print(f"=== float32 GM {name} ===")
        kernel = tilelang.compile(program_factory(1, num_blocks, threads))
        print(kernel.get_kernel_source())

        counter = torch.full((1,), initial, dtype=torch.float32, device=device)
        kernel(counter)
        torch.npu.synchronize()
        actual = counter.item()
        print(f"  Expected: {expected}, Got: {actual}")
        assert abs(actual - expected) < 0.5, f"Mismatch: {actual} != {expected}"
        print("  PASS")

    # Benchmark
    print("\n=== Benchmark ===")
    bench_kernel = tilelang.compile(atomic_add_gm_float(1, 64, 2048))
    counter = torch.zeros(1, dtype=torch.float32, device=device)

    def run():
        bench_kernel(counter)

    latency_ms = do_bench(run, backend="msprof", _n_warmup=10, _n_repeat=50)
    print(f"  {latency_ms:.3f} ms/iter")
