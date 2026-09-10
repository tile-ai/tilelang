"""Mutex-based SIMT vector add — teaching reference for get_buf/rls_buf sync.

This example demonstrates the legacy mutex-based buffer synchronization pattern using:
- T.ascend_get_buf / T.ascend_rls_buf for mutex-style buffer access control
- T.SimtVF(threads=N) for thread-parallel vector compute
- Manual T.Parallel loops (no auto-scheduling)

For production use, prefer the auto-scheduled equivalent in example_simtvf_vecadd.py.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def vector_add(N):
    NUM_BLOCKS = 64  # AI cores
    NUM_THREADS = 2048  # SIMT threads per core
    TILE_ELEMS = NUM_THREADS * 4  # 8192 floats = 32 KB per tile

    TOTAL_TILES = N // (TILE_ELEMS * NUM_BLOCKS)  # tiles per core
    NUM_STAGES = 2

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            temp1 = T.alloc_shared(
                (
                    NUM_STAGES,
                    TILE_ELEMS,
                ),
                "float32",
            )
            temp2 = T.alloc_shared(
                (
                    NUM_STAGES,
                    TILE_ELEMS,
                ),
                "float32",
            )
            temp3 = T.alloc_shared(
                (
                    NUM_STAGES,
                    TILE_ELEMS,
                ),
                "float32",
            )

            for iter in range(TOTAL_TILES):
                begin = (iter * NUM_BLOCKS + bx) * TILE_ELEMS
                end = (iter * NUM_BLOCKS + bx + 1) * TILE_ELEMS
                stage = iter % NUM_STAGES

                T.ascend_get_buf("PIPE_MTE2", stage + 2)
                T.copy(A[begin:end], temp1[stage, :])
                T.copy(B[begin:end], temp2[stage, :])
                T.ascend_rls_buf("PIPE_MTE2", stage + 2)

                T.ascend_get_buf("PIPE_V", stage + 2)
                T.ascend_get_buf("PIPE_V", stage)
                with T.SimtVF(threads=NUM_THREADS):
                    for i in T.Parallel(TILE_ELEMS):
                        temp3[stage, i] = temp1[stage, i] + temp2[stage, i]
                T.ascend_rls_buf("PIPE_V", stage)
                T.ascend_rls_buf("PIPE_V", stage + 2)

                T.ascend_get_buf("PIPE_MTE3", stage)
                T.copy(temp3[stage, :], C[begin:end])
                T.ascend_rls_buf("PIPE_MTE3", stage)

    return main


def ref_program(a, b):
    """Reference implementation: C = A + B"""
    return a + b


NUM_REPEATS = 10


def run_regression_perf(N=2**30):
    import torch

    device = torch.device("npu")
    program = vector_add(N)
    kernel = tilelang.compile(program, out_idx=-1, pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    a = torch.randn(N, dtype=torch.float32, device=device)
    b = torch.randn(N, dtype=torch.float32, device=device)
    kernel(a, b)
    torch.npu.synchronize()
    latency_ms = do_bench(lambda: kernel(a, b), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    gbps = (N * a.element_size() * 3) / (latency_ms / 1e3) / 1e9
    print(f"    [N={N}] {latency_ms * 1e3:.2f} us  |  {gbps:.1f} GB/s")
    return latency_ms


if __name__ == "__main__":
    import torch

    N = 2**30
    dtype = torch.float32
    device = torch.device("npu")

    # Create and compile kernel
    print(f"Compiling vector_add kernel (N={N})...")
    program = vector_add(N)
    kernel = tilelang.compile(program, out_idx=-1, pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    print("Compilation succeeded!")

    # Print generated source code
    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    # Create test data
    a = torch.randn(N, dtype=dtype, device=device)
    b = torch.randn(N, dtype=dtype, device=device)

    # Run kernel
    print("Running kernel on NPU...")
    c = kernel(a, b)
    torch.npu.synchronize()

    # Verify
    expected = ref_program(a, b)
    if not torch.equal(c, expected):
        max_diff = torch.max(torch.abs(c - expected)).item()
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")
    print("\nVerification passed!")

    # Benchmark
    latency_ms = do_bench(lambda: kernel(a, b), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    print(f"{latency_ms:.1f} ms/iter  |  {(N * a.element_size() * 3) / (latency_ms / 1e3) / 1e9:.1f} GB/s")
