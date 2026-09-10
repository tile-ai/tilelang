"""Cross-level multi-buffer example for Ascend auto-schedule.

`b_ub` is loaded from A at the OUTER pipelined level and consumed by an INNER
loop (SimdVF + store). The dependency on `b_ub` is collected at the inner level
while `b_ub` is multi-buffered at the outer level, so the sync flags must be
versioned by the outer (claim-level) iteration. This exercises that path.
"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench

NUM_CORES = 64
TILE_ELEMS = 8192
OUTER = 128
INNER = 1
NUM_STAGES = 2
N = NUM_CORES * TILE_ELEMS * OUTER
NUM_REPEATS = 50


def crosslevel_multibuffer():
    @T.prim_func
    def main(
        A: T.Tensor((N,), T.float32),
        C: T.Tensor((INNER, N), T.float32),
    ):
        with T.Kernel(NUM_CORES) as bx:
            b_ub = T.alloc_shared((TILE_ELEMS,), T.float32)

            for i in T.Pipelined(OUTER, num_stages=NUM_STAGES):
                begin = (i * NUM_CORES + bx) * TILE_ELEMS
                end = begin + TILE_ELEMS

                T.copy(A[begin:end], b_ub)

                for j in T.serial(INNER):
                    with T.SimdVF():
                        for k in T.Parallel(TILE_ELEMS):
                            b_ub[k] = b_ub[k] + T.float32(1.0)
                    T.copy(b_ub, C[j, begin:end])

    return main


def ref_program(a):
    # b_ub = A_tile; the SimdVF "+1" runs before the store, so C[j] = A + (j + 1).
    return torch.stack([a + float(j + 1) for j in range(INNER)])


def run_regression_perf():
    device = torch.device("npu")
    kernel = tilelang.compile(crosslevel_multibuffer(), out_idx=-1)
    a = torch.randn(N, dtype=torch.float32, device=device)
    kernel(a)
    torch.npu.synchronize()
    latency_ms = do_bench(lambda: kernel(a), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    elapsed_us = latency_ms * 1e3
    bytes_moved = N * 4 * (1 + INNER)
    bw_gbs = bytes_moved / (elapsed_us * 1e-6) / 1e9
    print(f"  {elapsed_us:.1f} us  {bw_gbs:.0f} GB/s")
    return latency_ms


if __name__ == "__main__":
    device = torch.device("npu")
    kernel = tilelang.compile(crosslevel_multibuffer(), out_idx=-1)

    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    a = torch.randn(N, dtype=torch.float32, device=device)
    out = kernel(a)
    torch.npu.synchronize()

    expected = ref_program(a)
    max_diff = (out - expected).abs().max().item()
    ok = "PASS" if max_diff < 1e-2 else f"FAIL (diff={max_diff:.1e})"

    latency_ms = do_bench(lambda: kernel(a), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    elapsed_us = latency_ms * 1e3
    bytes_moved = N * 4 * (1 + INNER)
    bw_gbs = bytes_moved / (elapsed_us * 1e-6) / 1e9
    print(f"  {elapsed_us:.1f} us  {bw_gbs:.0f} GB/s  {ok}")
