"""Pipelined vector addition using either SIMT or explicit SIMD instructions."""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def vector_add(N, mode="simt"):
    num_blocks, num_threads, tile, num_stages = 64, 2048, 8192, 2
    if mode not in ("simt", "simd"):
        raise ValueError("mode must be simt or simd")
    if N <= 0 or N % (tile * num_blocks):
        raise ValueError(f"N must be a positive multiple of {tile * num_blocks}")
    iterations = N // (tile * num_blocks)

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(num_blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a: num_stages, b: num_stages, c: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)
                if mode == "simt":
                    with T.SimtVF(threads=num_threads):
                        for i in T.Parallel(tile):
                            c[i] = a[i] + b[i]
                else:
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        for i in range(tile // 64):
                            a_reg = T.simd.vld(a[i * 64])
                            b_reg = T.simd.vld(b[i * 64])
                            result = T.simd.vadd(a_reg, b_reg, mask)
                            T.simd.vsts(c[i * 64], result, mask)
                T.copy(c, C[begin : begin + tile])

    return main


def run_regression_perf(N=2**30, mode="simt"):
    kernel = tilelang.compile(vector_add(N, mode), target="ascend", out_idx=-1)
    a, b = torch.randn(N, device="npu"), torch.randn(N, device="npu")
    latency = do_bench(lambda: kernel(a, b), backend="msprof", _n_warmup=30, _n_repeat=10)
    num_bytes = 3 * N * a.element_size()  # Two input reads and one output write.
    bandwidth = num_bytes / (latency * 1e6)
    print(f"vecadd ({mode}, N={N}): {latency * 1000:.2f} us | {bandwidth:.2f} GB/s")
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "simt", "simd"], default="all")
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks")
    args = parser.parse_args()
    n = 2**20
    a, b = torch.randn(n, device="npu"), torch.randn(n, device="npu")
    for mode in ("simt", "simd") if args.mode == "all" else (args.mode,):
        kernel = tilelang.compile(vector_add(n, mode), target="ascend", out_idx=-1)
        torch.testing.assert_close(kernel(a, b), a + b)
        print(f"vecadd ({mode}): correctness passed")
        if args.bench:
            run_regression_perf(mode=mode)
