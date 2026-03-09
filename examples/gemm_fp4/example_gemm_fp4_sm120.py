"""FP4 (float4_e2m1fn) GEMM on SM120 (RTX 5080/5090) using fragment-based MMA.

Uses mma.sync.aligned.kind::f8f6f4 instructions (not TCGEN05/TMEM).
Addresses https://github.com/tile-ai/tilelang/issues/1592
"""

import torch
import tilelang
import tilelang.language as T


def matmul_fp4(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages=2,
    threads=128,
):
    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


M, N, K = 256, 256, 256
block_M, block_N, block_K = 128, 128, 128
in_dtype = T.float4_e2m1fn
out_dtype = T.float32
accum_dtype = T.float32
num_stages = 2
threads = 128

print(f"Running FP4 GEMM: M={M}, N={N}, K={K}")
print(f"  block_M={block_M}, block_N={block_N}, block_K={block_K}")
print(f"  in_dtype={in_dtype}, out_dtype={out_dtype}, accum_dtype={accum_dtype}")

func = matmul_fp4(
    M, N, K,
    block_M, block_N, block_K,
    in_dtype, out_dtype, accum_dtype,
    num_stages, threads,
)

jit_kernel = tilelang.compile(
    func,
    out_idx=[2],
    target="cuda",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)

print("Compilation succeeded!")
print(jit_kernel.get_kernel_source())

profiler = jit_kernel.get_profiler()
latency = profiler.do_bench()
print(f"Latency: {latency} ms")
print(f"TFLOPS:  {2 * M * N * K / (latency / 1e3) / 1e12:.2f}")
