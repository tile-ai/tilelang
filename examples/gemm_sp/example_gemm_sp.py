import argparse

import tilelang
import tilelang.language as T

from tilelang.layout import make_cutlass_metadata_layout
from tilelang.utils.sparse import compress, randn_semi_sparse
from tilelang.contrib import nvcc
from triton.testing import do_bench

import torch

arch = nvcc.get_target_compute_version()

DEFAULT_CONFIG = {  # take best config from autotune script
    "4090": {
        T.float: {
            "block_M": 128,
            "block_N": 64,
            "block_K": 64,
            "num_stages": 1,
            "thread_num": 128,
            "policy": T.GemmWarpPolicy.Square,
            "enable_rasterization": True,
        },
        T.float16: {
            "block_M": 256,
            "block_N": 128,
            "block_K": 64,
            "num_stages": 2,
            "thread_num": 128,
            "policy": T.GemmWarpPolicy.Square,
            "enable_rasterization": True,
        },
    },
    "h20": {
        T.float: {
            "block_M": 128,
            "block_N": 64,
            "block_K": 128,
            "num_stages": 3,
            "thread_num": 128,
            "policy": T.GemmWarpPolicy.Square,
            "enable_rasterization": True,
        },
        T.float16: {
            "block_M": 128,
            "block_N": 64,
            "block_K": 128,
            "num_stages": 3,
            "thread_num": 128,
            "policy": T.GemmWarpPolicy.Square,
            "enable_rasterization": True,
        },
    },
}

ARCH_INFO = {"8.0": (16, "int16"), "8.9": (16, "int16"), "9.0": (8, "uint8")}


@tilelang.jit(out_idx=[-1])
def matmul_sp_fp16(M, N, K, accum_dtype, block_M, block_N, block_K, num_stages, thread_num, policy, enable_rasterization):
    e_factor, e_dtype = ARCH_INFO[arch]

    @T.prim_func
    def gemm_sp_fp16(
        A_sparse: T.Tensor((M, K // 2), T.float16),
        E: T.Tensor((M, K // e_factor), e_dtype),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), T.float16)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.disable_warp_group_reg_alloc()
            T.use_swizzle(panel_size=10, enable=enable_rasterization)
            T.annotate_layout(
                {
                    E: make_cutlass_metadata_layout(E, mma_dtype=T.float16, block_k=block_K, arch=arch),
                    E_shared: make_cutlass_metadata_layout(E_shared, mma_dtype=T.float16, block_k=block_K, arch=arch),
                }
            )
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_local, False, False, policy=policy)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm_sp_fp16


def main():
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    parser.add_argument("--accum_dtype", type=str, default=T.float, choices=[T.float, T.float16], help="Accumulation datatype")
    parser.add_argument("--cfg", type=str, choices=["4090", "h20"], default="4090")
    args = parser.parse_args()
    kernel = matmul_sp_fp16(args.m, args.n, args.k, args.accum_dtype, **DEFAULT_CONFIG[args.cfg][args.accum_dtype])

    a = randn_semi_sparse(args.m, args.k, device="cuda", dtype=torch.half)
    b = torch.randn(args.k, args.n, device="cuda", dtype=torch.half)

    a_sparse, e = compress(a, transposed=False, block_k=DEFAULT_CONFIG[args.cfg][args.accum_dtype]["block_K"], arch=arch)
    c = kernel(a_sparse, e, b)

    ref_c = a @ b

    assert not c.isnan().any(), "Reference result contains NaNs, please report an issue"
    torch.testing.assert_close(c, ref_c.to(c.dtype), rtol=1e-2, atol=1e-2)
    print(f"Precision check passed. diff: {(c - ref_c).abs().mean()}")

    latency = do_bench(lambda: kernel(a_sparse, e, b))
    ref_latency = do_bench(lambda: a @ b)

    total_flops = 2 * args.m * args.n * args.k
    tflops = total_flops / latency / 1e9
    ref_tflops = total_flops / ref_latency / 1e9
    print(f"Sparse TFLOPS: {tflops:.2f}, Latency: {latency / 1e3} s")
    print(f"Reference TFLOPS: {ref_tflops:.2f}, Latency: {ref_latency / 1e3:} s")


if __name__ == "__main__":
    main()
