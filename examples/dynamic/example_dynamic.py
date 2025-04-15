# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm

tilelang.testing.set_random_seed(0)
tilelang.disable_cache()


def matmul_dynamic_mnk(
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    M = tvm.te.var("m")
    N = tvm.te.var("n")
    K = tvm.te.var("k")

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)

    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

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
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def test_matmul_dynamic(M, N, K, block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype,
                        accum_dtype, num_stages, threads):
    print(
        f"M: {M}, N: {N}, K: {K}, block_M: {block_M}, block_N: {block_N}, block_K: {block_K}, trans_A: {trans_A}, trans_B: {trans_B}, in_dtype: {in_dtype}, out_dtype: {out_dtype}, accum_dtype: {accum_dtype}, num_stages: {num_stages}, threads: {threads}"
    )
    program = matmul_dynamic_mnk(block_M, block_N, block_K, trans_A, trans_B, in_dtype, out_dtype,
                                 accum_dtype, num_stages, threads)

    kernel = tilelang.compile(
        program, pass_configs={
            "tl.disable_dynamic_tail_split": True,
            "tl.dynamic_alignment": 8
        })

    import torch
    if trans_A:
        A = torch.rand(K, M, device="cuda", dtype=getattr(torch, in_dtype))
    else:
        A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    if trans_B:
        B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
    else:
        B = torch.rand(K, N, device="cuda", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

    kernel(A, B, C)

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    # Get Reference Result
    ref_c = ref_program(A, B)

    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(input_tensors=[A, B, C])
    print(f"Latency: {latency} ms")


if __name__ == "__main__":
    test_matmul_dynamic(16384, 16384, 16384, 128, 128, 32, False, False, "float16", "float16",
                        "float32", 3, 128)
