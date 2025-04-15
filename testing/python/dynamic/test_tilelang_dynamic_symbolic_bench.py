# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.

import torch
import torch.backends
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import pytest

tilelang.testing.set_random_seed(0)
tilelang.disable_cache()


def tl_matmul_block_static(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    num_threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(A: T.Tensor(A_shape, in_dtype), B: T.Tensor(B_shape, in_dtype), C: T.Tensor(
        (M, N), out_dtype)):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
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


def assert_tl_matmul_block_static(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages=3,
    num_threads=128,
):
    program = tl_matmul_block_static(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program)

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

    # print(kernel.get_kernel_source())

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

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench()
    print(f"Static Latency: {latency} ms")


def tl_matmul_block_dynamic_m(
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    num_threads,
):
    M = tvm.te.var("m")
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(A: T.Tensor(A_shape, in_dtype), B: T.Tensor(B_shape, in_dtype), C: T.Tensor(
        (M, N), out_dtype)):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
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


def assert_tl_matmul_block_dynamic_m(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    num_stages=3,
    num_threads=128,
    pass_configs=None,
):
    program = tl_matmul_block_dynamic_m(
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, pass_configs=pass_configs)

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

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(input_tensors=[A, B, C])
    print(f"Dynamic M Latency with pass_configs: {pass_configs} is {latency} ms")


def tl_matmul_block_dynamic_mn(
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    num_threads,
):
    M = tvm.te.var("m")
    N = tvm.te.var("n")
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(A: T.Tensor(A_shape, in_dtype), B: T.Tensor(B_shape, in_dtype), C: T.Tensor(
        (M, N), out_dtype)):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
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


def assert_tl_matmul_block_dynamic_mn(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    num_stages=3,
    num_threads=128,
    pass_configs=None,
):
    program = tl_matmul_block_dynamic_mn(
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, pass_configs=pass_configs)

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

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(input_tensors=[A, B, C])
    print(f"Dynamic MN Latency with pass_configs: {pass_configs} is {latency} ms")


def tl_matmul_block_dynamic_mnk(
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    num_threads,
):
    M = tvm.te.var("m")
    N = tvm.te.var("n")
    K = tvm.te.var("k")

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(A: T.Tensor(A_shape, in_dtype), B: T.Tensor(B_shape, in_dtype), C: T.Tensor(
        (M, N), out_dtype)):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
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


def assert_tl_matmul_block_dynamic_mnk(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    num_stages=3,
    num_threads=128,
    pass_configs=None,
):
    program = tl_matmul_block_dynamic_mnk(
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, pass_configs=pass_configs)

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

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(input_tensors=[A, B, C])
    print(f"Dynamic MNK Latency with pass_configs: {pass_configs} is {latency} ms")


@pytest.mark.skip("Skip static test")
def test_assert_tl_matmul_block_static(M, N, K, block_M, block_N, block_K):
    assert_tl_matmul_block_static(M, N, K, block_M, block_N, block_K, False, False, "float16",
                                  "float16", "float32")


@pytest.mark.skip("Skip dynamic m test")
def test_assert_tl_matmul_block_dynamic_m(M, N, K, block_M, block_N, block_K):
    assert_tl_matmul_block_dynamic_m(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        False,
        False,
        "float16",
        "float16",
        "float32",
        pass_configs={
            "tl.disable_dynamic_tail_split": True,
            "tl.dynamic_alignment": 8
        })
    assert_tl_matmul_block_dynamic_m(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        False,
        False,
        "float16",
        "float16",
        "float32",
        pass_configs={"tl.disable_dynamic_tail_split": False})


@pytest.mark.skip("Skip dynamic mn test")
def test_assert_tl_matmul_block_dynamic_mn(M, N, K, block_M, block_N, block_K):
    assert_tl_matmul_block_dynamic_mn(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        False,
        False,
        "float16",
        "float16",
        "float32",
        pass_configs={
            "tl.disable_dynamic_tail_split": True,
            "tl.dynamic_alignment": 8
        })
    assert_tl_matmul_block_dynamic_mn(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        False,
        False,
        "float16",
        "float16",
        "float32",
        pass_configs={"tl.disable_dynamic_tail_split": False})


@pytest.mark.skip("Skip dynamic mnk test")
def test_assert_tl_matmul_block_dynamic_mnk(M, N, K, block_M, block_N, block_K):
    assert_tl_matmul_block_dynamic_mnk(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        False,
        False,
        "float16",
        "float16",
        "float32",
        pass_configs={
            "tl.disable_dynamic_tail_split": True,
            "tl.dynamic_alignment": 8
        })
    assert_tl_matmul_block_dynamic_mnk(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        False,
        False,
        "float16",
        "float16",
        "float32",
        pass_configs={"tl.disable_dynamic_tail_split": False})


def test_all():
    test_assert_tl_matmul_block_static(16384, 16384, 16384, 128, 128, 32)
    test_assert_tl_matmul_block_dynamic_m(16384, 16384, 16384, 128, 128, 32)
    test_assert_tl_matmul_block_dynamic_mn(16384, 16384, 16384, 128, 128, 32)
    test_assert_tl_matmul_block_dynamic_mnk(16384, 16384, 16384, 128, 128, 32)


if __name__ == "__main__":
    tilelang.testing.main()
