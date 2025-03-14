# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple

import torch
import torch.backends
from tilelang import tvm as tvm
import tilelang.testing
from tvm import DataType
import tilelang as TL
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.intrinsics.mma_macro_generator import (
    TensorCoreIntrinEmitter,)
from tilelang.transform import simplify_prim_func
from tilelang.utils.tensor import map_torch_type

tilelang.testing.set_random_seed(0)


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


@simplify_prim_func
def tl_gemm(
    M,
    N,
    K,
    num_groups,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        "e4m3_float8",
    ], "Currently only e4m3_float8 is supported"
    assert out_dtype in [
        "bfloat16",
    ], "Currently only float16 is supported"

    micro_size_x = micro_size_y = micro_size_k = 16

    # This is a debug config
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 64
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)
    Scales_A_shape = (M, T.ceildiv(K, block_K) * num_groups)
    B_shape = (N, K)
    Scales_B_shape = (T.ceildiv(K, block_K), T.ceildiv(N, block_N))
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, in_dtype),
            C: T.Buffer((M, N), out_dtype),
            scales_a: T.Buffer(Scales_A_shape, "float32"),
            scales_b: T.Buffer(Scales_B_shape, "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            # Assume num_groups == 1
            Scale_A = T.alloc_shared((block_M), "float32", scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)
            C_local_final = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
                Scale_A: make_swizzle_layout(Scale_A),
            })

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for i in T.parallel(block_M):
                    Scale_A[i] = scales_a[by * block_M + i, ko]

                Scale_B = scales_b[bx, ko]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_shared,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_local, C_local)

                    C_local_final = C_local * Scale_A[ki * micro_size_k] * Scale_B


            # Perform STMatrix
            mma_emitter.stmatrix(
                C_local_final,
                C_shared,
            )

            # Store shared into global
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]

    return main


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (T.ceildiv(m, 128) * 128, T.ceildiv(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def assert_tl_gemm_correctness(M, N, K, num_groups, in_dtype, out_dtype, accum_dtype):
    gemm = tl_gemm(M, N, K, num_groups, in_dtype, out_dtype, accum_dtype)
    mod, params = TL.lower(gemm)
    src_code = mod.imported_modules[0].get_source()
    print(src_code)
    # src_code is the generated cuda source
    assert src_code is not None

    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)
    accum_dtype = map_torch_type(accum_dtype)

    A = torch.randn(M, K).to(torch.bfloat16).cuda()
    B = torch.randn(N, K).to(torch.bfloat16).cuda()
    A_fp8, A_scale = per_token_cast_to_fp8(A.clone())
    B_fp8, B_scale = per_block_cast_to_fp8(B.clone())

    C = torch.zeros(M, N, device="cuda", dtype=out_dtype)

    mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)

    mod(A_fp8, B_fp8, C, A_scale, B_scale)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch._scaled_mm(
        A_fp8,
        B_fp8,
        scale_a=A_scale,
        scale_b=B_scale,
        out_dtype=out_dtype
    )
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(8, 9)
def test_assert_tl_gemm():
    # only testing num_groups = 1 for now
    assert_tl_gemm_correctness(128, 128, 128, 1, "e4m3_float8", "bfloat16", "float32")


if __name__ == "__main__":
    tilelang.testing.main()
