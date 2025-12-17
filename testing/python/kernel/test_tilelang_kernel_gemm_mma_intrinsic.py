import torch
import torch.backends
from tilelang import tvm as tvm
import tilelang.testing
from tvm import DataType
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.intrinsics.mma_macro_generator import (
    TensorCoreIntrinEmitter,
)
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
def tl_matmul(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        T.float16,
        T.bfloat16,
        T.float8_e4m3fn,
        T.float8_e5m2,
        T.int8,
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        T.float16,
        T.float32,
        T.int32,
    ], "Currently only float16, float32 and int32 are supported"

    micro_size_x = micro_size_y = micro_size_k = 16

    is_float8 = in_dtype in [
        T.float8_e4m3fn,
        T.float8_e5m2,
        T.float8_e4m3fn,
        T.float8_e5m2fnuz,
    ]
    if out_dtype == T.int32 or is_float8:
        micro_size_k = 32

    # This is a debug config
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 32 if in_dtype == T.float16 else 64
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)
    B_shape = (N, K)
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
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared),
                }
            )

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

            # Perform STMatrix
            mma_emitter.stmatrix(
                C_local,
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


def assert_tl_matmul_correctness(M, N, K, in_dtype, out_dtype, accum_dtype):
    matmul = tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype)
    kernel = tilelang.compile(matmul, out_idx=[2])
    profiler = kernel.get_profiler()

    src_code = kernel.get_kernel_source()
    # src_code is the generated cuda source
    assert src_code is not None

    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)
    accum_dtype = map_torch_type(accum_dtype)

    if in_dtype in {torch.int8, torch.int32}:
        A = torch.randint(-128, 128, (M, K), dtype=torch.int8).to(in_dtype).cuda()
        B = torch.randint(-128, 128, (N, K), dtype=torch.int8).to(in_dtype).cuda()
    elif in_dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        A = torch.randn(M, K).to(in_dtype).cuda()
        B = torch.randn(N, K).to(in_dtype).cuda()
    else:
        A = torch.randn(M, K).to(in_dtype).cuda() - 0.5
        B = torch.randn(N, K).to(in_dtype).cuda() - 0.5

    C = kernel(A, B)

    latency = profiler.do_bench()

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(out_dtype)
    tilelang.testing.torch_assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(8, 0)
def test_assert_tl_matmul():
    assert_tl_matmul_correctness(128, 128, 128, T.float16, T.float16, T.float16)
    assert_tl_matmul_correctness(128, 256, 256, T.float16, T.float32, T.float32)
    assert_tl_matmul_correctness(128, 256, 256, T.int8, T.int32, T.int32)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(8, 0)
def test_assert_tl_matmul_bfloat16():
    assert_tl_matmul_correctness(256, 256, 256, T.bfloat16, T.float32, T.float32)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(8, 9)
def test_assert_tl_matmul_fp8():
    assert_tl_matmul_correctness(128, 128, 128, T.float8_e4m3fn, T.float32, T.float32)
    assert_tl_matmul_correctness(128, 128, 128, T.float8_e5m2, T.float32, T.float32)


if __name__ == "__main__":
    tilelang.testing.main()
