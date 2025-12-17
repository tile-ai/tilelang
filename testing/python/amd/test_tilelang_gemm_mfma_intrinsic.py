import pytest
import torch
import tilelang.testing
from tilelang import tvm as tvm
import tilelang.language as T
from tilelang.intrinsics import make_mfma_swizzle_layout as make_swizzle_layout
from tilelang.intrinsics.mfma_macro_generator import (
    MatrixCoreIntrinEmitter,
)
from tilelang.transform import simplify_prim_func

tilelang.testing.set_random_seed(0)


@simplify_prim_func
def tl_matmul(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    a_transposed=False,
    b_transposed=True,
    k_pack=1,
):
    micro_size_x = micro_size_y = micro_size_k = 16

    if in_dtype in {T.float8_e4m3fnuz, T.int8}:
        micro_size_k = 32

    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32

    chunk = 32 * k_pack

    shared_scope = "shared"
    cache_write_shared = False

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (K, M) if a_transposed else (M, K)
    B_shape = (N, K) if b_transposed else (K, N)
    A_shared_shape = (block_K, block_M) if a_transposed else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if b_transposed else (block_K, block_N)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 64
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (k_pack * micro_size_x * micro_size_k) // warp_size
    local_size_b = (k_pack * micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mfma_emitter = MatrixCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=a_transposed,
        b_transposed=b_transposed,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        k_pack=k_pack,
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

            for ko in T.Pipelined((K // block_K), num_stages=0):
                # Load A into shared memory
                if a_transposed:
                    T.copy(A[ko * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, ko * block_K], A_shared)

                # Load B into shared memory
                if b_transposed:
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                else:
                    T.copy(B[ko * block_K, bx * block_N], B_shared)

                for ki in T.serial(0, (block_K // (k_pack * micro_size_k))):
                    # Load A into fragment
                    mfma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                    )

                    # Load B into fragment
                    mfma_emitter.ldmatrix_b(
                        B_local,
                        B_shared,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_local, B_local, C_local)

            # Perform STMatrix
            if cache_write_shared:
                mfma_emitter.stmatrix(
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
            else:
                mfma_emitter.stmatrix(
                    C_local,
                    C,
                    pid_m=by,
                    pid_n=bx,
                )

    return main


def assert_tl_matmul_correctness(M, N, K, in_dtype, out_dtype, accum_dtype=T.float32, a_transposed=False, b_transposed=True, k_pack=1):
    matmul = tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype, a_transposed, b_transposed, k_pack)
    print(matmul)
    kernel = tilelang.compile(matmul)
    src_code = kernel.get_kernel_source()
    # src_code is the generated cuda source
    assert src_code is not None
    A_shape = (K, M) if a_transposed else (M, K)
    B_shape = (N, K) if b_transposed else (K, N)
    if in_dtype == T.int8:
        A = torch.randint(-128, 127, A_shape, device="cuda", dtype=torch.int8)
        B = torch.randint(-128, 127, B_shape, device="cuda", dtype=torch.int8)
    elif in_dtype == T.float8_e4m3fnuz:
        A = torch.rand(A_shape, device="cuda", dtype=torch.float16).to(getattr(torch, in_dtype))
        B = torch.rand(B_shape, device="cuda", dtype=torch.float16).to(getattr(torch, in_dtype))
    else:
        A = torch.rand(A_shape, device="cuda", dtype=getattr(torch, in_dtype))
        B = torch.rand(B_shape, device="cuda", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

    kernel(A, B, C)
    print(kernel.get_kernel_source())

    profiler = kernel.get_profiler()

    latency = profiler.do_bench()

    # Ensure that the latency is not None
    assert latency is not None

    if a_transposed and b_transposed:
        # Get Reference Result
        ref_c = torch.matmul(A.T.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))
    elif a_transposed and not b_transposed:
        # Get Reference Result
        ref_c = torch.matmul(A.Tto(torch.float32), B.to(torch.float32)).to(getattr(torch, out_dtype))
    elif not a_transposed and b_transposed:
        # Get Reference Result
        ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))
    else:
        # Get Reference Result
        ref_c = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(getattr(torch, out_dtype))

    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "M, N, K, in_dtype, out_dtype, accum_dtype, a_transposed, b_transposed, k_pack",
    [
        (128, 128, 128, T.float16, T.float16, T.float32, False, True, 1),
        (128, 256, 256, T.float16, T.float32, T.float32, False, True, 1),
        (128, 256, 256, T.float16, T.float32, T.float32, False, True, 2),
        (128, 128, 128, T.int8, T.int32, T.int32, False, True, 1),
        (128, 256, 256, T.int8, T.int32, T.int32, False, True, 1),
        (128, 256, 256, T.int8, T.int32, T.int32, False, True, 2),
        (128, 256, 256, T.int8, T.int32, T.int32, False, False, 1),
        (128, 256, 256, T.int8, T.int32, T.int32, False, False, 2),
        (128, 128, 128, T.float8_e4m3fnuz, T.float16, T.float32, False, True, 1),
    ],
)
@tilelang.testing.requires_rocm
def test_assert_tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype, a_transposed, b_transposed, k_pack):
    assert_tl_matmul_correctness(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        accum_dtype=accum_dtype,
        a_transposed=a_transposed,
        b_transposed=b_transposed,
        k_pack=k_pack,
    )
    assert_tl_matmul_correctness(128, 256, 256, T.float8_e4m3fnuz, T.float32)
    assert_tl_matmul_correctness(128, 256, 256, T.float8_e4m3fnuz, T.float32, k_pack=2)
    assert_tl_matmul_correctness(128, 256, 256, T.float8_e4m3fnuz, T.float32, b_transposed=False)
    assert_tl_matmul_correctness(128, 256, 256, T.float8_e4m3fnuz, T.float32, b_transposed=False, k_pack=2)


if __name__ == "__main__":
    tilelang.testing.main()
