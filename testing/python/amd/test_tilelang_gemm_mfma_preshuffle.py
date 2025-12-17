import pytest
import torch
import tilelang.testing
from tilelang import tvm as tvm
import tilelang.language as T
from tilelang.intrinsics import make_mfma_swizzle_layout as make_swizzle_layout
from tilelang.intrinsics.mfma_macro_generator import MatrixCorePreshuffleIntrinEmitter
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
    b_preshuffle=False,
    b_g2l_load=False,
):
    micro_size_x = micro_size_y = micro_size_k = 16

    if in_dtype in {T.float8_e4m3fnuz, T.int8}:
        micro_size_k = 32

    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32

    # for preshuffle_b, warp_layout = {1, 4}
    if b_preshuffle:
        block_row_warps = 1
        block_col_warps = 4
        warp_row_tiles = 64
        warp_col_tiles = 16

    chunk = 256 * k_pack

    pack_size_k = micro_size_k * k_pack

    shared_scope = "shared"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (K, M) if a_transposed else (M, K)
    if b_preshuffle:
        B_shape = (
            (N // micro_size_y, K // pack_size_k, micro_size_y, pack_size_k)
            if b_transposed
            else (K // pack_size_k, N // micro_size_y, pack_size_k, micro_size_y)
        )
    else:
        B_shape = (N, K) if b_transposed else (K, N)

    A_shared_shape = (block_K, block_M) if a_transposed else (block_M, block_K)
    if b_preshuffle:
        B_shared_shape = (
            (block_N // micro_size_y, block_K // pack_size_k, micro_size_y, pack_size_k)
            if b_transposed
            else (block_K // pack_size_k, block_N // micro_size_y, pack_size_k, micro_size_y)
        )
    else:
        B_shared_shape = (block_N, block_K) if b_transposed else (block_K, block_N)

    warp_size = 64
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (k_pack * micro_size_x * micro_size_k) // warp_size
    local_size_b = (k_pack * micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mfma_emitter = MatrixCorePreshuffleIntrinEmitter(
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
        b_preshuffle=b_preshuffle,
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
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_swizzle_layout(A_shared),
                }
            )

            num_ko = K // block_K
            num_ki = block_K // (k_pack * micro_size_k)

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined(num_ko, num_stages=0):
                # Load A into shared memory
                if a_transposed:
                    T.copy(A[ko * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, ko * block_K], A_shared)

                # Load B into shared memory
                if b_g2l_load is False:
                    if b_transposed:
                        for j, k, jj, kk in T.Parallel(block_N // micro_size_y, block_K // pack_size_k, micro_size_y, pack_size_k):
                            B_shared[j, k, jj, kk] = B[bx * block_N // micro_size_y + j, ko * block_K // pack_size_k + k, jj, kk]
                    else:
                        for k, j, kk, jj in T.Parallel(block_K // pack_size_k, block_N // micro_size_y, pack_size_k, micro_size_y):
                            B_shared[k, j, kk, jj] = B[ko * block_K // pack_size_k + k, bx * block_N // micro_size_y + j, kk, jj]

                for ki in T.serial(0, num_ki):
                    # Load A S2L
                    mfma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                    )

                    if b_g2l_load:
                        # Load B G2L
                        mfma_emitter.ldmatrix_b(B_local, B, ki + ko * num_ki, pid_m=by, pid_n=bx)
                    else:
                        # Load B S2L
                        mfma_emitter.ldmatrix_b(
                            B_local,
                            B_shared,
                            ki,
                        )

                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_local, B_local, C_local)

            # Perform STMatrix
            mfma_emitter.stmatrix(
                C_local,
                C,
                pid_m=by,
                pid_n=bx,
            )

    return main


def shuffle_weight(
    x: torch.Tensor,
    layout=(16, 32),
    k_pack=1,
    is_transpose=False,
) -> torch.Tensor:
    IN, IK = layout
    BK = IK * k_pack
    BN = IN

    N, K = (x.shape[-2], x.shape[-1]) if is_transpose else (x.shape[-1], x.shape[-2])
    assert N % BN == 0
    assert K % BK == 0

    x = x.view(N // BN, BN, K // BK, BK) if is_transpose else x.view(K // BK, BK, N // BN, BN)
    x = x.permute(0, 2, 1, 3)
    return x.contiguous()


def assert_tl_matmul_correctness(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype=T.float32,
    a_transposed=False,
    b_transposed=True,
    k_pack=1,
    b_preshuffle=False,
    b_g2l_load=False,
):
    matmul = tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype, a_transposed, b_transposed, k_pack, b_preshuffle, b_g2l_load)
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

    B_preshuffle = B
    if b_preshuffle:
        B_preshuffle = shuffle_weight(B_preshuffle, k_pack=k_pack, is_transpose=b_transposed)
        kernel(A, B_preshuffle, C)
    else:
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
    "M, N, K, in_dtype, out_dtype, accum_dtype, a_transposed, b_transposed, k_pack, b_preshuffle, b_g2l_load",
    [
        (256, 256, 512, T.int8, T.int32, T.int32, False, True, 1, True, False),
        (256, 256, 512, T.int8, T.int32, T.int32, False, False, 1, True, False),
        (256, 256, 512, T.int8, T.int32, T.int32, False, True, 2, True, False),
        (256, 256, 512, T.int8, T.int32, T.int32, False, False, 2, True, False),
        (256, 256, 512, T.float8_e4m3fnuz, T.float32, T.float32, False, True, 1, True, False),
        (256, 256, 512, T.float8_e4m3fnuz, T.float32, T.float32, False, False, 1, True, False),
        (256, 256, 512, T.float8_e4m3fnuz, T.float32, T.float32, False, True, 2, True, False),
        (256, 256, 512, T.float8_e4m3fnuz, T.float32, T.float32, False, False, 2, True, False),
    ],
)
@tilelang.testing.requires_rocm
def test_assert_tl_matmul(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    a_transposed,
    b_transposed,
    k_pack,
    b_preshuffle,
    b_g2l_load,
):
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
        b_preshuffle=b_preshuffle,
        b_g2l_load=b_g2l_load,
    )


if __name__ == "__main__":
    tilelang.testing.main()
