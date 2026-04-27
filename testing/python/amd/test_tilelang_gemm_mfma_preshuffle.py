import pytest
import torch
import tilelang
import tilelang.testing
from tilelang import tvm as tvm
import tilelang.language as T
from tilelang.intrinsics import make_mfma_swizzle_layout as make_swizzle_layout
from tilelang.intrinsics.mfma_macro_generator import MatrixCorePreshuffleIntrinEmitter
from tilelang.transform import simplify_prim_func
from tilelang.utils import determine_fp8_type

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
    a_preshuffle=False,
    b_preshuffle=False,
    b_g2l_load=False,
    block_row_warps=None,
    block_col_warps=None,
    warp_row_tiles=None,
    warp_col_tiles=None,
    chunk=None,
    num_stages=0,
    panel_size=10,
    mfma_shape=None,
):
    """Build a TileLang MFMA kernel for ``A @ B^T`` (with optional preshuffle).

    The (block_*_warps, warp_*_tiles, chunk, num_stages, panel_size) parameters
    expose the underlying CK-style template knobs so that an external autotuner
    can pick per-shape tile / wave / pipeline configurations.

    ``mfma_shape`` selects which MFMA instruction to use, as an ``(M, N, K)``
    tuple.  Supported int8 shapes on CDNA4 (gfx950):
        (16, 16, 32)  — default, ``v_mfma_i32_16x16x32_i8``
        (16, 16, 64)  — doubled-K, ``v_mfma_i32_16x16x64_i8``
        (32, 32, 32)  — doubled-MN, ``v_mfma_i32_32x32x32_i8``
    """
    if mfma_shape is not None:
        micro_size_x, micro_size_y, micro_size_k = mfma_shape
    else:
        micro_size_x = micro_size_y = 16
        micro_size_k = 32 if in_dtype.bits == 8 else 16

    if block_row_warps is None:
        block_row_warps = 2
    if block_col_warps is None:
        block_col_warps = 2
    if warp_row_tiles is None:
        warp_row_tiles = max(32, micro_size_x)
    if warp_col_tiles is None:
        warp_col_tiles = max(32, micro_size_y)

    # Legacy heuristic: if the caller did not override any tile knob and we are
    # in B-only preshuffle mode, keep the historical 1x4 warp grid.
    _all_tile_defaults = (
        block_row_warps == 2
        and block_col_warps == 2
        and warp_row_tiles == max(32, micro_size_x)
        and warp_col_tiles == max(32, micro_size_y)
    )
    if _all_tile_defaults and b_preshuffle and not a_preshuffle:
        block_row_warps, block_col_warps = 1, 4
        warp_row_tiles = max(64, micro_size_x)
        warp_col_tiles = max(16, micro_size_y)

    if chunk is None:
        chunk = 256 * k_pack

    # ---- structural validation (catch invalid configs early) ----
    assert warp_row_tiles % micro_size_x == 0, f"warp_row_tiles={warp_row_tiles} must be a multiple of micro_size_x={micro_size_x}"
    assert warp_col_tiles % micro_size_y == 0, f"warp_col_tiles={warp_col_tiles} must be a multiple of micro_size_y={micro_size_y}"
    assert chunk % (k_pack * micro_size_k) == 0, f"chunk={chunk} must be a multiple of k_pack*micro_size_k={k_pack * micro_size_k}"
    block_M_check = block_row_warps * warp_row_tiles
    block_N_check = block_col_warps * warp_col_tiles
    assert M % block_M_check == 0, f"M={M} must be a multiple of block_M={block_M_check}"
    assert N % block_N_check == 0, f"N={N} must be a multiple of block_N={block_N_check}"
    assert K % chunk == 0, f"K={K} must be a multiple of chunk={chunk}"

    pack_size_k = micro_size_k * k_pack

    shared_scope = "shared"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    if a_preshuffle:
        A_shape = (
            (K // pack_size_k, M // micro_size_x, pack_size_k, micro_size_x)
            if a_transposed
            else (M // micro_size_x, K // pack_size_k, micro_size_x, pack_size_k)
        )
    else:
        A_shape = (K, M) if a_transposed else (M, K)
    if b_preshuffle:
        B_shape = (
            (N // micro_size_y, K // pack_size_k, micro_size_y, pack_size_k)
            if b_transposed
            else (K // pack_size_k, N // micro_size_y, pack_size_k, micro_size_y)
        )
    else:
        B_shape = (N, K) if b_transposed else (K, N)

    if a_preshuffle:
        A_shared_shape = (
            (block_K // pack_size_k, block_M // micro_size_x, pack_size_k, micro_size_x)
            if a_transposed
            else (block_M // micro_size_x, block_K // pack_size_k, micro_size_x, pack_size_k)
        )
    else:
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
        a_preshuffle=a_preshuffle,
        b_preshuffle=b_preshuffle,
        mfma_shape=mfma_shape,
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

            layout_map = {}
            if not a_preshuffle:
                layout_map[A_shared] = make_swizzle_layout(A_shared)
            if not b_preshuffle:
                layout_map[B_shared] = make_swizzle_layout(B_shared)
            if layout_map:
                T.annotate_layout(layout_map)

            num_ko = K // block_K
            num_ki = block_K // (k_pack * micro_size_k)

            # Improve L2 Cache
            T.use_swizzle(panel_size=panel_size)

            T.clear(C_local)

            for ko in T.Pipelined(num_ko, num_stages=num_stages):
                # Load A into shared memory
                if a_preshuffle:
                    if a_transposed:
                        for k, i, kk, ii in T.Parallel(block_K // pack_size_k, block_M // micro_size_x, pack_size_k, micro_size_x):
                            A_shared[k, i, kk, ii] = A[ko * block_K // pack_size_k + k, by * block_M // micro_size_x + i, kk, ii]
                    else:
                        for i, k, ii, kk in T.Parallel(block_M // micro_size_x, block_K // pack_size_k, micro_size_x, pack_size_k):
                            A_shared[i, k, ii, kk] = A[by * block_M // micro_size_x + i, ko * block_K // pack_size_k + k, ii, kk]
                else:
                    if a_transposed:
                        T.copy(A[ko * block_K, by * block_M], A_shared)
                    else:
                        T.copy(A[by * block_M, ko * block_K], A_shared)

                # Load B into shared memory
                if b_g2l_load is False:
                    if b_preshuffle:
                        if b_transposed:
                            for j, k, jj, kk in T.Parallel(block_N // micro_size_y, block_K // pack_size_k, micro_size_y, pack_size_k):
                                B_shared[j, k, jj, kk] = B[bx * block_N // micro_size_y + j, ko * block_K // pack_size_k + k, jj, kk]
                        else:
                            for k, j, kk, jj in T.Parallel(block_K // pack_size_k, block_N // micro_size_y, pack_size_k, micro_size_y):
                                B_shared[k, j, kk, jj] = B[ko * block_K // pack_size_k + k, bx * block_N // micro_size_y + j, kk, jj]
                    else:
                        if b_transposed:
                            T.copy(B[bx * block_N, ko * block_K], B_shared)
                        else:
                            T.copy(B[ko * block_K, bx * block_N], B_shared)

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
    a_preshuffle=False,
    b_preshuffle=False,
    b_g2l_load=False,
    mfma_shape=None,
):
    matmul = tl_matmul(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        accum_dtype,
        a_transposed,
        b_transposed,
        k_pack,
        a_preshuffle,
        b_preshuffle,
        b_g2l_load,
        mfma_shape=mfma_shape,
    )
    kernel = tilelang.compile(matmul)
    assert kernel.get_kernel_source() is not None

    A_shape = (K, M) if a_transposed else (M, K)
    B_shape = (N, K) if b_transposed else (K, N)
    if in_dtype == T.int8:
        A = torch.randint(-128, 127, A_shape, device="cuda", dtype=torch.int8)
        B = torch.randint(-128, 127, B_shape, device="cuda", dtype=torch.int8)
    elif "float8" in str(in_dtype):
        A = torch.rand(A_shape, device="cuda", dtype=torch.float16).to(getattr(torch, in_dtype))
        B = torch.rand(B_shape, device="cuda", dtype=torch.float16).to(getattr(torch, in_dtype))
    else:
        A = torch.rand(A_shape, device="cuda", dtype=getattr(torch, in_dtype))
        B = torch.rand(B_shape, device="cuda", dtype=getattr(torch, in_dtype))

    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

    shuf_layout = (mfma_shape[0], mfma_shape[2]) if mfma_shape else (16, 32)
    A_in = shuffle_weight(A, layout=shuf_layout, k_pack=k_pack, is_transpose=not a_transposed) if a_preshuffle else A
    B_in = shuffle_weight(B, layout=shuf_layout, k_pack=k_pack, is_transpose=b_transposed) if b_preshuffle else B
    kernel(A_in, B_in, C)

    if a_transposed and b_transposed:
        ref_c = torch.matmul(A.T.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))
    elif a_transposed and not b_transposed:
        ref_c = torch.matmul(A.T.to(torch.float32), B.to(torch.float32)).to(getattr(torch, out_dtype))
    elif not a_transposed and b_transposed:
        ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))
    else:
        ref_c = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(getattr(torch, out_dtype))

    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "M, N, K, in_dtype, out_dtype, accum_dtype, a_transposed, b_transposed, k_pack, a_preshuffle, b_preshuffle, b_g2l_load",
    [
        # B-only preshuffle
        (256, 256, 512, T.int8, T.int32, T.int32, False, True, 1, False, True, False),
        (256, 256, 512, T.int8, T.int32, T.int32, False, False, 1, False, True, False),
        (256, 256, 512, T.int8, T.int32, T.int32, False, True, 2, False, True, False),
        (256, 256, 512, T.int8, T.int32, T.int32, False, False, 2, False, True, False),
        (256, 256, 512, determine_fp8_type(), T.float32, T.float32, False, True, 1, False, True, False),
        (256, 256, 512, determine_fp8_type(), T.float32, T.float32, False, False, 1, False, True, False),
        (256, 256, 512, determine_fp8_type(), T.float32, T.float32, False, True, 2, False, True, False),
        (256, 256, 512, determine_fp8_type(), T.float32, T.float32, False, False, 2, False, True, False),
        # No preshuffle
        (256, 256, 512, T.int8, T.int32, T.int32, False, True, 1, False, False, False),
        (256, 256, 512, determine_fp8_type(), T.float32, T.float32, False, True, 1, False, False, False),
        # A-only preshuffle
        (256, 256, 512, T.int8, T.int32, T.int32, False, True, 1, True, False, False),
        (256, 256, 512, T.int8, T.int32, T.int32, True, True, 1, True, False, False),
        (256, 256, 512, determine_fp8_type(), T.float32, T.float32, False, True, 1, True, False, False),
        (256, 256, 512, determine_fp8_type(), T.float32, T.float32, True, True, 1, True, False, False),
        # A+B preshuffle together (default 2x2 warp grid)
        (256, 256, 512, T.int8, T.int32, T.int32, False, True, 1, True, True, False),
        (256, 256, 512, determine_fp8_type(), T.float32, T.float32, False, True, 1, True, True, False),
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
    a_preshuffle,
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
        a_preshuffle=a_preshuffle,
        b_preshuffle=b_preshuffle,
        b_g2l_load=b_g2l_load,
    )


# ---- CDNA4 extended MFMA shapes: 16x16x64 and 32x32x32 for int8 ----
@pytest.mark.parametrize(
    "M, N, K, in_dtype, out_dtype, accum_dtype, b_transposed, k_pack, b_preshuffle, mfma_shape",
    [
        # v_mfma_i32_16x16x64_i8 — doubled K throughput (kp=1 only, micro_k=64)
        (256, 256, 512, T.int8, T.int32, T.int32, True, 1, False, (16, 16, 64)),
        (256, 256, 512, T.int8, T.int32, T.int32, True, 1, True, (16, 16, 64)),
        # v_mfma_i32_32x32x32_i8 — doubled MN throughput
        (256, 256, 512, T.int8, T.int32, T.int32, True, 1, False, (32, 32, 32)),
        (256, 256, 512, T.int8, T.int32, T.int32, True, 1, True, (32, 32, 32)),
    ],
)
@tilelang.testing.requires_rocm
def test_assert_tl_matmul_extended_mfma(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    b_transposed,
    k_pack,
    b_preshuffle,
    mfma_shape,
):
    assert_tl_matmul_correctness(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        accum_dtype=accum_dtype,
        b_transposed=b_transposed,
        k_pack=k_pack,
        b_preshuffle=b_preshuffle,
        mfma_shape=mfma_shape,
    )


if __name__ == "__main__":
    tilelang.testing.main()
