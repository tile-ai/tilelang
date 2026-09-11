"""Compare TileLang's SM120 MXFP4 block-scaled GEMM against CUTLASS.

Run from the repository root:

    python -m maint.gemm.gemm_sm120.correctness_evaluation_mxf4_vs_cutlass

The scale factors are UE8M0 (power-of-two), so both engines perform the
same exact fp32 arithmetic and the comparison is bitwise.
"""

import os
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang.cuda.language.intrinsics import TensorCoreIntrinEmitterSM120, make_mma_swizzle_layout
from tilelang.transform import simplify_prim_func


_SF_VEC_SIZE = 32
_WORD_SPAN = _SF_VEC_SIZE * 4  # K elements covered by one uint32 scale word


@simplify_prim_func
def _make_tilelang_mxf4_kernel(m: int, n: int, k: int, micro_k: int):
    in_dtype = T.float4_e2m1fn
    out_dtype = T.float32
    accum_dtype = T.float32

    micro_size_x = 16
    micro_size_y = 16
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = k
    shared_scope = "shared.dyn"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    SFA_shared_shape = (block_M, block_K // _WORD_SPAN)
    SFB_shared_shape = (block_N, block_K // _WORD_SPAN)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_k) // warp_size
    local_size_b = (micro_size_y * micro_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    mma_emitter = TensorCoreIntrinEmitterSM120(
        is_blockscaled=True,
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
        kind="mxf4nvf4",
        scale_vec_size=micro_k // _SF_VEC_SIZE,
        stype="ue8m0",
    )

    @T.prim_func
    def main(
        A: T.Tensor((m, k), in_dtype),
        B: T.Tensor((n, k), in_dtype),
        SFA: T.Tensor((m, k // _WORD_SPAN), T.uint32),
        SFB: T.Tensor((n, k // _WORD_SPAN), T.uint32),
        C: T.Tensor((m, n), out_dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M), threads=threads) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            SFA_shared = T.alloc_shared(SFA_shared_shape, T.uint32, scope=shared_scope)
            SFB_shared = T.alloc_shared(SFB_shared_shape, T.uint32, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_mma_swizzle_layout(A_shared),
                    B_shared: make_mma_swizzle_layout(B_shared),
                }
            )
            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((k // block_K), num_stages=2):
                for i, k_inner in T.Parallel(block_M, block_K):
                    A_shared[i, k_inner] = A[by * block_M + i, ko * block_K + k_inner]

                for j, k_inner in T.Parallel(block_N, block_K):
                    B_shared[j, k_inner] = B[bx * block_N + j, ko * block_K + k_inner]

                for i, k_inner in T.Parallel(block_M, block_K // _WORD_SPAN):
                    SFA_shared[i, k_inner] = SFA[by * block_M + i, ko * (block_K // _WORD_SPAN) + k_inner]

                for j, k_inner in T.Parallel(block_N, block_K // _WORD_SPAN):
                    SFB_shared[j, k_inner] = SFB[bx * block_N + j, ko * (block_K // _WORD_SPAN) + k_inner]

                for ki in T.serial(0, (block_K // micro_k)):
                    mma_emitter.ldmatrix_a(A_local, A_shared, ki)
                    mma_emitter.ldmatrix_b(B_local, B_shared, ki)
                    mma_emitter.mma(
                        A_local,
                        B_local,
                        C_local,
                        ki,
                        SFA_buf=SFA_shared,
                        SFB_buf=SFB_shared,
                        k_start=0,
                        sf_a_granularity_k=_SF_VEC_SIZE,
                        sf_b_granularity_k=_SF_VEC_SIZE,
                    )

            mma_emitter.stmatrix(C_local, C_shared)

            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]

    return main


def _pack_tilelang_sf_u32(sf_bytes):
    assert sf_bytes.dtype == torch.uint8
    mn, sf_blocks = sf_bytes.shape
    assert sf_blocks % 4 == 0
    words = sf_bytes.reshape(mn, sf_blocks // 4, 4).to(torch.int64)
    packed = words[:, :, 0] | (words[:, :, 1] << 8) | (words[:, :, 2] << 16) | (words[:, :, 3] << 24)
    return packed.to(torch.uint32).contiguous()


def _pack_cutlass_sf_bytes(sf_bytes):
    """Pack logical (MN, K/32) UE8M0 bytes into CUTLASS Sm1xxBlockScaledConfig<32> order.

    Blk_MN=128 / Blk_SF=4 are scale-vector-size independent, so the byte
    layout matches the NVFP4 packer with the K/32 index in place of K/16.
    """
    assert sf_bytes.dtype == torch.uint8
    mn, sf_blocks = sf_bytes.shape
    assert mn == 128
    assert sf_blocks % 4 == 0

    out = torch.empty((mn * sf_blocks,), device=sf_bytes.device, dtype=torch.uint8)
    for sf_idx in range(sf_blocks):
        k_word_group = sf_idx // 4
        byte_in_word = sf_idx % 4
        for row in range(mn):
            offset = k_word_group * mn * 4 + (row % 32) * 16 + (row // 32) * 4 + byte_in_word
            out[offset] = sf_bytes[row, sf_idx]
    return out.contiguous()


def _decode_rowmajor_fp4(packed, rows: int, cols: int):
    fp4_e2m1_values = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    )
    u = packed.contiguous().view(torch.uint8)
    lut = torch.tensor(fp4_e2m1_values, device=packed.device, dtype=torch.float32)
    out = torch.empty((rows, cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(u & 0x0F).long()]
    out[:, 1::2] = lut[((u >> 4) & 0x0F).long()]
    return out


def _decode_ue8m0(sf_bytes):
    return torch.pow(2.0, (sf_bytes.to(torch.int32) - 127).to(torch.float32))


def _blockscaled_mxf4_reference(a, b, sfa_logical, sfb_logical):
    m, packed_k = a.shape
    n, packed_b_k = b.shape
    assert packed_b_k == packed_k
    k = packed_k * 2
    a_f32 = _decode_rowmajor_fp4(a, m, k)
    b_f32 = _decode_rowmajor_fp4(b, n, k)
    sfa = _decode_ue8m0(sfa_logical).repeat_interleave(_SF_VEC_SIZE, dim=1)
    sfb = _decode_ue8m0(sfb_logical).repeat_interleave(_SF_VEC_SIZE, dim=1)
    return (a_f32 * sfa) @ (b_f32 * sfb).T


def _build_cutlass_extension():
    from torch.utils.cpp_extension import load

    source_path = Path(__file__).resolve().with_name("cutlass_mxf4_ref.cu")
    repo_root = source_path.parents[3]
    cutlass_root_env = os.environ.get("CUTLASS_ROOT")
    cutlass_root = Path(cutlass_root_env) if cutlass_root_env else repo_root / "3rdparty" / "cutlass"
    include_paths = [
        cutlass_root / "include",
        cutlass_root / "tools" / "util" / "include",
    ]
    if not all(path.exists() for path in include_paths):
        raise RuntimeError(
            "CUTLASS headers were not found. Set CUTLASS_ROOT to a CUTLASS checkout "
            "or populate the repository's 3rdparty/cutlass submodule."
        )

    return load(
        name="tilelang_cutlass_mxf4_ref",
        sources=[str(source_path)],
        extra_include_paths=[str(path) for path in include_paths],
        extra_cuda_cflags=[
            "-std=c++20",
            "-arch=sm_120a",
            "--expt-relaxed-constexpr",
            "-DCUTLASS_ARCH_MMA_SM120_SUPPORTED",
        ],
        extra_cflags=["-std=c++20"],
        verbose=True,
    )


def run_compare() -> None:
    m = 128
    n = 128
    k = 256
    micro_k = 64

    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"

    input_mode = os.environ.get("MXF4_INPUT_MODE", "random")
    if input_mode == "constant":
        input_byte = int(os.environ.get("MXF4_INPUT_BYTE", "0x22"), 0)
        a = torch.full((m, k // 2), input_byte, device="cuda", dtype=torch.uint8).view(torch.int8)
        b = torch.full((n, k // 2), input_byte, device="cuda", dtype=torch.uint8).view(torch.int8)
    elif input_mode == "random":
        a = torch.randint(-128, 128, (m, k // 2), device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 128, (n, k // 2), device="cuda", dtype=torch.int8)
    else:
        raise ValueError(f"Unsupported MXF4_INPUT_MODE={input_mode!r}")

    scale_mode = os.environ.get("MXF4_SCALE_MODE", "varying")
    if scale_mode == "constant":
        sfa_logical = torch.full((m, k // _SF_VEC_SIZE), 0x7F, device="cuda", dtype=torch.uint8)
        sfb_logical = torch.full((n, k // _SF_VEC_SIZE), 0x7F, device="cuda", dtype=torch.uint8)
    elif scale_mode == "varying":
        # Power-of-two exponents around 1.0, varying by row and K group so a
        # byte-pair (parity) mix-up in the 2X path changes the result. The
        # window is kept narrow (2^-8..2^7): the bitwise assertion below rests
        # on both engines accumulating K in order with an fp32 accumulator, and
        # a modest magnitude spread keeps summation rounding tame on top of
        # that shared order.
        row = torch.arange(m, device="cuda", dtype=torch.int32).reshape(m, 1)
        col = torch.arange(k // _SF_VEC_SIZE, device="cuda", dtype=torch.int32).reshape(1, k // _SF_VEC_SIZE)
        sfa_logical = (0x77 + ((row * 3 + col * 5) % 16)).to(torch.uint8)
        sfb_logical = (0x77 + ((row * 7 + col * 11) % 16)).to(torch.uint8)
    else:
        raise ValueError(f"Unsupported MXF4_SCALE_MODE={scale_mode!r}")

    sfa_tl = _pack_tilelang_sf_u32(sfa_logical)
    sfb_tl = _pack_tilelang_sf_u32(sfb_logical)
    sfa_cutlass = _pack_cutlass_sf_bytes(sfa_logical)
    sfb_cutlass = _pack_cutlass_sf_bytes(sfb_logical)

    kernel = tilelang.compile(
        _make_tilelang_mxf4_kernel(m, n, k, micro_k),
        target="cuda",
        out_idx=[4],
    )
    c_tl = kernel(a, b, sfa_tl, sfb_tl)

    cutlass_ref = _build_cutlass_extension()
    c_in = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    c_cutlass = torch.empty((m, n), device="cuda", dtype=torch.float32)
    cutlass_ref.cutlass_mxf4_gemm_128x128x256(a, b, sfa_cutlass, sfb_cutlass, c_in, c_cutlass)

    torch.cuda.synchronize()
    ref = _blockscaled_mxf4_reference(a, b, sfa_logical, sfb_logical)
    print("scale_mode:", scale_mode)
    print("input_mode:", input_mode)
    print("max_abs_diff:", (c_tl - c_cutlass).abs().max().item())
    print("max_abs_diff_tilelang_ref:", (c_tl - ref).abs().max().item())
    print("max_abs_diff_cutlass_ref:", (c_cutlass - ref).abs().max().item())
    if os.environ.get("MXF4_SKIP_ASSERT", "0") == "1":
        return
    torch.testing.assert_close(c_tl, c_cutlass, rtol=0.0, atol=0.0)
    print("TileLang MXFP4 block-scale output matches CUTLASS exactly.")


if __name__ == "__main__":
    run_compare()
