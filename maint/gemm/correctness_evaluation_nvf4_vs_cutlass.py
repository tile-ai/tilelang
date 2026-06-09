import os
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.intrinsics import TensorCoreIntrinEmitterWithBlockScale, get_swizzle_layout
from tilelang.transform import simplify_prim_func


M = 128
N = 128
K = 256
MICRO_K = 64
SF_VEC_SIZE = 16
SCALE_WORD_K = MICRO_K // SF_VEC_SIZE
FP4_E2M1_VALUES = (
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


def _make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape
    can_swizzle = shape[-1] * tvm.DataType(dtype).bits % 512 == 0
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


@simplify_prim_func
def _make_tilelang_nvf4_kernel():
    in_dtype = T.float4_e2m1fn
    out_dtype = T.float32
    accum_dtype = T.float32

    micro_size_x = 16
    micro_size_y = 16
    micro_size_k = MICRO_K

    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = K
    shared_scope = "shared.dyn"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    SFA_shared_shape = (block_M, block_K // micro_size_k)
    SFB_shared_shape = (block_N, block_K // micro_size_k)
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

    mma_emitter = TensorCoreIntrinEmitterWithBlockScale(
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
        scale_vec_size=4,
        stype="ue4m3",
    )

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((M, K // MICRO_K), T.uint32),
        SFB: T.Tensor((N, K // MICRO_K), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
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
                    A_shared: _make_swizzle_layout(A_shared),
                    B_shared: _make_swizzle_layout(B_shared),
                }
            )
            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=2):
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for i, k in T.Parallel(block_M, block_K // micro_size_k):
                    SFA_shared[i, k] = SFA[by * block_M + i, ko * (block_K // micro_size_k) + k]

                for j, k in T.Parallel(block_N, block_K // micro_size_k):
                    SFB_shared[j, k] = SFB[bx * block_N + j, ko * (block_K // micro_size_k) + k]

                for ki in T.serial(0, (block_K // micro_size_k)):
                    mma_emitter.ldmatrix_a(A_local, A_shared, ki)
                    mma_emitter.ldmatrix_b(B_local, B_shared, ki)
                    mma_emitter.mma(
                        A_local,
                        B_local,
                        C_local,
                        SFA_shared,
                        SFB_shared,
                        ki=ki,
                        k_start=0,
                        sf_a_granularity_k=SF_VEC_SIZE,
                        sf_b_granularity_k=SF_VEC_SIZE,
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
    import torch

    assert sf_bytes.dtype == torch.uint8
    mn, sf_blocks = sf_bytes.shape
    assert sf_blocks % SCALE_WORD_K == 0
    words = sf_bytes.reshape(mn, sf_blocks // SCALE_WORD_K, SCALE_WORD_K).to(torch.int64)
    packed = words[:, :, 0] | (words[:, :, 1] << 8) | (words[:, :, 2] << 16) | (words[:, :, 3] << 24)
    return packed.to(torch.uint32).contiguous()


def _pack_cutlass_sf_bytes(sf_bytes):
    """Pack logical (MN, K/16) UE4M3 bytes into CUTLASS Sm1xxBlockScaledConfig<16> order."""
    import torch

    assert sf_bytes.dtype == torch.uint8
    mn, sf_blocks = sf_bytes.shape
    assert mn == 128
    assert sf_blocks == K // SF_VEC_SIZE

    out = torch.empty((mn * sf_blocks,), device=sf_bytes.device, dtype=torch.uint8)
    for sf_idx in range(sf_blocks):
        k64 = sf_idx // SCALE_WORD_K
        k_word = sf_idx % SCALE_WORD_K
        for row in range(mn):
            offset = k64 * 512 + (row % 32) * 16 + (row // 32) * 4 + k_word
            out[offset] = sf_bytes[row, sf_idx]
    return out.contiguous()


def _pack_cutlass_colmajor_fp4_from_rowmajor(packed, rows: int, cols: int):
    """Repack logical row-major FP4 bytes into CUTLASS ColumnMajor packed storage."""
    import torch

    assert packed.dtype == torch.int8
    logical_bytes = packed.contiguous().view(torch.uint8)
    lo = logical_bytes & 0x0F
    hi = (logical_bytes >> 4) & 0x0F
    logical = torch.empty((rows, cols), device=packed.device, dtype=torch.uint8)
    logical[:, 0::2] = lo
    logical[:, 1::2] = hi

    col_major = logical.T.contiguous().reshape(-1)
    packed_col_major = (col_major[0::2] | (col_major[1::2] << 4)).contiguous()
    return packed_col_major.view(torch.int8)


def _swap_fp4_nibbles(packed):
    import torch

    u = packed.contiguous().view(torch.uint8)
    return (((u & 0x0F) << 4) | ((u >> 4) & 0x0F)).contiguous().view(torch.int8)


def _decode_rowmajor_fp4(packed, rows: int, cols: int):
    import torch

    u = packed.contiguous().view(torch.uint8)
    lut = torch.tensor(FP4_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    out = torch.empty((rows, cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(u & 0x0F).long()]
    out[:, 1::2] = lut[((u >> 4) & 0x0F).long()]
    return out


def _build_cutlass_extension():
    from torch.utils.cpp_extension import load

    repo = Path(__file__).resolve().parents[2]
    cutlass_root_env = os.environ.get("CUTLASS_ROOT")
    cutlass_root = Path(cutlass_root_env) if cutlass_root_env else repo / "3rdparty" / "cutlass"

    return load(
        name="tilelang_cutlass_nvf4_ref",
        sources=[str(repo / "maint" / "gemm" / "cutlass_nvf4_ref.cu")],
        extra_include_paths=[
            str(cutlass_root / "include"),
            str(cutlass_root / "tools" / "util" / "include"),
        ],
        extra_cuda_cflags=[
            "-std=c++20",
            "-arch=sm_120a",
            "--expt-relaxed-constexpr",
            "-DCUTLASS_ARCH_MMA_SM120_SUPPORTED",
        ],
        extra_cflags=["-std=c++20"],
        verbose=True,
    )


def run_compare():
    import torch

    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"

    input_mode = os.environ.get("NVF4_INPUT_MODE", "random")
    if input_mode == "constant":
        byte = int(os.environ.get("NVF4_INPUT_BYTE", "0x22"), 0)
        a = torch.full((M, K // 2), byte, device="cuda", dtype=torch.uint8).view(torch.int8)
        b = torch.full((N, K // 2), byte, device="cuda", dtype=torch.uint8).view(torch.int8)
    elif input_mode == "a_random_b_constant":
        byte = int(os.environ.get("NVF4_INPUT_BYTE", "0x22"), 0)
        a = torch.randint(-128, 128, (M, K // 2), device="cuda", dtype=torch.int8)
        b = torch.full((N, K // 2), byte, device="cuda", dtype=torch.uint8).view(torch.int8)
    elif input_mode == "a_constant_b_random":
        byte = int(os.environ.get("NVF4_INPUT_BYTE", "0x22"), 0)
        a = torch.full((M, K // 2), byte, device="cuda", dtype=torch.uint8).view(torch.int8)
        b = torch.randint(-128, 128, (N, K // 2), device="cuda", dtype=torch.int8)
    elif input_mode == "random":
        a = torch.randint(-128, 128, (M, K // 2), device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 128, (N, K // 2), device="cuda", dtype=torch.int8)
    else:
        raise ValueError(f"Unsupported NVF4_INPUT_MODE={input_mode!r}")

    scale_mode = os.environ.get("NVF4_SCALE_MODE", "varying")
    if scale_mode == "constant":
        sfa_logical = torch.full((M, K // SF_VEC_SIZE), 0x38, device="cuda", dtype=torch.uint8)
        sfb_logical = torch.full((N, K // SF_VEC_SIZE), 0x38, device="cuda", dtype=torch.uint8)
    elif scale_mode in {"varying", "sfa_varying", "sfb_varying", "k_varying", "mn_varying"}:
        # Avoid UE4M3 NaN encodings while still varying scales by row and K block.
        sfa_logical = torch.empty((M, K // SF_VEC_SIZE), device="cuda", dtype=torch.uint8)
        sfb_logical = torch.empty((N, K // SF_VEC_SIZE), device="cuda", dtype=torch.uint8)
        row = torch.arange(M, device="cuda", dtype=torch.int32).reshape(M, 1)
        col = torch.arange(K // SF_VEC_SIZE, device="cuda", dtype=torch.int32).reshape(1, K // SF_VEC_SIZE)
        sfa_logical.copy_((0x20 + ((row * 3 + col * 5) % 32)).to(torch.uint8))
        sfb_logical.copy_((0x20 + ((row * 7 + col * 11) % 32)).to(torch.uint8))
        if scale_mode == "k_varying":
            sfa_logical.copy_((0x20 + ((col * 5) % 32)).expand_as(sfa_logical).to(torch.uint8))
            sfb_logical.copy_((0x20 + ((col * 11) % 32)).expand_as(sfb_logical).to(torch.uint8))
        elif scale_mode == "mn_varying":
            sfa_logical.copy_((0x20 + ((row * 3) % 32)).expand_as(sfa_logical).to(torch.uint8))
            sfb_logical.copy_((0x20 + ((row * 7) % 32)).expand_as(sfb_logical).to(torch.uint8))
        if scale_mode == "sfa_varying":
            sfb_logical.fill_(0x38)
        elif scale_mode == "sfb_varying":
            sfa_logical.fill_(0x38)
    else:
        raise ValueError(f"Unsupported NVF4_SCALE_MODE={scale_mode!r}")

    sfa_tl = _pack_tilelang_sf_u32(sfa_logical)
    sfb_tl = _pack_tilelang_sf_u32(sfb_logical)
    sfa_cutlass = _pack_cutlass_sf_bytes(sfa_logical)
    sfb_cutlass = _pack_cutlass_sf_bytes(sfb_logical)

    a_tl = _swap_fp4_nibbles(a) if os.environ.get("NVF4_SWAP_A_FOR_TILELANG", "0") == "1" else a
    b_tl = _swap_fp4_nibbles(b) if os.environ.get("NVF4_SWAP_B_FOR_TILELANG", "0") == "1" else b

    kernel = tilelang.compile(_make_tilelang_nvf4_kernel(), target="cuda", out_idx=[4])
    c_tl = kernel(a_tl, b_tl, sfa_tl, sfb_tl)

    cutlass_ref = _build_cutlass_extension()
    if os.environ.get("NVF4_CUTLASS_REPACK_B", "0") == "1":
        b_cutlass = _pack_cutlass_colmajor_fp4_from_rowmajor(b, N, K)
    else:
        b_cutlass = b
    c_in = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    c_cutlass = torch.empty((M, N), device="cuda", dtype=torch.float32)
    cutlass_ref.cutlass_nvf4_gemm_128x128x256(a, b_cutlass, sfa_cutlass, sfb_cutlass, c_in, c_cutlass)

    torch.cuda.synchronize()
    diff = (c_tl - c_cutlass).abs()
    diff_t = (c_tl - c_cutlass.T).abs()
    ref = _decode_rowmajor_fp4(a, M, K) @ _decode_rowmajor_fp4(b, N, K).T
    diff_tl_ref = (c_tl - ref).abs()
    diff_cutlass_ref = (c_cutlass - ref).abs()
    print("scale_mode:", scale_mode)
    print("input_mode:", input_mode)
    print("max_abs_diff:", diff.max().item())
    print("mean_abs_diff:", diff.mean().item())
    print("max_abs_diff_transposed:", diff_t.max().item())
    print("mean_abs_diff_transposed:", diff_t.mean().item())
    print("max_abs_diff_tilelang_ref:", diff_tl_ref.max().item())
    print("mean_abs_diff_tilelang_ref:", diff_tl_ref.mean().item())
    print("max_abs_diff_cutlass_ref:", diff_cutlass_ref.max().item())
    print("mean_abs_diff_cutlass_ref:", diff_cutlass_ref.mean().item())
    if os.environ.get("NVF4_SKIP_ASSERT", "0") == "1":
        return
    torch.testing.assert_close(c_tl, c_cutlass, rtol=0.0, atol=0.0)
    print("TileLang NVF4 block-scale output matches CUTLASS exactly.")


if __name__ == "__main__":
    run_compare()
