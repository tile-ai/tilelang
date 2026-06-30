import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.cuda.intrinsics.layout.mma_layout import mma_load_a_32x32_to_shared_16x64_layout
from tilelang.intrinsics import TensorCoreIntrinEmitterWithBlockScale, get_swizzle_layout
from tilelang.transform import simplify_prim_func


_FP4_E2M1_VALUES = (
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
def _make_nvf4_matmul_codegen_kernel(M, N, K, num_stages=2):
    assert K % 64 == 0
    in_dtype = T.float4_e2m1fn
    out_dtype = T.float32
    accum_dtype = T.float32

    micro_size_k = 64

    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = K
    shared_scope = "shared.dyn"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)
    B_shape = (N, K)
    SFA_shape = (M, K // micro_size_k)
    SFB_shape = (N, K // micro_size_k)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    SFA_shared_shape = (block_M, block_K // micro_size_k)
    SFB_shared_shape = (block_N, block_K // micro_size_k)

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        SFA: T.Tensor(SFA_shape, T.uint32),
        SFB: T.Tensor(SFB_shape, T.uint32),
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
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.use_swizzle(panel_size=10)

            for ko in T.Pipelined((K // block_K), num_stages=num_stages):
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for i, k in T.Parallel(block_M, block_K // micro_size_k):
                    SFA_shared[i, k] = SFA[by * block_M + i, ko * (block_K // micro_size_k) + k]

                for j, k in T.Parallel(block_N, block_K // micro_size_k):
                    SFB_shared[j, k] = SFB[bx * block_N + j, ko * (block_K // micro_size_k) + k]

                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    clear_accum=True,
                    k_start=ko * block_K,
                    sf_a_granularity_k=16,
                    sf_b_granularity_k=16,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _decode_rowmajor_fp4(packed, rows: int, cols: int):
    import torch

    u = packed.contiguous().view(torch.uint8)
    lut = torch.tensor(_FP4_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    out = torch.empty((rows, cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(u & 0x0F).long()]
    out[:, 1::2] = lut[((u >> 4) & 0x0F).long()]
    return out


def _make_packed_fp4_inputs(M: int, N: int, K: int, input_mode: str):
    import torch

    if input_mode == "constant":
        a = torch.full((M, K // 2), 0x22, device="cuda", dtype=torch.uint8).view(torch.int8)
        b = torch.full((N, K // 2), 0x22, device="cuda", dtype=torch.uint8).view(torch.int8)
    elif input_mode == "random":
        a = torch.randint(-128, 128, (M, K // 2), device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 128, (N, K // 2), device="cuda", dtype=torch.int8)
    elif input_mode == "a_random_b_alternating":
        a = torch.randint(-128, 128, (M, K // 2), device="cuda", dtype=torch.int8)
        b = torch.full((N, K // 2), 0x21, device="cuda", dtype=torch.uint8).view(torch.int8)
    elif input_mode == "a_constant_b_random":
        a = torch.full((M, K // 2), 0x22, device="cuda", dtype=torch.uint8).view(torch.int8)
        b = torch.randint(-128, 128, (N, K // 2), device="cuda", dtype=torch.int8)
    else:
        raise ValueError(f"Unsupported input_mode={input_mode!r}")
    return a, b


def _make_constant_scale_words(rows: int, K: int, byte: int = 0x38):
    import torch

    word = byte | (byte << 8) | (byte << 16) | (byte << 24)
    return torch.full((rows, K // 64), word, device="cuda", dtype=torch.uint32)


def _pack_scale_words(scale_bytes):
    import torch

    scale_i64 = scale_bytes.to(torch.int64).reshape(scale_bytes.shape[0], -1, 4)
    word = scale_i64[:, :, 0]
    word = word | (scale_i64[:, :, 1] << 8)
    word = word | (scale_i64[:, :, 2] << 16)
    word = word | (scale_i64[:, :, 3] << 24)
    return word.to(torch.uint32).contiguous()


def _make_varying_power_of_two_scale_words(rows: int, K: int):
    import torch

    scale_choices = torch.tensor([0x30, 0x38, 0x40], device="cuda", dtype=torch.uint8)
    row = torch.arange(rows, device="cuda", dtype=torch.int64)[:, None]
    col = torch.arange(K // 16, device="cuda", dtype=torch.int64)[None, :]
    scale_bytes = scale_choices[(row + 2 * col) % scale_choices.numel()]
    return _pack_scale_words(scale_bytes), scale_bytes


def _decode_ue4m3_scale_bytes(scale_bytes):
    import torch

    u = scale_bytes.to(torch.int32)
    exponent = (u >> 3) & 0x0F
    mantissa = u & 0x07
    normal = (1.0 + mantissa.to(torch.float32) / 8.0) * torch.pow(2.0, exponent.to(torch.float32) - 7.0)
    subnormal = (mantissa.to(torch.float32) / 8.0) * torch.pow(torch.tensor(2.0, device=scale_bytes.device), -6.0)
    return torch.where(exponent == 0, subnormal, normal)


def _reference_constant_scale_gemm(A, B, M: int, N: int, K: int):
    a_f32 = _decode_rowmajor_fp4(A, M, K)
    b_f32 = _decode_rowmajor_fp4(B, N, K)
    return a_f32 @ b_f32.T


def _reference_blockscaled_gemm(A, B, SFA, SFB, M: int, N: int, K: int):
    a_f32 = _decode_rowmajor_fp4(A, M, K)
    b_f32 = _decode_rowmajor_fp4(B, N, K)
    sfa = _decode_ue4m3_scale_bytes(SFA).repeat_interleave(16, dim=1)
    sfb = _decode_ue4m3_scale_bytes(SFB).repeat_interleave(16, dim=1)
    return (a_f32 * sfa) @ (b_f32 * sfb).T


def test_nvf4_mma_block_scale_fragment_layouts_match_cute():
    # CUTLASS/CuTe SM120 ALayout:
    # Layout<Shape<Shape<_4,_8>, Shape<_8,_2,_2>>,
    #        Stride<Stride<_128,_1>, Stride<_16,_8,_512>>>.
    seen = set()
    for thread_id in range(32):
        for local_id in range(32):
            coord = mma_load_a_32x32_to_shared_16x64_layout(thread_id, local_id)
            assert 0 <= coord[0] < 16
            assert 0 <= coord[1] < 64
            seen.add(coord)
    assert len(seen) == 16 * 64

    assert mma_load_a_32x32_to_shared_16x64_layout(0, 0) == (0, 0)
    assert mma_load_a_32x32_to_shared_16x64_layout(0, 8) == (8, 0)
    assert mma_load_a_32x32_to_shared_16x64_layout(31, 31) == (15, 63)


def test_nvf4_mma_block_scale_lane_scale_mapping_matches_cute():
    sfa_rows = [TensorCoreIntrinEmitterWithBlockScale._sfa_row_in_atom(tx) for tx in range(32)]
    sfb_cols = [TensorCoreIntrinEmitterWithBlockScale._sfb_col_in_atom(tx) for tx in range(32)]

    assert sfa_rows == [
        0,
        8,
        0,
        8,
        1,
        9,
        1,
        9,
        2,
        10,
        2,
        10,
        3,
        11,
        3,
        11,
        4,
        12,
        4,
        12,
        5,
        13,
        5,
        13,
        6,
        14,
        6,
        14,
        7,
        15,
        7,
        15,
    ]
    assert sfb_cols == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "mxf4nvf4", "scale_vec_size": 2, "stype": "ue4m3"},
        {"kind": "mxf4nvf4", "scale_vec_size": 4, "stype": "ue8m0"},
        {"kind": "mxf4", "scale_vec_size": 4, "stype": "ue4m3"},
    ],
)
def test_nvf4_mma_block_scale_rejects_unsupported_configs(kwargs):
    with pytest.raises(ValueError, match="Unsupported SM120 block-scale MMA config"):
        TensorCoreIntrinEmitterWithBlockScale(
            a_dtype=T.float4_e2m1fn,
            b_dtype=T.float4_e2m1fn,
            accum_dtype=T.float32,
            a_transposed=False,
            b_transposed=True,
            block_row_warps=2,
            block_col_warps=2,
            warp_row_tiles=32,
            warp_col_tiles=32,
            chunk=256,
            **kwargs,
        )


@pytest.mark.parametrize(
    "dtype_kwargs",
    [
        {"a_dtype": T.float16, "b_dtype": T.float4_e2m1fn, "accum_dtype": T.float32},
        {"a_dtype": T.float4_e2m1fn, "b_dtype": T.float16, "accum_dtype": T.float32},
        {"a_dtype": T.float4_e2m1fn, "b_dtype": T.float4_e2m1fn, "accum_dtype": T.float16},
    ],
)
def test_nvf4_mma_block_scale_rejects_incompatible_dtypes(dtype_kwargs):
    with pytest.raises(ValueError, match="mxf4nvf4 expects"):
        TensorCoreIntrinEmitterWithBlockScale(
            a_transposed=False,
            b_transposed=True,
            block_row_warps=2,
            block_col_warps=2,
            warp_row_tiles=32,
            warp_col_tiles=32,
            chunk=256,
            kind="mxf4nvf4",
            scale_vec_size=4,
            stype="ue4m3",
            **dtype_kwargs,
        )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@pytest.mark.parametrize("K", [64, 128, 256])
def test_nvf4_mma_block_scale_codegen(K):
    kernel = tilelang.compile(
        _make_nvf4_matmul_codegen_kernel(128, 128, K),
        target="cuda",
        out_idx=[4],
    )
    src = kernel.get_kernel_source()
    assert "#include <tl_templates/cuda/gemm.h>" in src
    assert "#include <tl_templates/cuda/instruction/mma_block_scale.h>" not in src
    assert "sm120_mma_sync_blockscaled" in src
    assert "SFA_shared" in src
    assert "SFB_shared" in src
    assert "scale_a_local" not in src
    assert "scale_b_local" not in src
    assert "SM120MmaBlockScaledKind::kMxf4nvf4" in src
    assert "SM120MmaScaleType::kUE4M3" in src
    fp4_tile_bytes = 128 * K // 2
    sf_tile_bytes = 128 * (K // 64) * 4
    assert f"void* B_shared = ((void*)((char*)buf_dyn_shmem + {fp4_tile_bytes}));" in src
    assert f"void* SFA_shared = ((void*)((char*)buf_dyn_shmem + {2 * fp4_tile_bytes}));" in src
    assert f"void* SFB_shared = ((void*)((char*)buf_dyn_shmem + {2 * fp4_tile_bytes + sf_tile_bytes}));" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
def test_nvf4_mma_block_scale_packed_smem_offsets():
    kernel = tilelang.compile(
        _make_nvf4_matmul_codegen_kernel(256, 256, 256, num_stages=3),
        target="cuda",
        out_idx=[4],
    )
    src = kernel.get_kernel_source()
    assert "void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));" in src
    assert "void* B_shared = ((void*)((char*)buf_dyn_shmem + 24576));" in src
    assert "void* SFA_shared = ((void*)((char*)buf_dyn_shmem + 49152));" in src
    assert "void* SFB_shared = ((void*)((char*)buf_dyn_shmem + 52224));" in src


def test_nvf4_mma_block_scale_packed_smem_non_alias_offset_units():
    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float4_e2m1fn),
        B: T.Tensor((16,), T.float4_e2m1fn),
    ):
        with T.Kernel(1, threads=32):
            a = T.alloc_shared((3,), T.float4_e2m1fn)
            b = T.alloc_shared((4,), T.float4_e2m1fn)
            a[0] = A[0]
            b[0] = A[1]
            B[0] = a[0]
            B[1] = b[0]

    mod = tvm.IRModule.from_expr(
        before.with_attr("global_symbol", "main").with_attr(
            "target",
            tvm.target.Target("webgpu"),
        )
    )
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.MergeSharedMemoryAllocations()(mod)

    src = mod.script()
    # Three logical FP4 values use two physical bytes. After 16-byte alignment,
    # the next buffer starts at logical FP4 offset 32, not byte offset 16.
    assert "b[32]" in src
    assert "b[16]" not in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@pytest.mark.parametrize(
    "K,input_mode",
    [
        (64, "random"),
        (128, "a_random_b_alternating"),
        (256, "a_constant_b_random"),
        (256, "random"),
    ],
)
def test_nvf4_mma_block_scale_constant_scale_correctness(K, input_mode):
    import torch

    torch.manual_seed(0)
    M = N = 128
    kernel = tilelang.compile(
        _make_nvf4_matmul_codegen_kernel(M, N, K),
        target="cuda",
        out_idx=[4],
    )

    A, B = _make_packed_fp4_inputs(M, N, K, input_mode)
    SFA = _make_constant_scale_words(M, K)
    SFB = _make_constant_scale_words(N, K)

    C = kernel(A, B, SFA, SFB)
    ref = _reference_constant_scale_gemm(A, B, M, N, K)
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
def test_nvf4_mma_block_scale_varying_scale_correctness():
    import torch

    torch.manual_seed(0)
    M = N = 128
    K = 128
    kernel = tilelang.compile(
        _make_nvf4_matmul_codegen_kernel(M, N, K),
        target="cuda",
        out_idx=[4],
    )

    A, B = _make_packed_fp4_inputs(M, N, K, "random")
    SFA, sfa_bytes = _make_varying_power_of_two_scale_words(M, K)
    SFB, sfb_bytes = _make_varying_power_of_two_scale_words(N, K)

    C = kernel(A, B, SFA, SFB)
    ref = _reference_blockscaled_gemm(A, B, sfa_bytes, sfb_bytes, M, N, K)
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    tilelang.testing.main()
