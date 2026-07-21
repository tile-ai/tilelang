import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.cuda.intrinsics.layout.mma_layout import mma_load_a_32x32_to_shared_16x64_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import SM120BlockScaledFullTilePackageContract
from tilelang.intrinsics import TensorCoreIntrinEmitter, get_swizzle_layout
from tilelang.quantize.nvfp4 import blockscaled_chunk_kmajor_word_offset
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


def _make_blockscale_emitter(**kwargs):
    return TensorCoreIntrinEmitter(
        is_blockscaled=True,
        kind="mxf4nvf4",
        scale_vec_size=4,
        stype="ue4m3",
        **kwargs,
    )


def _make_sm120_fulltile_contract():
    emitter = _make_blockscale_emitter(
        a_dtype=T.float4_e2m1fn,
        b_dtype=T.float4_e2m1fn,
        accum_dtype=T.float32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=64,
        warp_col_tiles=64,
        chunk=256,
        reduce_k=1,
        num_elems_per_byte=2,
    )
    return SM120BlockScaledFullTilePackageContract.for_package_pingpong(
        emitter,
        sf_layout="blockscaled_chunk_kmajor",
    )


def _oracle_blockscaled_chunk_kmajor_flat_word(row: int, kblock: int) -> int:
    physical_row, physical_word = blockscaled_chunk_kmajor_word_offset(row, kblock)
    return physical_row * 4 + physical_word


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


def test_tensor_core_intrin_emitter_mma_keeps_base_positional_signature():
    """The block-scale extension must not shift the base positional signature.

    The non-blockscaled gemm lowering calls ``emitter.mma(A, B, C, ki)``
    positionally, so every scale-related parameter has to stay keyword-only.
    """
    import inspect

    from tilelang.cuda.intrinsics.macro.mma_macro_generator import _TensorCoreIntrinEmitterBase

    base = list(inspect.signature(_TensorCoreIntrinEmitterBase.mma).parameters.values())
    override = list(inspect.signature(TensorCoreIntrinEmitter.mma).parameters.values())
    assert [p.name for p in override[: len(base)]] == [p.name for p in base]
    for extra in override[len(base) :]:
        assert extra.kind == inspect.Parameter.KEYWORD_ONLY, extra.name


def test_copy_ue4m3_scale_tile_rejects_non_uint32_scales():
    with pytest.raises(ValueError, match="uint32"):

        @T.prim_func
        def main(SF: T.Tensor((128, 8), T.int32), Out: T.Tensor((1,), T.int32)):
            with T.Kernel(1, threads=128):
                SF_shared = T.alloc_shared((128, 4), T.int32)
                T.copy_ue4m3_scale_tile(SF, SF_shared, 0, 0)


def test_sm120_mma_blockscaled_strategy_helpers_are_not_public_api():
    assert not hasattr(T, "sm120_mma_blockscaled")
    assert not hasattr(T, "sm120_mma_blockscaled_kblock_fulltile")
    assert not hasattr(T, "sm120_mma_blockscaled_kblock_fulltile_ab_owner_wide")
    assert not hasattr(
        T,
        "sm120_mma_blockscaled_kblock_fulltile_afull_bpanel_owner_wide",
    )
    assert not hasattr(T, "sm120_mma_blockscaled_kblock_fulltile_package_pingpong")
    assert not hasattr(T, "sm120_mma_blockscaled_cute_consumer_bridge")


@simplify_prim_func
def _make_nvf4_matmul_codegen_kernel(
    M,
    N,
    K,
    num_stages=2,
    *,
    block_row_warps=2,
    block_col_warps=2,
    warp_row_tiles=32,
    warp_col_tiles=32,
    sf_layout=None,
):
    assert K % 64 == 0
    in_dtype = T.float4_e2m1fn
    out_dtype = T.float32
    accum_dtype = T.float32

    micro_size_k = 64

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
                    sf_layout=sf_layout,
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
    sfa_rows = [TensorCoreIntrinEmitter._sfa_row_in_atom(tx) for tx in range(32)]
    sfb_cols = [TensorCoreIntrinEmitter._sfb_col_in_atom(tx) for tx in range(32)]

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
        TensorCoreIntrinEmitter(
            is_blockscaled=True,
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
        TensorCoreIntrinEmitter(
            is_blockscaled=True,
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


def test_sm120_fulltile_package_contract_matches_current_pingpong_path():
    contract = _make_sm120_fulltile_contract()

    assert (contract.tile_m, contract.tile_n, contract.tile_k) == (128, 128, 256)
    assert (contract.block_row_warps, contract.block_col_warps) == (2, 2)
    assert (contract.warp_rows, contract.warp_cols) == (4, 4)
    assert (contract.warp_row_tiles, contract.warp_col_tiles) == (64, 64)
    assert (contract.kblocks, contract.micro_size_k) == (4, 64)
    assert (contract.scale_package_words_sfa, contract.scale_package_words_sfb) == (2, 2)
    assert (contract.issue_count_per_warp, contract.issue_count_per_warpgroup) == (32, 128)


def test_sm120_fulltile_package_contract_describes_compact_selector_copy_view():
    contract = _make_sm120_fulltile_contract()

    assert contract.compact_selector_scale_rows(lane=0, warp_m=0, warp_n=0) == ((0, 32), (0, 32))
    assert contract.compact_selector_scale_rows(lane=1, warp_m=0, warp_n=0) == ((8, 40), (8, 40))
    assert contract.compact_selector_scale_rows(lane=2, warp_m=1, warp_n=1) == ((80, 112), (80, 112))
    assert contract.compact_selector_scale_rows(lane=31, warp_m=1, warp_n=1) == ((95, 127), (95, 127))


@pytest.mark.parametrize("lane", [0, 1, 2, 17, 31])
@pytest.mark.parametrize("warp_m", [0, 1])
@pytest.mark.parametrize("warp_n", [0, 1])
@pytest.mark.parametrize("kblock", [0, 2, 3])
def test_sm120_fulltile_package_contract_word_offsets_match_source_layout_oracle(lane, warp_m, warp_n, kblock):
    contract = _make_sm120_fulltile_contract()

    sfa_rows, sfb_rows = contract.compact_selector_scale_rows(lane=lane, warp_m=warp_m, warp_n=warp_n)
    expected_sfa = tuple(_oracle_blockscaled_chunk_kmajor_flat_word(row, kblock) for row in sfa_rows)
    expected_sfb = tuple(_oracle_blockscaled_chunk_kmajor_flat_word(row, kblock) for row in sfb_rows)

    assert contract.compact_selector_scale_word_offsets(lane=lane, warp_m=warp_m, warp_n=warp_n, kblock=kblock) == (
        expected_sfa,
        expected_sfb,
    )


def test_sm120_fulltile_package_contract_describes_pingpong_lifecycle():
    contract = _make_sm120_fulltile_contract()

    assert contract.package_pingpong_lifecycle() == (
        ("copy", 0, 0),
        ("copy", 1, 1),
        ("gemm", 0, 0),
        ("copy", 0, 2),
        ("gemm", 1, 1),
        ("copy", 1, 3),
        ("gemm", 0, 2),
        ("gemm", 1, 3),
    )


def test_sm120_fulltile_package_contract_pingpong_lifecycle_is_data_ready():
    contract = _make_sm120_fulltile_contract()

    package_kblock = {}
    gemmed_kblocks = []
    for op, package_id, kblock in contract.package_pingpong_lifecycle():
        if op == "copy":
            package_kblock[package_id] = kblock
        elif op == "gemm":
            assert package_kblock[package_id] == kblock
            gemmed_kblocks.append(kblock)
        else:
            raise AssertionError(f"unexpected package lifecycle op: {op}")

    assert gemmed_kblocks == [0, 1, 2, 3]


def test_sm120_fulltile_package_contract_describes_omma_sf_issue_schedule():
    contract = _make_sm120_fulltile_contract()
    schedule = contract.omma_sf_issue_schedule_per_warp()

    assert len(schedule) == contract.issue_count_per_warp
    assert schedule[0] == (0, 0, 0, 0, 0, 0, 0)
    assert schedule[1] == (0, 0, 1, 0, 0, 0, 1)
    assert schedule[2] == (0, 1, 0, 0, 0, 0, 2)
    assert schedule[3] == (0, 1, 1, 0, 0, 0, 3)
    assert schedule[16] == (2, 0, 0, 1, 0, 0, 0)
    assert schedule[-1] == (3, 3, 1, 1, 1, 1, 3)

    for mma_i, mma_j, n8_half, sfa_word, sfb_word, scale_a_tid, scale_b_tid in schedule:
        assert sfa_word == (0 if mma_i < 2 else 1)
        assert sfb_word == (0 if mma_j < 2 else 1)
        assert scale_a_tid == (mma_i & 1)
        assert scale_b_tid == (mma_j & 1) * 2 + n8_half

    assert sorted({issue[5] for issue in schedule}) == [0, 1]
    assert sorted({issue[6] for issue in schedule}) == [0, 1, 2, 3]
    assert sum(1 for issue in schedule if issue[3] == 0) == 16
    assert sum(1 for issue in schedule if issue[3] == 1) == 16
    assert sum(1 for issue in schedule if issue[4] == 0) == 16
    assert sum(1 for issue in schedule if issue[4] == 1) == 16


def test_sm120_fulltile_package_contract_rejects_shape_drift():
    emitter = _make_blockscale_emitter(
        a_dtype=T.float4_e2m1fn,
        b_dtype=T.float4_e2m1fn,
        accum_dtype=T.float32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=4,
        warp_row_tiles=64,
        warp_col_tiles=64,
        chunk=256,
        reduce_k=1,
        num_elems_per_byte=2,
    )

    with pytest.raises(ValueError, match="128x128x256"):
        SM120BlockScaledFullTilePackageContract.for_package_pingpong(
            emitter,
            sf_layout="blockscaled_chunk_kmajor",
        )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("K", [64, 128, 256])
def test_nvf4_mma_block_scale_codegen(K):
    kernel = tilelang.compile(
        _make_nvf4_matmul_codegen_kernel(128, 128, K),
        target="cuda",
        out_idx=[4],
    )
    src = kernel.get_kernel_source()
    assert "#include <tl_templates/cuda/gemm_sm120.h>" in src
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
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_nvf4_mma_block_scale_rejects_legacy_cutlass_128x4_layout_alias():
    with pytest.raises(ValueError, match="Unsupported SM120 scale layout: cutlass_128x4"):
        tilelang.compile(
            _make_nvf4_matmul_codegen_kernel(
                128,
                128,
                256,
                warp_row_tiles=64,
                warp_col_tiles=64,
                sf_layout="cutlass_128x4",
            ),
            target="cuda",
            out_idx=[4],
        )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_nvf4_mma_block_scale_package_pingpong_contract_lowers_fulltile():
    kernel = tilelang.compile(
        _make_nvf4_matmul_codegen_kernel(
            128,
            128,
            256,
            warp_row_tiles=64,
            warp_col_tiles=64,
            sf_layout="blockscaled_chunk_kmajor",
        ),
        target="cuda",
        out_idx=[4],
    )

    src = kernel.get_kernel_source()
    assert "sm120_mma_blockscaled_kblock_fulltile_package_pingpong" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
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
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
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
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
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
