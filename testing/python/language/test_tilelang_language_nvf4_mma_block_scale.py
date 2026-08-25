import importlib.util
import sys
from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.cuda.intrinsics.layout.mma_layout import mma_load_a_32x32_to_shared_16x64_layout
from tilelang.cuda.intrinsics.macro.mma_sm120_macro_generator import SM120BlockScaleTile
from tilelang.cuda.language.intrinsics import TensorCoreIntrinEmitterSM120, get_swizzle_layout
from examples.dequantize_gemm.quantize.nvfp4 import (
    blockscaled_chunk_kmajor_word_offset,
    swizzle_blockscaled_chunk_kmajor_scale_words,
)
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
    return TensorCoreIntrinEmitterSM120(
        is_blockscaled=True,
        kind="mxf4nvf4",
        scale_vec_size=4,
        stype="ue4m3",
        **kwargs,
    )


def _make_sm120_fulltile_contract(
    chunk=256,
    *,
    block_row_warps=2,
    block_col_warps=2,
    warp_row_tiles=64,
    warp_col_tiles=64,
):
    emitter = _make_blockscale_emitter(
        a_dtype=T.float4_e2m1fn,
        b_dtype=T.float4_e2m1fn,
        accum_dtype=T.float32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        reduce_k=1,
        num_elems_per_byte=2,
    )
    return SM120BlockScaleTile.from_emitter(
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

    from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter as MMAIntrinEmitter

    base = list(inspect.signature(MMAIntrinEmitter.mma).parameters.values())
    override = list(inspect.signature(TensorCoreIntrinEmitterSM120.mma).parameters.values())
    assert [p.name for p in override[: len(base)]] == [p.name for p in base]
    for extra in override[len(base) :]:
        assert extra.kind == inspect.Parameter.KEYWORD_ONLY, extra.name


def test_default_tensor_core_intrin_emitter_stays_arch_neutral():
    from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter as MMAIntrinEmitter
    from tilelang.cuda.op.gemm.gemm_mma import GemmMMA
    from tilelang.cuda.op.gemm.gemm_mma_sm120 import (
        GEMM_INST_MMA_BLOCK_SCALED,
        GemmMMASm120BlockScaled,
    )
    from tilelang.cuda.language.intrinsics import TensorCoreIntrinEmitter
    from tilelang.tileop.gemm.registry import resolve_gemm_impl

    assert TensorCoreIntrinEmitter is MMAIntrinEmitter
    assert GemmMMA.intrin_emitter_cls is MMAIntrinEmitter
    assert issubclass(TensorCoreIntrinEmitterSM120, MMAIntrinEmitter)
    assert GemmMMASm120BlockScaled.intrin_emitter_cls is TensorCoreIntrinEmitterSM120
    assert (
        resolve_gemm_impl(
            GEMM_INST_MMA_BLOCK_SCALED,
            tvm.target.Target({"kind": "cuda", "arch": "sm_120"}),
        )
        is GemmMMASm120BlockScaled
    )


def test_generic_mma_gemm_has_no_blockscaled_lowering_branch():
    source = (Path(__file__).resolve().parents[3] / "tilelang/cuda/op/gemm/gemm_mma.py").read_text()
    assert "is_blockscaled" not in source
    assert "TensorCoreIntrinEmitterBlockScaled" not in source


def test_sm120_mma_blockscaled_strategy_helpers_are_not_public_api():
    # NVFP4-specific staging helpers stay out of the general T.* surface; the
    # scale-tile addressing lives in examples.dequantize_gemm.quantize.nvfp4.
    assert not hasattr(T, "copy_ue4m3_scale_tile")
    assert not hasattr(T, "ue4m3_scale_tile_source_coords")
    assert not hasattr(T, "sm120_mma_blockscaled")
    assert not hasattr(T, "sm120_mma_blockscaled_kblock_fulltile")
    assert not hasattr(T, "sm120_mma_blockscaled_kblock_fulltile_ab_owner_wide")
    assert not hasattr(
        T,
        "sm120_mma_blockscaled_kblock_fulltile_afull_bpanel_owner_wide",
    )
    assert not hasattr(T, "sm120_mma_blockscaled_fulltile")
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


def _make_varying_ue8m0_scale_words(rows: int, K: int, granularity: int):
    # Adjacent granularity groups get different power-of-two scales so a
    # byte-pair (parity) mix-up in the 2X path changes the result.
    import torch

    scale_choices = torch.tensor([0x7E, 0x7F, 0x80], device="cuda", dtype=torch.uint8)
    row = torch.arange(rows, device="cuda", dtype=torch.int64)[:, None]
    col = torch.arange(K // granularity, device="cuda", dtype=torch.int64)[None, :]
    scale_bytes = scale_choices[(row + 2 * col) % scale_choices.numel()]
    return _pack_scale_words(scale_bytes), scale_bytes


def _decode_ue8m0_scale_bytes(scale_bytes):
    import torch

    return torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0)


def _reference_ue8m0_blockscaled_gemm(A, B, sfa_bytes, sfb_bytes, M: int, N: int, K: int, granularity: int):
    a_f32 = _decode_rowmajor_fp4(A, M, K)
    b_f32 = _decode_rowmajor_fp4(B, N, K)
    sfa = _decode_ue8m0_scale_bytes(sfa_bytes).repeat_interleave(granularity, dim=1)
    sfb = _decode_ue8m0_scale_bytes(sfb_bytes).repeat_interleave(granularity, dim=1)
    return (a_f32 * sfa) @ (b_f32 * sfb).T


@simplify_prim_func
def _make_mxf4_matmul_codegen_kernel(
    M,
    N,
    K,
    num_stages=2,
    *,
    sf_granularity_k=32,
    scale_dtype="ue8m0",
    sf_layout=None,
    block_row_warps=2,
    block_col_warps=2,
    warp_row_tiles=32,
    warp_col_tiles=32,
):
    assert K % 64 == 0
    in_dtype = T.float4_e2m1fn
    out_dtype = T.float32
    accum_dtype = T.float32

    word_span = sf_granularity_k * 4

    chunk = K
    shared_scope = "shared.dyn"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)
    B_shape = (N, K)
    SFA_shape = (M, K // word_span)
    SFB_shape = (N, K // word_span)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    SFA_shared_shape = (block_M, block_K // word_span)
    SFB_shared_shape = (block_N, block_K // word_span)

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

                for i, k in T.Parallel(block_M, block_K // word_span):
                    SFA_shared[i, k] = SFA[by * block_M + i, ko * (block_K // word_span) + k]

                for j, k in T.Parallel(block_N, block_K // word_span):
                    SFB_shared[j, k] = SFB[bx * block_N + j, ko * (block_K // word_span) + k]

                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    clear_accum=True,
                    k_start=ko * block_K,
                    sf_a_granularity_k=sf_granularity_k,
                    sf_b_granularity_k=sf_granularity_k,
                    sf_layout=sf_layout,
                    scale_dtype=scale_dtype,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


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
    sfa_rows = [TensorCoreIntrinEmitterSM120._sfa_row_in_atom(tx) for tx in range(32)]
    sfb_cols = [TensorCoreIntrinEmitterSM120._sfb_col_in_atom(tx) for tx in range(32)]

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
        {"kind": "mxf4", "scale_vec_size": 4, "stype": "ue4m3"},
    ],
)
def test_nvf4_mma_block_scale_rejects_unsupported_configs(kwargs):
    with pytest.raises(ValueError, match="Unsupported SM120 block-scale MMA config"):
        TensorCoreIntrinEmitterSM120(
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


def test_sm120_mma_and_ldscale_reject_chunk_kmajor_scale_layout():
    # The chunk-kmajor scale layout is served exclusively by
    # mma_blockscaled_fulltile; the per-word mma()/ldscale() paths only
    # understand rowmajor scale addressing.
    emitter = TensorCoreIntrinEmitterSM120(
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
    )
    with pytest.raises(ValueError, match="mma_blockscaled_fulltile"):
        emitter.mma(None, None, None, 0, SFA_buf=object(), SFB_buf=object(), sf_layout="blockscaled_chunk_kmajor")
    with pytest.raises(ValueError, match="mma_blockscaled_fulltile"):
        emitter.ldscale(None, None, None, None, None, sf_layout="blockscaled_chunk_kmajor")


def _make_ue8m0_emitter(scale_vec_size, chunk):
    return TensorCoreIntrinEmitterSM120(
        is_blockscaled=True,
        a_dtype=T.float4_e2m1fn,
        b_dtype=T.float4_e2m1fn,
        accum_dtype=T.float32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=64,
        warp_col_tiles=64,
        chunk=chunk,
        kind="mxf4nvf4",
        scale_vec_size=scale_vec_size,
        stype="ue8m0",
    )


def test_mxf4_2x_ue8m0_emitter_contract():
    emitter = _make_ue8m0_emitter(2, 256)
    assert (emitter.kind, emitter.scale_vec_size, emitter.stype) == ("mxf4nvf4", 2, "ue8m0")
    assert emitter.sf_vec_size == 32

    # One uint32 word covers K=128: word column is the k64-atom pair index,
    # the byte id selects the 2-byte half by atom parity.
    word_ks = [emitter._scale_word_k(0, ki, 32) for ki in range(4)]
    byte_ids = [emitter._scale_byte_id(0, ki) for ki in range(4)]
    analyzer = tvm.arith.Analyzer()
    assert [int(analyzer.simplify(w)) for w in word_ks] == [0, 0, 1, 1]
    assert [int(analyzer.simplify(tvm.tirx.const(0, "int32") + b)) for b in byte_ids] == [0, 2, 0, 2]

    # 2X packs K=128 per word; narrower block_K cannot be staged.
    for bad_chunk in (64, 192):
        with pytest.raises(ValueError, match="multiple of"):
            _make_ue8m0_emitter(2, bad_chunk)


def test_mxf4nvf4_4x_ue8m0_emitter_contract():
    emitter = _make_ue8m0_emitter(4, 256)
    assert (emitter.kind, emitter.scale_vec_size, emitter.stype) == ("mxf4nvf4", 4, "ue8m0")
    assert emitter.sf_vec_size == 16
    # 4X keeps the whole-word consumption: byte id is the literal 0.
    assert emitter._scale_byte_id(0, 3) == 0


def test_sm120_fulltile_contract_tracks_2x_words_per_stage():
    contract = SM120BlockScaleTile.from_emitter(
        _make_ue8m0_emitter(2, 256),
        sf_layout="blockscaled_chunk_kmajor",
    )
    assert (contract.kblocks, contract.sf_vec_size) == (4, 32)
    assert contract.words_per_stage == 2
    # Word columns range over words_per_stage, not kblocks.
    with pytest.raises(ValueError, match="word column"):
        contract._scale_word_offset(0, 2, 128)


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
        TensorCoreIntrinEmitterSM120(
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
    assert (contract.sfa_words, contract.sfb_words) == (2, 2)
    assert (contract.warp_issues, contract.warpgroup_issues) == (32, 128)


def test_sm120_fulltile_package_contract_supports_k128_path():
    contract = _make_sm120_fulltile_contract(chunk=128)

    assert (contract.tile_m, contract.tile_n, contract.tile_k) == (128, 128, 128)
    assert (contract.kblocks, contract.micro_size_k) == (2, 64)
    assert contract.package_pingpong_lifecycle() == (
        ("copy", 0, 0),
        ("copy", 1, 1),
        ("gemm", 0, 0),
        ("gemm", 1, 1),
    )


def test_sm120_fulltile_package_contract_derives_arbitrary_reasonable_block_shape():
    contract = _make_sm120_fulltile_contract(chunk=192)

    assert (contract.tile_m, contract.tile_n, contract.tile_k) == (128, 128, 192)
    assert (contract.warp_rows, contract.warp_cols, contract.kblocks) == (4, 4, 3)
    assert (contract.sfa_words, contract.sfb_words) == (2, 2)
    assert (contract.warp_issues, contract.warpgroup_issues) == (32, 128)


def test_sm120_fulltile_package_contract_rejects_non_atom_multiple_tiles():
    # The kmajor scale source stacks 128-row BlockScaledBasicChunk atoms, so a
    # staged tile must cover whole atoms; 64x192 was silently mis-addressed
    # before the packer-atom fix.
    with pytest.raises(ValueError, match="multiples of 128"):
        _make_sm120_fulltile_contract(chunk=192, warp_row_tiles=32, warp_col_tiles=96)


def test_sm120_fulltile_contract_scale_words_follow_packer_atoms_for_tile_m_256():
    # tile_m=256 stacks two 128-row packer atoms. The flat word offset must
    # follow the host packer (atom index = row // 128, fixed 4-group split
    # inside the atom), not a tile_rows-wide group split.
    contract = _make_sm120_fulltile_contract(warp_row_tiles=128)

    assert (contract.tile_m, contract.tile_n) == (256, 128)
    assert (contract.sfa_words, contract.sfb_words) == (4, 2)
    assert contract.words_per_stage == contract.kblocks

    for lane in (0, 1, 2, 17, 31):
        for warp_m in range(contract.block_row_warps):
            for warp_n in range(contract.block_col_warps):
                sfa_rows, sfb_rows = contract.compact_selector_scale_rows(lane, warp_m, warp_n)
                for kblock in range(contract.kblocks):
                    sfa_offsets, sfb_offsets = contract.compact_selector_scale_word_offsets(lane, warp_m, warp_n, kblock)
                    for row, flat in zip(sfa_rows, sfa_offsets):
                        atom_base = (row // 128) * (contract.words_per_stage * 128)
                        assert flat == atom_base + _oracle_blockscaled_chunk_kmajor_flat_word(row % 128, kblock)
                    for row, flat in zip(sfb_rows, sfb_offsets):
                        # tile_n=128 stays inside a single atom.
                        assert flat == _oracle_blockscaled_chunk_kmajor_flat_word(row, kblock)


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


@pytest.mark.parametrize("kblocks", [1, 2, 3, 4, 5])
def test_sm120_fulltile_package_contract_pingpong_lifecycle_is_data_ready(kblocks):
    contract = _make_sm120_fulltile_contract(chunk=kblocks * 64)

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

    assert gemmed_kblocks == list(range(kblocks))


def test_sm120_fulltile_package_contract_describes_omma_sf_issue_schedule():
    contract = _make_sm120_fulltile_contract()
    schedule = contract.omma_sf_issue_schedule_per_warp()

    assert len(schedule) == contract.warp_issues
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


def test_sm120_fulltile_package_contract_rejects_odd_warp_atom_grid():
    emitter = _make_blockscale_emitter(
        a_dtype=T.float4_e2m1fn,
        b_dtype=T.float4_e2m1fn,
        accum_dtype=T.float32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=48,
        warp_col_tiles=64,
        chunk=256,
        reduce_k=1,
        num_elems_per_byte=2,
    )

    with pytest.raises(ValueError, match="positive even MMA atom grid"):
        SM120BlockScaleTile.from_emitter(
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
    assert "#include <tl_templates/cuda/instruction/mma_block_scale.h>" in src
    assert "#include <tl_templates/cuda/gemm_sm120.h>" not in src
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
@pytest.mark.parametrize(
    "M, N, K, warp_row_tiles, warp_col_tiles",
    [
        (128, 128, 64, 64, 64),
        (128, 128, 128, 64, 64),
        (128, 128, 192, 64, 64),
        (128, 128, 256, 64, 64),
    ],
)
def test_nvf4_mma_block_scale_fulltile_is_frontend_lowered(M, N, K, warp_row_tiles, warp_col_tiles):
    kernel = tilelang.compile(
        _make_nvf4_matmul_codegen_kernel(
            M,
            N,
            K,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            sf_layout="blockscaled_chunk_kmajor",
        ),
        target="cuda",
        out_idx=[4],
    )

    src = kernel.get_kernel_source()
    assert "sm120_mma_blockscaled_fulltile" not in src
    assert "tl::ptx_ldmatrix_x4" in src
    assert "tl::sm120_mma_sync_blockscaled<" in src


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


def test_mma_gemm_blockscaled_rejects_mismatched_scale_mode():
    with pytest.raises(ValueError, match="Unsupported SM120 block-scale mode"):
        _make_mxf4_matmul_codegen_kernel(128, 128, 256, sf_granularity_k=32, scale_dtype="ue4m3")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("K", [128, 256])
def test_mxf4_mma_block_scale_codegen(K):
    kernel = tilelang.compile(
        _make_mxf4_matmul_codegen_kernel(128, 128, K),
        target="cuda",
        out_idx=[4],
    )
    src = kernel.get_kernel_source()
    assert "tl::sm120_mma_sync_blockscaled<" in src
    assert "SM120MmaScaleType::kUE8M0" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize(
    "K,sf_layout",
    [
        (128, None),
        (256, None),
        (128, "blockscaled_chunk_kmajor"),
        (256, "blockscaled_chunk_kmajor"),
    ],
)
def test_mxf4_mma_block_scale_varying_scale_correctness(K, sf_layout):
    import torch

    torch.manual_seed(0)
    M = N = 128
    # kmajor staging requires whole 128-row packer atoms per tile.
    tiles = 64 if sf_layout else 32
    kernel = tilelang.compile(
        _make_mxf4_matmul_codegen_kernel(
            M,
            N,
            K,
            sf_layout=sf_layout,
            warp_row_tiles=tiles,
            warp_col_tiles=tiles,
        ),
        target="cuda",
        out_idx=[4],
    )

    A, B = _make_packed_fp4_inputs(M, N, K, "random")
    SFA, sfa_bytes = _make_varying_ue8m0_scale_words(M, K, 32)
    SFB, sfb_bytes = _make_varying_ue8m0_scale_words(N, K, 32)
    if sf_layout:
        SFA = swizzle_blockscaled_chunk_kmajor_scale_words(SFA, block_words=K // 128)
        SFB = swizzle_blockscaled_chunk_kmajor_scale_words(SFB, block_words=K // 128)

    C = kernel(A, B, SFA, SFB)
    ref = _reference_ue8m0_blockscaled_gemm(A, B, sfa_bytes, sfb_bytes, M, N, K, 32)
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_mxf4nvf4_4x_ue8m0_varying_scale_correctness():
    # granularity 16 + ue8m0: the CUDA 13.1 scale_vec::4X.ue8m0 instruction.
    import torch

    torch.manual_seed(0)
    M = N = 128
    K = 256
    kernel = tilelang.compile(
        _make_mxf4_matmul_codegen_kernel(M, N, K, sf_granularity_k=16),
        target="cuda",
        out_idx=[4],
    )

    A, B = _make_packed_fp4_inputs(M, N, K, "random")
    SFA, sfa_bytes = _make_varying_ue8m0_scale_words(M, K, 16)
    SFB, sfb_bytes = _make_varying_ue8m0_scale_words(N, K, 16)

    C = kernel(A, B, SFA, SFB)
    ref = _reference_ue8m0_blockscaled_gemm(A, B, sfa_bytes, sfb_bytes, M, N, K, 16)
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Example tail-tile behavior (moved from test_tilelang_sm120_nvfp4_example_cli).
# ---------------------------------------------------------------------------


def _load_sm120_example(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    example = repo_root / "examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py"
    monkeypatch.setattr(sys, "argv", [str(example)])
    spec = importlib.util.spec_from_file_location("sm120_nvfp4_blockscaled_gemm_example", example)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_sm120_nvfp4_example_kernel_handles_mn_tail_tiles(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")

    from examples.dequantize_gemm.quantize import swizzle_blockscaled_chunk_kmajor_scale_words

    module = _load_sm120_example(monkeypatch)
    for M, N, K in [(257, 384, 512), (130, 128, 256), (128, 136, 256)]:
        kernel = module.sm120_nvfp4_blockscaled_gemm(M, N, K)

        A = module._make_packed_fp4(M, K, seed=3)
        B = module._make_packed_fp4(N, K, seed=4)
        SFA_semantic = module._make_binary_scale_words(M, K, seed=5)
        SFB_semantic = module._make_binary_scale_words(N, K, seed=6)
        SFA = swizzle_blockscaled_chunk_kmajor_scale_words(SFA_semantic).reshape(-1, 4)
        SFB = swizzle_blockscaled_chunk_kmajor_scale_words(SFB_semantic).reshape(-1, 4)
        assert SFA.shape[0] % 128 == 0
        assert SFB.shape[0] % 128 == 0

        C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        kernel(A, B, SFA, SFB, C)
        torch.cuda.synchronize()
        module._verify(A, B, SFA_semantic, SFB_semantic, C, torch.bfloat16)


def test_sm120_nvfp4_example_kernel_rejects_unsupported_tails(monkeypatch):
    import pytest

    module = _load_sm120_example(monkeypatch)
    # simultaneous M and N tails hit a known copy-lowering boundary bug
    with pytest.raises(ValueError, match="simultaneous M and N tail"):
        module.sm120_nvfp4_blockscaled_gemm(257, 136, 512)
    # bf16 output rows must stay 16-byte aligned
    with pytest.raises(AssertionError, match="multiple of 8"):
        module.sm120_nvfp4_blockscaled_gemm(128, 130, 256)


if __name__ == "__main__":
    tilelang.testing.main()
