import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.cuda.intrinsics.layout.mma_layout import mma_load_a_32x32_to_shared_16x64_layout
from tilelang.intrinsics import TensorCoreIntrinEmitter, get_swizzle_layout
from tilelang.quantize import (
    pack_blockscaled_chunk_kmajor_scale_bytes,
    quantize_bf16_to_nvfp4_blockscaled,
    swizzle_blockscaled_chunk_kmajor_scale_words,
    unswizzle_blockscaled_chunk_kmajor_scale_words,
)
from tilelang.quantize.nvfp4 import (
    blockscaled_chunk_kmajor_word_offset,
    decode_packed_fp4_e2m1,
    decode_ue4m3_scale_bytes,
    encode_fp4_e2m1_values,
    encode_ue4m3_scale_bytes,
    pack_nvfp4_scale_bytes,
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
    return TensorCoreIntrinEmitter(
        is_blockscaled=True,
        kind="mxf4nvf4",
        scale_vec_size=4,
        stype="ue4m3",
        **kwargs,
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


def test_sm120_mma_blockscaled_strategy_helpers_are_not_public_api():
    # NVFP4-specific staging helpers stay out of the general T.* surface; the
    # scale-tile addressing lives in tilelang.quantize.nvfp4.
    assert not hasattr(T, "copy_ue4m3_scale_tile")
    assert not hasattr(T, "ue4m3_scale_tile_source_coords")
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

    from tilelang.quantize import swizzle_blockscaled_chunk_kmajor_scale_words

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


# ---------------------------------------------------------------------------
# Scale layout / packer contract (moved from
# testing/python/quantize/test_tilelang_quantize_nvfp4_scale_layout).
# These pin the blockscaled_chunk_kmajor packing against independent oracles,
# the CuTeDSL canonical SF byte layout, and the maint TileLang quantizer.
# ---------------------------------------------------------------------------


def _load_maint_quantizer():
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "maint/gemm/gemm_sm120/tilelang_nvfp4_quantizer.py"
    spec = importlib.util.spec_from_file_location("tilelang_nvfp4_quantizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tilelang_quantize_bf16_to_nvfp4_blockscaled


def _pack_semantic_scale_words(scale_bytes):
    scale_i64 = scale_bytes.to(torch.int64).reshape(scale_bytes.shape[0], scale_bytes.shape[1] // 4, 4)
    words = scale_i64[:, :, 0]
    words = words | (scale_i64[:, :, 1] << 8)
    words = words | (scale_i64[:, :, 2] << 16)
    words = words | (scale_i64[:, :, 3] << 24)
    return words.to(torch.uint32).contiguous()


def _expected_blockscaled_chunk_kmajor_byte_location(row: int, k16_idx: int, k64_cols: int = 4) -> tuple[int, int, int]:
    k64_word = k16_idx // 4
    byte_lane = k16_idx % 4
    row_block = row // 128
    row_in_block = row % 128
    flat_word = row_block * 128 * k64_cols + k64_word * 128 + (row_in_block % 32) * 4 + (row_in_block // 32)
    physical_row = flat_word // k64_cols
    physical_word = flat_word % k64_cols
    return physical_row, physical_word, byte_lane


def _packed_byte(packed, row: int, word: int, byte_lane: int) -> int:
    return (int(packed[row, word].item()) >> (8 * byte_lane)) & 0xFF


def _unpack_with_oracle(packed, rows: int, k16_cols: int):
    out = torch.empty((rows, k16_cols), dtype=torch.uint8)
    k64_cols = k16_cols // 4
    for row in range(rows):
        for k16_idx in range(k16_cols):
            physical_row, physical_word, byte_lane = _expected_blockscaled_chunk_kmajor_byte_location(row, k16_idx, k64_cols)
            out[row, k16_idx] = _packed_byte(packed, physical_row, physical_word, byte_lane)
    return out


def test_blockscaled_chunk_kmajor_word_offset_fixed_cases():
    expected = {
        (0, 0): (0, 0),
        (0, 1): (32, 0),
        (0, 2): (64, 0),
        (0, 3): (96, 0),
        (31, 0): (31, 0),
        (32, 0): (0, 1),
        (63, 1): (63, 1),
        (64, 2): (64, 2),
        (96, 3): (96, 3),
        (127, 3): (127, 3),
    }
    for (row, k64_word), physical in expected.items():
        assert blockscaled_chunk_kmajor_word_offset(row, k64_word) == physical


def test_pack_blockscaled_chunk_kmajor_scale_bytes_fixed_byte_offsets():
    rows = 256
    k16_cols = 32
    scale_bytes = torch.zeros((rows, k16_cols), dtype=torch.uint8)
    cases = [
        (0, 0, 0x11),
        (32, 0, 0x22),
        (64, 8, 0x33),
        (96, 12, 0x44),
        (127, 15, 0x55),
        (128, 16, 0x66),
        (159, 31, 0x77),
        (255, 27, 0x88),
    ]
    for row, k16_idx, value in cases:
        scale_bytes[row, k16_idx] = value

    packed = pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes)

    assert packed.shape == (rows, k16_cols // 4)
    assert packed.dtype == torch.uint32
    for row, k16_idx, value in cases:
        physical_row, physical_word, byte_lane = _expected_blockscaled_chunk_kmajor_byte_location(row, k16_idx, packed.shape[1])
        assert _packed_byte(packed, physical_row, physical_word, byte_lane) == value


def test_pack_blockscaled_chunk_kmajor_scale_bytes_random_binary_512x32_matches_oracle():
    rows = 512
    k16_cols = 512 // 16
    generator = torch.Generator(device="cpu").manual_seed(17)
    scale_bytes = torch.randint(0, 2, (rows, k16_cols), generator=generator, dtype=torch.uint8) * 0x38

    packed = pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes)

    assert packed.shape == (rows, k16_cols // 4)
    assert packed.dtype == torch.uint32
    assert torch.equal(_unpack_with_oracle(packed, rows, k16_cols), scale_bytes)


def test_pack_blockscaled_chunk_kmajor_scale_bytes_matches_cutedsl_blocked_sf_layout():
    """Byte-level cross-compatibility with the CuTeDSL/CUTLASS canonical SF layout.

    CuTeDSL builds SFA/SFB with ``blockscaled_utils.tile_atom_to_shape_SF``:
    atom ``((32,4),(16,4)):((16,4),(0,1))`` tiled with order ``(2,1,3)``, e.g.
    for ``(MN=256, K=512, L=1)`` the layout prints as
    ``(((32,4),2),((16,4),8),(1,1)):(((16,4),4096),((0,1),512),(0,0))``.
    The packed uint32 tensor must carry exactly those bytes so one buffer can
    feed both the TileLang SM120 path and a CuTeDSL NVFP4 blockscaled GEMM
    (``tl_words.view(torch.uint8)`` / ``sf_u8.view(torch.uint32)`` are
    zero-copy bridges between the two views).
    """
    for rows, k in ((128, 256), (256, 512), (384, 1024)):
        k16_cols = k // 16
        rest_k = k16_cols // 4
        generator = torch.Generator(device="cpu").manual_seed(rows + k)
        scale_bytes = torch.randint(0, 256, (rows, k16_cols), generator=generator, dtype=torch.uint8)

        packed_bytes = pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes).view(torch.uint8).reshape(-1)

        m = torch.arange(rows).unsqueeze(1)
        k16 = torch.arange(k16_cols).unsqueeze(0)
        cutedsl_offset = (m % 32) * 16 + ((m // 32) % 4) * 4 + (m // 128) * (512 * rest_k) + (k16 % 4) + (k16 // 4) * 512
        assert torch.equal(packed_bytes[cutedsl_offset.reshape(-1)].reshape(rows, k16_cols), scale_bytes)


def test_pack_blockscaled_chunk_kmajor_scale_bytes_matches_word_swizzle():
    rows = 512
    k16_cols = 512 // 16
    scale_bytes = torch.arange(rows * k16_cols, dtype=torch.uint8).reshape(rows, k16_cols)

    semantic_words = _pack_semantic_scale_words(scale_bytes)
    assert torch.equal(pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes), swizzle_blockscaled_chunk_kmajor_scale_words(semantic_words))


def test_pack_nvfp4_scale_bytes_default_matches_blockscaled_chunk_kmajor_layout():
    rows = 512
    k16_cols = 512 // 16
    scale_bytes = torch.arange(rows * k16_cols, dtype=torch.uint8).reshape(rows, k16_cols)

    expected = pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes)
    actual = pack_nvfp4_scale_bytes(scale_bytes)

    assert torch.equal(actual, expected)


def test_blockscaled_chunk_kmajor_scale_packer_rejects_invalid_shapes():
    # rows that are not a multiple of 128 are zero-padded, not rejected
    padded = pack_blockscaled_chunk_kmajor_scale_bytes(torch.zeros((127, 16), dtype=torch.uint8))
    assert padded.shape == (128, 4)

    with pytest.raises(ValueError, match="K/16 columns multiple of 16"):
        pack_blockscaled_chunk_kmajor_scale_bytes(torch.zeros((128, 12), dtype=torch.uint8))

    with pytest.raises(TypeError, match="torch.uint8"):
        pack_blockscaled_chunk_kmajor_scale_bytes(torch.zeros((128, 16), dtype=torch.int32))


def test_encode_fp4_e2m1_values_and_pack_order():
    values = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0]], dtype=torch.float32)
    codes = encode_fp4_e2m1_values(values)
    assert torch.equal(codes, torch.tensor([[0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x9, 0xA]], dtype=torch.uint8))

    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.int8)
    assert torch.equal(decode_packed_fp4_e2m1(packed), values)


def test_encode_ue4m3_scale_bytes_known_values():
    values = torch.tensor([0.0, 2.0**-9, 2.0**-6, 1.0, 2.0, 448.0], dtype=torch.float32)
    encoded = encode_ue4m3_scale_bytes(values, rounding="nearest")
    assert torch.equal(encoded, torch.tensor([0x00, 0x01, 0x08, 0x38, 0x40, 0x7E], dtype=torch.uint8))
    torch.testing.assert_close(decode_ue4m3_scale_bytes(encoded), values)
    assert torch.isnan(decode_ue4m3_scale_bytes(torch.tensor([0x7F], dtype=torch.uint8))).all()


def test_quantize_nvfp4_blockscaled_bf16_activation_contract():
    rows = 128
    cols = 256
    x = torch.zeros((rows, cols), dtype=torch.bfloat16)
    pattern = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0])
    x[0, :16] = pattern.to(torch.bfloat16)

    packed_fp4, packed_scales, scale_bytes = quantize_bf16_to_nvfp4_blockscaled(x, return_scale_bytes=True)

    assert packed_fp4.shape == (rows, cols // 2)
    assert packed_fp4.dtype == torch.int8
    assert packed_scales.shape == (rows, cols // 64)
    assert packed_scales.dtype == torch.uint32
    assert scale_bytes.shape == (rows, cols // 16)
    assert scale_bytes.dtype == torch.uint8
    assert scale_bytes[0, 0].item() == 0x38
    assert torch.equal(packed_scales, pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes))

    decoded = decode_packed_fp4_e2m1(packed_fp4) * decode_ue4m3_scale_bytes(scale_bytes).repeat_interleave(16, dim=1)
    torch.testing.assert_close(decoded[0, :16], pattern, rtol=0.0, atol=0.0)


def test_quantize_nvfp4_blockscaled_random_bf16_has_bounded_error():
    rows = 128
    cols = 256
    generator = torch.Generator(device="cpu").manual_seed(19)
    x = (torch.randn((rows, cols), generator=generator, dtype=torch.float32) * 2.0).to(torch.bfloat16)

    packed_fp4, scale_source, scale_bytes = quantize_bf16_to_nvfp4_blockscaled(x, return_scale_bytes=True)
    decoded = decode_packed_fp4_e2m1(packed_fp4) * decode_ue4m3_scale_bytes(scale_bytes).repeat_interleave(16, dim=1)

    scale = decode_ue4m3_scale_bytes(scale_bytes).repeat_interleave(16, dim=1)
    error = (decoded - x.to(torch.float32)).abs()
    assert torch.isfinite(decoded).all()
    assert torch.all(error <= scale + 1e-6)
    assert torch.equal(scale_source, pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes))


def test_quantize_nvfp4_blockscaled_explicit_layout_matches_default():
    rows = 128
    cols = 256
    generator = torch.Generator(device="cpu").manual_seed(23)
    x = (torch.randn((rows, cols), generator=generator, dtype=torch.float32) * 2.0).to(torch.bfloat16)

    default = quantize_bf16_to_nvfp4_blockscaled(x)
    explicit = quantize_bf16_to_nvfp4_blockscaled(x, scale_layout="blockscaled_chunk_kmajor")

    assert torch.equal(explicit[0], default[0])
    assert torch.equal(explicit[1], default[1])


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize("rows, cols", [(128, 256), (256, 512)])
def test_tilelang_quantize_nvfp4_blockscaled_matches_reference_layout_and_error_bound(rows, cols):
    generator = torch.Generator(device="cuda").manual_seed(rows + cols)
    x = (torch.randn((rows, cols), generator=generator, device="cuda", dtype=torch.float32) * 2.0).to(torch.bfloat16)

    tilelang_quantize = _load_maint_quantizer()
    packed_tl, scale_source_tl = tilelang_quantize(x)
    _, scale_source_ref, scale_bytes_ref = quantize_bf16_to_nvfp4_blockscaled(x, return_scale_bytes=True)

    assert packed_tl.shape == (rows, cols // 2)
    assert packed_tl.dtype == torch.int8
    assert scale_source_tl.shape == (rows, cols // 64)
    assert scale_source_tl.dtype == torch.uint32
    assert torch.equal(scale_source_tl.cpu(), scale_source_ref.cpu())

    semantic_words = unswizzle_blockscaled_chunk_kmajor_scale_words(scale_source_tl)
    assert torch.equal(swizzle_blockscaled_chunk_kmajor_scale_words(semantic_words).cpu(), scale_source_tl.cpu())

    scale = decode_ue4m3_scale_bytes(scale_bytes_ref).repeat_interleave(16, dim=1)
    decoded = decode_packed_fp4_e2m1(packed_tl) * scale
    error = (decoded - x.to(torch.float32)).abs()
    assert torch.isfinite(decoded).all()
    assert torch.all(error <= scale + 1e-6)


def test_swizzle_blockscaled_chunk_kmajor_pads_rows_to_full_tiles():
    rows, cols = 130, 8
    generator = torch.Generator(device="cpu").manual_seed(23)
    words = torch.randint(0, 2**31, (rows, cols), generator=generator, dtype=torch.int64).to(torch.uint32)

    swizzled = swizzle_blockscaled_chunk_kmajor_scale_words(words)
    assert swizzled.shape == (256, cols)

    back = unswizzle_blockscaled_chunk_kmajor_scale_words(swizzled)
    assert torch.equal(back[:rows], words)
    assert bool((back[rows:] == 0).all())
