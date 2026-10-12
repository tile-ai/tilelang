"""SM120/SM121 ``mma.sync kind::mxf8f6f4.block_scale.scale_vec::1X`` (MXFP8)."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.cuda.intrinsics.macro.mma_sm120_macro_generator import block_scale_mma_kind_for_dtypes
from tilelang.cuda.language.intrinsics import TensorCoreIntrinEmitterSM120


def _make_mxf8f6f4_emitter(a_dtype=T.float8_e4m3fn, b_dtype=T.float8_e4m3fn, **kwargs):
    return TensorCoreIntrinEmitterSM120(
        is_blockscaled=True,
        kind="mxf8f6f4",
        scale_vec_size=1,
        stype="ue8m0",
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        accum_dtype=T.float32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=128,
        **kwargs,
    )


def test_mxf8f6f4_emitter_uses_k32_atoms():
    emitter = _make_mxf8f6f4_emitter()
    assert emitter.mma_prefix == "m16n8k32"
    assert emitter.micro_size_k == 32
    assert (emitter.a_dtype_abbrv, emitter.b_dtype_abbrv) == ("e4m3", "e4m3")
    assert not emitter.is_nvf4


def test_mxf8f6f4_emitter_accepts_mixed_fp8_operands():
    emitter = _make_mxf8f6f4_emitter(b_dtype=T.float8_e5m2)
    assert (emitter.a_dtype_abbrv, emitter.b_dtype_abbrv) == ("e4m3", "e5m2")


def test_mxf8f6f4_emitter_takes_unpacked_fp4_operands():
    emitter = _make_mxf8f6f4_emitter(b_dtype=T.float4_e2m1_unpacked)
    assert (emitter.a_dtype_abbrv, emitter.b_dtype_abbrv) == ("e4m3", "e2m1")
    assert emitter._ldmatrix_variant(T.float4_e2m1_unpacked, True) == "b4x16_p64"
    assert emitter._ldmatrix_variant(T.float8_e4m3fn, True) is None


def test_mxf8f6f4_emitter_rejects_packed_fp4_operands():
    # kind::mxf8f6f4 reads one byte per element; packed E2M1 cannot feed it.
    with pytest.raises(ValueError, match="mxf8f6f4 expects"):
        _make_mxf8f6f4_emitter(b_dtype=T.float4_e2m1fn)


def test_mxf8f6f4_scale_byte_selects_k32_block():
    emitter = _make_mxf8f6f4_emitter()
    # One UE8M0 byte per K32 atom; four atoms share a packed scale word.
    assert [emitter._scale_byte_id(0, ki, 32) for ki in range(4)] == [0, 1, 2, 3]
    assert [emitter._scale_word_k(0, ki, 32) for ki in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="packed scale words covering 128"):
        emitter._scale_word_k(0, 0, 16)


@pytest.mark.parametrize(
    "a_dtype, b_dtype, expected",
    [
        (T.float8_e4m3fn, T.float8_e4m3fn, ("mxf8f6f4", 1, "ue8m0")),
        (T.float8_e5m2, T.float8_e4m3fn, ("mxf8f6f4", 1, "ue8m0")),
        (T.float4_e2m1fn, T.float4_e2m1fn, ("mxf4nvf4", 4, "ue4m3")),
        (T.float8_e4m3fn, T.float4_e2m1_unpacked, ("mxf8f6f4", 1, "ue8m0")),
        (T.float8_e4m3fn, T.float4_e2m1fn, ("mxf8f6f4", 1, "ue8m0")),
    ],
)
def test_block_scale_kind_follows_operand_dtypes(a_dtype, b_dtype, expected):
    assert block_scale_mma_kind_for_dtypes(str(a_dtype), str(b_dtype)) == expected


def _make_mxfp8_matmul_kernel(M, N, K, block_K=128, fragment_a=False, in_dtype=T.float8_e4m3fn, b_fp4=False):
    sf_words = K // 128
    block_words = block_K // 128
    # FP4 weights stay packed in global memory and expand into padded shared rows.
    b_dtype = T.float4_e2m1fn if b_fp4 else in_dtype
    b_shared_dtype = T.float4_e2m1_unpacked if b_fp4 else in_dtype

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), b_dtype),
        SFA: T.Tensor((M, sf_words), T.uint32),
        SFB: T.Tensor((N, sf_words), T.uint32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=128) as (bx, by):
            A_tile = T.alloc_fragment((64, block_K), in_dtype) if fragment_a else T.alloc_shared((64, block_K), in_dtype)
            B_shared = T.alloc_shared((64, block_K), b_shared_dtype)
            SFA_shared = T.alloc_shared((64, block_words), T.uint32)
            SFB_shared = T.alloc_shared((64, block_words), T.uint32)
            C_local = T.alloc_fragment((64, 64), T.float32)
            T.clear(C_local)
            for ko in T.Pipelined(K // block_K, num_stages=2):
                T.copy(A[by * 64, ko * block_K], A_tile)
                T.copy(B[bx * 64, ko * block_K], B_shared)
                for i, w in T.Parallel(64, block_words):
                    SFA_shared[i, w] = SFA[by * 64 + i, ko * block_words + w]
                for j, w in T.Parallel(64, block_words):
                    SFB_shared[j, w] = SFB[bx * 64 + j, ko * block_words + w]
                T.mma_gemm_blockscaled(
                    A_tile,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    k_start=0,
                    sf_a_granularity_k=32,
                    sf_b_granularity_k=32,
                )
            T.copy(C_local, C[by * 64, bx * 64])

    return main


@tilelang.testing.requires_cuda(support_required="compile-only")
@pytest.mark.parametrize("arch", ["sm_120a", "sm_121a"])
def test_mxf8f6f4_codegen(arch):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        result = tilelang.lower(_make_mxfp8_matmul_kernel(128, 128, 256), target=target, enable_device_compile=False)
    src = result.kernel_source
    assert "#include <tl_templates/cuda/instruction/mma_block_scale.h>" in src
    assert "tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0" in src
    assert "tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3" in src
    assert "SM120MmaBlockScaledKind::kMxf4nvf4" not in src


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_mxf8f6f4_fp4_operand_codegen():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_121a"})
    with target:
        result = tilelang.lower(_make_mxfp8_matmul_kernel(128, 128, 256, b_fp4=True), target=target, enable_device_compile=False)
    src = result.kernel_source
    assert "tl::ptx_ldmatrix_b4x16_p64_x4" in src
    assert "tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE2M1" in src


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_mxf8f6f4_rejects_fragment_a():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_121a"})
    with target, pytest.raises(Exception, match="mxf8f6f4 block-scaled GEMM currently requires A in shared memory"):
        tilelang.lower(_make_mxfp8_matmul_kernel(64, 64, 128, fragment_a=True), target=target, enable_device_compile=False)


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_mxf8f6f4_rejects_compact_kmajor_scales():
    @T.prim_func
    def main(O: T.Tensor((64, 64), T.float32)):
        with T.Kernel(1, threads=128):
            A = T.alloc_shared((64, 128), T.float8_e4m3fn)
            B = T.alloc_shared((64, 128), T.float8_e4m3fn)
            SFA = T.alloc_shared((64, 1), T.uint32)
            SFB = T.alloc_shared((64, 1), T.uint32)
            C = T.alloc_fragment((64, 64), T.float32)
            T.mma_gemm_blockscaled(
                A,
                B,
                C,
                SFA,
                SFB,
                transpose_B=True,
                clear_accum=True,
                k_start=0,
                sf_a_granularity_k=32,
                sf_b_granularity_k=32,
                sf_layout="blockscaled_chunk_kmajor",
            )
            T.copy(C, O)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_121a"})
    with target, pytest.raises(Exception, match="mxf8f6f4 block-scaled GEMM currently requires sf_layout='rowmajor'"):
        tilelang.lower(main, target=target, enable_device_compile=False)


def _random_mxfp8(rows, K, seed):
    """Small-integer E4M3 values and power-of-two UE8M0 scales.

    Every partial sum stays an exactly representable FP32 integer multiple of a
    power of two, so the kernel must match the reference bit for bit.
    """
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randint(-4, 5, (rows, K), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    exponents = (127 + torch.randint(-2, 3, (rows, K // 32), generator=gen, device="cuda")).to(torch.uint8)
    return values, exponents


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _random_mxfp4(rows, K, seed):
    """Every E2M1 code, packed two per byte (low nibble first), with UE8M0 scales."""
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 16, (rows, K), generator=gen, device="cuda").to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8).contiguous()
    exponents = (127 + torch.randint(-2, 3, (rows, K // 32), generator=gen, device="cuda")).to(torch.uint8)
    values = torch.tensor(_E2M1_VALUES, device="cuda")[codes.long()]
    return packed, exponents, values


def _reference(a, sfa, b, sfb):
    import torch

    def dequant(values, exponents):
        return values.float() * torch.exp2(exponents.float() - 127.0).repeat_interleave(32, dim=1)

    return dequant(a, sfa) @ dequant(b, sfb).T


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
@pytest.mark.parametrize(
    "M, N, K, block_K, fragment_a",
    [
        (64, 64, 128, 128, False),
        (128, 192, 512, 128, False),
        (128, 128, 512, 256, False),
    ],
)
def test_mxf8f6f4_mma_block_scale_correctness(M, N, K, block_K, fragment_a):
    import torch

    a, sfa = _random_mxfp8(M, K, seed=1)
    b, sfb = _random_mxfp8(N, K, seed=2)
    kernel = tilelang.compile(_make_mxfp8_matmul_kernel(M, N, K, block_K, fragment_a), target="cuda", out_idx=[4])
    out = kernel(a, b, sfa.view(torch.uint32), sfb.view(torch.uint32))
    torch.testing.assert_close(out, _reference(a, sfa, b, sfb), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
@pytest.mark.parametrize("M, N, K", [(64, 64, 128), (128, 192, 512)])
def test_mxf8f6f4_fp8_fp4_correctness(M, N, K):
    import torch

    a, sfa = _random_mxfp8(M, K, seed=3)
    b_packed, sfb, b_values = _random_mxfp4(N, K, seed=4)
    kernel = tilelang.compile(_make_mxfp8_matmul_kernel(M, N, K, b_fp4=True), target="cuda", out_idx=[4])
    out = kernel(a, b_packed, sfa.view(torch.uint32), sfb.view(torch.uint32))
    torch.testing.assert_close(out, _reference(a, sfa, b_values, sfb), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
