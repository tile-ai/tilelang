"""SM120/SM121 ``mma.sync kind::mxf4nvf4.block_scale.scale_vec::2X`` (MXFP4, UE8M0 per 32)."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.cuda.intrinsics.macro.mma_sm120_macro_generator import block_scale_mma_kind_for_dtypes
from tilelang.cuda.language.intrinsics import TensorCoreIntrinEmitterSM120

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _emitter():
    return TensorCoreIntrinEmitterSM120(
        is_blockscaled=True,
        kind="mxf4nvf4",
        scale_vec_size=2,
        stype="ue8m0",
        a_dtype=T.float4_e2m1fn,
        b_dtype=T.float4_e2m1fn,
        accum_dtype=T.float32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=128,
        num_elems_per_byte=2,
    )


def test_mxfp4_selected_by_granularity():
    assert block_scale_mma_kind_for_dtypes("float4_e2m1fn", "float4_e2m1fn", 32) == ("mxf4nvf4", 2, "ue8m0")
    assert block_scale_mma_kind_for_dtypes("float4_e2m1fn", "float4_e2m1fn", 16) == ("mxf4nvf4", 4, "ue4m3")


def test_mxfp4_emitter_selects_scale_halves():
    emitter = _emitter()
    assert emitter.mma_prefix == "m16n8k64"
    assert emitter.uses_packed_fp4 and not emitter.is_nvf4
    # Each K64 atom reads one half of the packed word: selector 0, then 2.
    assert [emitter._scale_byte_id(0, ki, 32) for ki in range(2)] == [0, 2]
    assert [emitter._scale_word_k(0, ki, 32) for ki in range(4)] == [0, 0, 1, 1]


def _make_mxfp4_matmul_kernel(M, N, K):
    words = K // 128

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float4_e2m1fn),
        B: T.Tensor((N, K), T.float4_e2m1fn),
        SFA: T.Tensor((M, words), T.uint32),
        SFB: T.Tensor((N, words), T.uint32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=128) as (bx, by):
            A_shared = T.alloc_shared((64, 128), T.float4_e2m1fn)
            B_shared = T.alloc_shared((64, 128), T.float4_e2m1fn)
            SFA_shared = T.alloc_shared((64, 1), T.uint32)
            SFB_shared = T.alloc_shared((64, 1), T.uint32)
            C_local = T.alloc_fragment((64, 64), T.float32)
            T.clear(C_local)
            for ko in T.serial(K // 128):
                for i, k in T.Parallel(64, 128):
                    A_shared[i, k] = A[by * 64 + i, ko * 128 + k]
                for j, k in T.Parallel(64, 128):
                    B_shared[j, k] = B[bx * 64 + j, ko * 128 + k]
                for i in T.Parallel(64):
                    SFA_shared[i, 0] = SFA[by * 64 + i, ko]
                for j in T.Parallel(64):
                    SFB_shared[j, 0] = SFB[bx * 64 + j, ko]
                T.mma_gemm_blockscaled(
                    A_shared,
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


@pytest.mark.parametrize("arch", ["sm_120a", "sm_121a"])
def test_mxfp4_codegen(arch):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        result = tilelang.lower(_make_mxfp4_matmul_kernel(64, 64, 256), target=target, enable_device_compile=False)
    assert "tl::SM120MmaBlockScaledKind::kMxf4nvf4, 2, tl::SM120MmaScaleType::kUE8M0" in result.kernel_source


def _random_mxfp4(rows, K, seed):
    """Packed E2M1 codes and power-of-two UE8M0 scales; products sum exactly in FP32."""
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 16, (rows, K), generator=gen, device="cuda").to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8).contiguous()
    exponents = (127 + torch.randint(-2, 3, (rows, K // 32), generator=gen, device="cuda")).to(torch.uint8)
    values = torch.tensor(_E2M1_VALUES, device="cuda")[codes.long()]
    return packed, exponents, values * torch.exp2(exponents.float() - 127.0).repeat_interleave(32, dim=1)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
@pytest.mark.parametrize("M, N, K", [(64, 64, 128), (128, 192, 512)])
def test_mxfp4_mma_block_scale_correctness(M, N, K):
    import torch

    a, sfa, a_values = _random_mxfp4(M, K, seed=5)
    b, sfb, b_values = _random_mxfp4(N, K, seed=6)
    kernel = tilelang.compile(_make_mxfp4_matmul_kernel(M, N, K), target="cuda", out_idx=[4])
    out = kernel(a, b, sfa.view(torch.uint32), sfb.view(torch.uint32))
    torch.testing.assert_close(out, a_values @ b_values.T, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
