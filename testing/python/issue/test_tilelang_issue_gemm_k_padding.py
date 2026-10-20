import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _make_gemm(m, n, k, tile_k, a_scope="shared", dtype="float16", accum_dtype="float32", transpose_b=False):
    b_shape = (n, k) if transpose_b else (k, n)
    b_tile_shape = (n, tile_k) if transpose_b else (tile_k, n)

    @T.prim_func
    def gemm(A: T.Tensor((m, k), dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor((m, n), accum_dtype)):
        with T.Kernel(1, threads=128):
            if a_scope == "shared":
                A_tile = T.alloc_shared((m, tile_k), dtype)
            else:
                A_tile = T.alloc_fragment((m, tile_k), dtype)
            B_tile = T.alloc_shared(b_tile_shape, dtype)
            C_local = T.alloc_fragment((m, n), accum_dtype)
            # Padding the computation requires zeros along K in both operands.
            T.clear(A_tile)
            T.clear(B_tile)
            T.copy(A, A_tile[:, :k])
            if transpose_b:
                T.copy(B, B_tile[:, :k])
            else:
                T.copy(B, B_tile[:k, :])
            T.gemm(A_tile, B_tile, C_local, transpose_B=transpose_b, clear_accum=True)
            T.copy(C_local, C)

    return gemm


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("m,k,a_scope", [(64, 20, "shared"), (64, 28, "shared"), (16, 20, "local.fragment")])
def test_gemm_irregular_k_reports_padding_before_layout_errors(m, k, a_scope):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target, pytest.raises(tvm.error.InternalError, match=rf"tile K={k} must be divisible by 16") as error:
        tilelang.lower(_make_gemm(m, 64, k, k, a_scope, accum_dtype="float16"), target=target)
    assert "Pad the K tile to 32 and zero-fill the extra K elements of both A and B" in str(error.value)


@tilelang.testing.requires_cuda
def test_gemm_int4_fragment_requires_padding_before_staging():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target, pytest.raises(tvm.error.InternalError, match="tile K=32 must be divisible by 64"):
        tilelang.lower(_make_gemm(64, 64, 32, 32, "local.fragment", "int4", "int32", transpose_b=True), target=target)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@pytest.mark.parametrize("k,tile_k,transpose_b", [(8, 16, False), (20, 32, False), (28, 32, True)])
def test_gemm_zero_padded_k_uses_mma_and_matches_reference(k, tile_k, transpose_b):
    m = n = 64
    kernel = tilelang.compile(
        _make_gemm(m, n, k, tile_k, transpose_b=transpose_b),
        target="cuda",
        out_idx=[2],
        pass_configs={"tl.disable_wgmma": True},
    )
    assert "tl::mma_sync" in kernel.get_kernel_source()
    a = ((torch.arange(m * k, device="cuda") % 7 - 3) / 4).reshape(m, k).half()
    b = ((torch.arange(k * n, device="cuda") % 5 - 2) / 4).reshape(k, n).half()
    actual_b = b.T.contiguous() if transpose_b else b
    expected = (a.cpu().double() @ b.cpu().double()).float()
    torch.testing.assert_close(kernel(a, actual_b).cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
