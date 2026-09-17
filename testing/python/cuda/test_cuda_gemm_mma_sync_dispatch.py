import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _make_gemm_kernel(dtype, accum_dtype):
    M = N = K = 64

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype),
             B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), accum_dtype)):
        with T.Kernel(1, 1, threads=128):
            As = T.alloc_shared((M, K), dtype)
            Bs = T.alloc_shared((K, N), dtype)
            Cl = T.alloc_fragment((M, N), accum_dtype)
            T.clear(Cl)
            T.copy(A, As)
            T.copy(B, Bs)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, C)

    return main


@tilelang.testing.requires_cuda
def test_gemm_bf16_accum_fp16_rejected_at_frontend():
    # Regression for bf16 x bf16 -> fp16: previously accepted by the frontend
    # and rejected by an nvcc static_assert; now must fail during lowering
    # with the native mma.sync diagnostic instead.
    with pytest.raises(tvm.error.InternalError, match=r"requires native mma\.sync lowering"):
        tilelang.compile(
            _make_gemm_kernel("bfloat16", "float16"),
            out_idx=[2],
            target={"kind": "cuda", "arch": "sm_89"},
        )


@pytest.mark.parametrize(
    "dtype,accum_dtype",
    [
        ("float16", "float16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
    ],
)
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(8)
def test_gemm_mma_sync_supported_dtype_combo_runs(dtype, accum_dtype):
    import torch

    kernel = tilelang.compile(
        _make_gemm_kernel(dtype, accum_dtype),
        out_idx=[2],
        target="cuda",
    )
    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=getattr(torch, dtype))
    b = torch.randn(64, 64, device="cuda", dtype=getattr(torch, dtype))
    out = kernel(a, b)
    ref = (a.float() @ b.float()).to(getattr(torch, accum_dtype))
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)