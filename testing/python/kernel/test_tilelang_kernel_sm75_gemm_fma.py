import torch

import tilelang
import tilelang.language as T
import tilelang.testing

_SM75 = {"kind": "cuda", "arch": "sm_75"}


def _make_ss_gemm(M, N, K, in_dtype, accum_dtype, threads=128):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared((M, K), in_dtype)
            B_shared = T.alloc_shared((K, N), in_dtype)
            C_local = T.alloc_fragment((M, N), accum_dtype)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
            T.copy(C_local, C)

    return main


# ---------------------------------------------------------------------------
# Compile-only dispatch checks pinned to an explicit sm_75 target, so they run
# on any CUDA runner regardless of the host GPU. Turing has no fp32/tf32,
# fp64, or bf16 mma.sync atoms; these dtypes must lower through the FMA
# fallback instead of tl::mma_sync (which fails nvcc's static_assert for bf16
# and compiles into a cutlass runtime trap for fp32/fp64).
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_sm75_fp32_gemm_compiles_to_fma_fallback():
    kernel = tilelang.compile(_make_ss_gemm(64, 64, 32, T.float32, T.float32), target=_SM75, out_idx=[2])
    assert "tl::mma_sync" not in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
def test_sm75_fp64_gemm_compiles_to_fma_fallback():
    kernel = tilelang.compile(_make_ss_gemm(32, 32, 32, T.float64, T.float64), target=_SM75, out_idx=[2])
    assert "tl::mma_sync" not in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
def test_sm75_bf16_gemm_compiles_to_fma_fallback():
    kernel = tilelang.compile(_make_ss_gemm(64, 64, 32, T.bfloat16, T.float32), target=_SM75, out_idx=[2])
    assert "tl::mma_sync" not in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
def test_sm75_fp16_gemm_keeps_mma_path():
    """f16 has native Turing atoms; the fallback must not demote it."""
    kernel = tilelang.compile(_make_ss_gemm(64, 64, 32, T.float16, T.float32), target=_SM75, out_idx=[2])
    assert "tl::mma_sync" in kernel.get_kernel_source()


# ---------------------------------------------------------------------------
# Execution checks on real Turing hardware.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_fp32_gemm_fma_matches_torch():
    kernel = tilelang.compile(_make_ss_gemm(64, 64, 32, T.float32, T.float32), target="cuda", out_idx=[2])
    a = torch.randn((64, 32), device="cuda", dtype=torch.float32)
    b = torch.randn((32, 64), device="cuda", dtype=torch.float32)
    c = kernel(a, b)
    tilelang.testing.torch_assert_close(c, a @ b, rtol=1e-4, atol=1e-4)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_fp64_gemm_fma_matches_torch():
    kernel = tilelang.compile(_make_ss_gemm(32, 32, 32, T.float64, T.float64), target="cuda", out_idx=[2])
    a = torch.randn((32, 32), device="cuda", dtype=torch.float64)
    b = torch.randn((32, 32), device="cuda", dtype=torch.float64)
    c = kernel(a, b)
    tilelang.testing.torch_assert_close(c, a @ b, rtol=1e-8, atol=1e-8)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_bf16_gemm_fma_matches_torch():
    kernel = tilelang.compile(_make_ss_gemm(64, 64, 32, T.bfloat16, T.float32), target="cuda", out_idx=[2])
    a = torch.randn((64, 32), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((32, 64), device="cuda", dtype=torch.bfloat16)
    c = kernel(a, b)
    ref = a.float() @ b.float()
    tilelang.testing.torch_assert_close(c, ref, rtol=1e-2, atol=1e-2, max_mismatched_ratio=0.01)


if __name__ == "__main__":
    tilelang.testing.main()
