"""CuTeDSL runtime lowering of ``T.clamp`` (min/max, matching the CUDA target)."""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


@tilelang.jit(target="cutedsl")
def clamp_global_kernel(dtype, N=256, threads=64):
    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), Lo: T.Tensor((N,), dtype), Hi: T.Tensor((N,), dtype)):
        out = T.empty((N,), dtype)
        with T.Kernel(1, threads=threads):
            for i in T.Parallel(N):
                out[i] = T.clamp(A[i], Lo[i], Hi[i])
        return out

    return kernel


@tilelang.jit(target="cutedsl")
def clamp_fragment_kernel(dtype, M=64, N=128, threads=128):
    @T.prim_func
    def kernel(A: T.Tensor((M, N), dtype)):
        out = T.empty((M, N), dtype)
        with T.Kernel(1, threads=threads):
            a_local = T.alloc_fragment((M, N), dtype)
            T.copy(A, a_local)
            for i, j in T.Parallel(M, N):
                a_local[i, j] = T.clamp(a_local[i, j], -6.0, 6.0)
            T.copy(a_local, out)
        return out

    return kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_cutedsl_clamp_matches_torch(dtype):
    """Vectorized clamp against per-element bounds."""
    _require_cutedsl()
    torch_dtype = getattr(torch, dtype)
    a = (torch.randn(256, device="cuda") * 10).to(torch_dtype)
    lo = (torch.randn(256, device="cuda") * -3).to(torch_dtype)
    hi = lo + torch.rand(256, device="cuda").to(torch_dtype) * 4
    out = clamp_global_kernel(dtype)(a, lo, hi)
    torch.testing.assert_close(out, torch.clamp(a, lo, hi), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_cutedsl_clamp_fragment_scalar_bounds():
    """Clamp of a fragment against broadcast constant bounds."""
    _require_cutedsl()
    a = torch.randn(64, 128, device="cuda") * 10
    out = clamp_fragment_kernel("float32")(a)
    torch.testing.assert_close(out, a.clamp(-6, 6), atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
