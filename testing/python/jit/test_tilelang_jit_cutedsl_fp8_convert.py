"""CuTeDSL FP8 conversion/store lowering."""

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
def cast_store_kernel(src, dst, N=256, threads=64):
    @T.prim_func
    def kernel(A: T.Tensor((N,), src)):
        out = T.empty((N,), dst)
        with T.Kernel(1, threads=threads):
            for i in T.Parallel(N):
                out[i] = T.cast(A[i], dst)
        return out

    return kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize("src", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize("dst", ["float8_e4m3fn", "float8_e5m2"])
def test_cutedsl_fp8_cast_store(src, dst):
    """Narrowing to FP8 must use the FN encoding the codegen emits and match torch bit-exactly."""
    _require_cutedsl()
    a = (torch.randn(256, device="cuda") * 8).to(getattr(torch, src))
    out = cast_store_kernel(src, dst)(a)
    ref = a.float().to(getattr(torch, dst))
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out.view(torch.uint8), ref.view(torch.uint8), atol=0, rtol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize("dst", ["bfloat16", "float16", "float32"])
def test_cutedsl_fp8_cast_widen(dst):
    _require_cutedsl()
    a = (torch.randn(256, device="cuda") * 8).to(torch.float8_e4m3fn)
    out = cast_store_kernel("float8_e4m3fn", dst)(a)
    torch.testing.assert_close(out, a.to(getattr(torch, dst)), atol=0, rtol=0)


def test_fp8_conversion_compile_hint(monkeypatch):
    """libNVVM failures of FP8 kernels before sm_90 get an actionable explanation."""
    from tilelang.jit.adapter.cutedsl import checks

    source = "_cast.store(tl.cast_tensor(x, cutlass.Float8E4M3FN))"
    monkeypatch.setattr(checks._importlib_metadata, "version", lambda _: "4.7.1")
    assert "nvidia-cutlass-dsl>=4.8.0" in checks.fp8_conversion_compile_hint("sm_89", source)
    assert "sm_89 or newer" in checks.fp8_conversion_compile_hint("sm_86", source)
    assert checks.fp8_conversion_compile_hint("sm_90a", source) is None
    assert checks.fp8_conversion_compile_hint("sm_89", "_cast.store(x)") is None
    monkeypatch.setattr(checks._importlib_metadata, "version", lambda _: "4.8.0")
    assert checks.fp8_conversion_compile_hint("sm_89", source) is None


if __name__ == "__main__":
    tilelang.testing.main()
