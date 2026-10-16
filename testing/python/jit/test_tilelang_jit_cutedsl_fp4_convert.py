"""CuTeDSL conversion/store lowering for sub-byte float types."""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

# FP4 E2M1 magnitudes for nibble payloads 0-7; bit 3 is the sign.
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _require_cutedsl():
    """Skip when the CuTeDSL Python stack is unavailable."""
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


@tilelang.jit(target="cutedsl")
def fp4_cast_store_kernel(N=256, threads=64):
    """Cast bf16 -> float4_e2m1fn and store the packed bytes."""

    @T.prim_func
    def kernel(A: T.Tensor((N,), "bfloat16")):
        out = T.empty((N,), "float4_e2m1fn")
        with T.Kernel(1, threads=threads):
            for i in T.Parallel(N):
                out[i] = T.cast(A[i], "float4_e2m1fn")
        return out

    return kernel


@tilelang.testing.requires_cuda
def test_cutedsl_fp4_cast_store_is_packed():
    """`float4_e2m1fn` casts must lower, and the result must be packed (N/2).

    The CuTeDSL conversion helpers used to call the removed MLIR ops
    `vector.extractelement` / `vector.insertelement`, and the adapter used to
    materialize the output with the unpacked logical shape.
    """
    _require_cutedsl()

    out = fp4_cast_store_kernel()(torch.randn(256, dtype=torch.bfloat16, device="cuda"))
    assert tuple(out.shape) == (128,), f"FP4 output must be packed, got {tuple(out.shape)}"
    assert out.dtype == torch.float4_e2m1fn_x2
    assert out.view(torch.uint8).any()


@tilelang.testing.requires_cuda
def test_cutedsl_fp4_cast_values_are_fp4():
    """Every stored nibble must decode to a representable FP4 E2M1 value."""
    _require_cutedsl()

    values = torch.linspace(-6.0, 6.0, 256, dtype=torch.bfloat16, device="cuda")
    out = fp4_cast_store_kernel()(values)

    decoded = []
    for byte in out.view(torch.uint8).cpu().numpy().tolist():
        for nibble in (byte & 0x0F, byte >> 4):
            magnitude = _FP4_MAGNITUDES[nibble & 0x7]
            decoded.append(-magnitude if nibble & 0x8 else magnitude)

    assert set(decoded) <= {m for m in _FP4_MAGNITUDES} | {-m for m in _FP4_MAGNITUDES}
    assert min(decoded) < 0 < max(decoded)


if __name__ == "__main__":
    tilelang.testing.main()
