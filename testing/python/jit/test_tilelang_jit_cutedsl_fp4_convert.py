"""CuTeDSL conversion/store lowering for sub-byte float types."""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

# FP4 E2M1 magnitudes for nibble payloads 0-7; bit 3 is the sign.
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quantize_fp4(value):
    """Round ``value`` to its FP4 E2M1 nibble: nearest, ties to the even index,
    saturating at 6.0."""
    magnitude = abs(float(value))
    index = min(range(8), key=lambda i: (abs(_FP4_MAGNITUDES[i] - magnitude), i % 2))
    return index | (0x8 if value < 0 else 0x0)


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
    """Every stored nibble must be the FP4 rounding of its input element."""
    _require_cutedsl()

    values = torch.linspace(-6.0, 6.0, 256, dtype=torch.bfloat16, device="cuda")
    out = fp4_cast_store_kernel()(values)

    nibbles = []
    for byte in out.view(torch.uint8).cpu().numpy().tolist():
        nibbles.extend((byte & 0x0F, byte >> 4))

    # The packing order is element 2k in the low nibble of byte k; compare each
    # element against its own rounding so a wrong order or a wrong conversion
    # fails instead of merely producing representable values.
    expected = [_quantize_fp4(value) for value in values.float().cpu().tolist()]
    mismatches = [(i, got, want) for i, (got, want) in enumerate(zip(nibbles, expected, strict=True)) if got != want]
    assert not mismatches, f"first FP4 mismatches (index, got, want): {mismatches[:8]}"


if __name__ == "__main__":
    tilelang.testing.main()
