"""Tests for the per-call ``nan_propagate`` kwarg on T.reduce_max / reduce_min /
reduce_absmax for float16 and bfloat16 buffers (CUDA only)."""

import math

import torch

import tilelang
import tilelang.testing
import tilelang.language as T

_DTYPES = [("float16", T.float16, torch.float16), ("bfloat16", T.bfloat16, torch.bfloat16)]


def _compile(prim_func):
    return tilelang.compile(prim_func, out_idx=-1, target="cuda")


def _make_reduce_kernel(reduce_fn, length, dtype, *, nan_propagate):

    @T.prim_func
    def kernel(a: T.Tensor((length,), dtype), out: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=32):
            frag = T.alloc_fragment((length,), dtype)
            out_frag = T.alloc_fragment((1,), dtype)
            T.copy(a, frag)
            reduce_fn(frag, out_frag, nan_propagate=nan_propagate)
            T.copy(out_frag, out)

    return kernel


# ---------------------------------------------------------------------------
# Source-level checks: confirm the right reducer / intrinsic is emitted.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_reduce_max_default_uses_plain_op():
    k = _compile(_make_reduce_kernel(T.reduce_max, 64, T.float16, nan_propagate=False))
    src = k.get_kernel_source()
    assert "tl::MaxOp" in src and "MaxOpNan" not in src
    assert "__hmax(" in src and "__hmax_nan" not in src


@tilelang.testing.requires_cuda
def test_reduce_max_nan_propagate_uses_nan_op():
    k = _compile(_make_reduce_kernel(T.reduce_max, 64, T.float16, nan_propagate=True))
    src = k.get_kernel_source()
    assert "tl::MaxOpNan" in src
    assert "__hmax_nan" in src


@tilelang.testing.requires_cuda
def test_reduce_min_nan_propagate_uses_nan_op():
    k = _compile(_make_reduce_kernel(T.reduce_min, 64, T.bfloat16, nan_propagate=True))
    src = k.get_kernel_source()
    assert "tl::MinOpNan" in src
    assert "__hmin_nan" in src


@tilelang.testing.requires_cuda
def test_reduce_absmax_nan_propagate_uses_nan_op():
    k = _compile(_make_reduce_kernel(T.reduce_absmax, 64, T.float16, nan_propagate=True))
    src = k.get_kernel_source()
    assert "tl::MaxOpNan" in src
    assert "__hmax_nan" in src


# ---------------------------------------------------------------------------
# Runtime behavioral checks: NaN actually propagates only when requested.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_reduce_max_runtime_nan_behavior():
    for _, tl_dtype, torch_dtype in _DTYPES:
        length = 64
        a = torch.arange(length, dtype=torch.float32).to(torch_dtype).cuda()
        a[7] = float("nan")

        k_default = _compile(_make_reduce_kernel(T.reduce_max, length, tl_dtype, nan_propagate=False))
        k_nan = _compile(_make_reduce_kernel(T.reduce_max, length, tl_dtype, nan_propagate=True))

        out_default = k_default(a)
        out_nan = k_nan(a)

        assert not math.isnan(out_default.float().item()), f"{tl_dtype}: default reduce_max should ignore NaN, got {out_default}"
        assert math.isnan(out_nan.float().item()), f"{tl_dtype}: nan_propagate reduce_max should return NaN, got {out_nan}"


@tilelang.testing.requires_cuda
def test_reduce_min_runtime_nan_behavior():
    for _, tl_dtype, torch_dtype in _DTYPES:
        length = 64
        a = torch.arange(length, dtype=torch.float32).to(torch_dtype).cuda()
        a[13] = float("nan")

        k_default = _compile(_make_reduce_kernel(T.reduce_min, length, tl_dtype, nan_propagate=False))
        k_nan = _compile(_make_reduce_kernel(T.reduce_min, length, tl_dtype, nan_propagate=True))

        assert not math.isnan(k_default(a).float().item())
        assert math.isnan(k_nan(a).float().item())


if __name__ == "__main__":
    tilelang.testing.main()
