"""Regression test for issue #4425.

T.reduce_max / T.reduce_min on float16/bfloat16 emitted
``-std::numeric_limits<half_t>::infinity()`` into the generated CUDA source.
CUTLASS only defines ``std::numeric_limits<cutlass::half_t>`` /
``std::numeric_limits<cutlass::bfloat16_t>`` when ``__CUDACC_RTC__`` is *not*
defined, and ``tl_templates/cuda/nvrtc_std.h`` provides no
``std::numeric_limits`` at all, so NVRTC compilation failed with
``namespace "std" has no member "numeric_limits"``.

The fix emits the infinity/NaN bit patterns through the CUTLASS bitcast
helpers (``half_t::bitcast(0x7c00)``, ``bfloat16_t::bitcast(0x7f80)`` and
``bitcast(0x7fff)`` for NaN), which exist under both NVRTC and nvcc.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.testing import requires_cuda

SM80_TARGET = {"kind": "cuda", "arch": "sm_80"}

# (dtype, reduce op, expected identity constant in the generated source)
_REDUCE_CASES = [
    ("float16", "max", "-half_t::bitcast(0x7c00)"),
    ("float16", "min", "half_t::bitcast(0x7c00)"),
    ("bfloat16", "max", "-bfloat16_t::bitcast(0x7f80)"),
    ("bfloat16", "min", "bfloat16_t::bitcast(0x7f80)"),
]


def _make_reduce_kernel(dtype, op):
    @T.prim_func
    def main(A: T.Tensor((64, 64), dtype), B: T.Tensor((64,), dtype)):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((64, 64), dtype)
            dst = T.alloc_fragment((64,), dtype)
            T.copy(A, src)
            if op == "max":
                T.reduce_max(src, dst, dim=1, clear=True)
            else:
                T.reduce_min(src, dst, dim=1, clear=True)
            T.copy(dst, B)

    return main


def _lower_to_cuda_source(func) -> str:
    with tvm.target.Target(SM80_TARGET):
        artifact = tilelang.lower(func, target=SM80_TARGET)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


@requires_cuda
@pytest.mark.parametrize("dtype,op,expected", _REDUCE_CASES)
def test_issue_4425_reduce_fp16_bf16_no_numeric_limits(dtype, op, expected):
    """fp16/bf16 reduce identity must not use std::numeric_limits.

    The NVRTC prelude (nvrtc_std.h) has no std::numeric_limits and CUTLASS
    hides its specializations under __CUDACC_RTC__, so the old emission
    failed NVRTC compilation. The identity must be spelled via bitcast.
    """
    src = _lower_to_cuda_source(_make_reduce_kernel(dtype, op))
    assert "std::numeric_limits" not in src, f"reduce_{op} on {dtype} must not emit std::numeric_limits (unavailable under NVRTC):\n{src}"
    assert expected in src, f"reduce_{op} on {dtype} should initialize with {expected}:\n{src}"


@requires_cuda
def test_issue_4425_fp16_nan_constant_no_numeric_limits():
    """An fp16 NaN constant must also avoid std::numeric_limits."""

    @T.prim_func
    def main(A: T.Tensor((4,), "float16"), B: T.Tensor((4,), "float16")):
        with T.Kernel(1):
            for i in T.grid(4):
                B[i] = A[i] + T.float16(float("nan"))

    src = _lower_to_cuda_source(main)
    assert "std::numeric_limits" not in src, f"fp16 NaN constant must not emit std::numeric_limits:\n{src}"
    assert "half_t::bitcast(0x7fff)" in src, f"fp16 NaN constant should be half_t::bitcast(0x7fff):\n{src}"


@requires_cuda
@pytest.mark.parametrize("dtype,op", [(d, o) for d, o, _ in _REDUCE_CASES])
def test_issue_4425_reduce_runs(dtype, op):
    """End-to-end check: the bitcast identities give correct results."""
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    a = torch.randn((64, 64), dtype=torch_dtype, device="cuda")
    ref = a.amax(dim=1) if op == "max" else a.amin(dim=1)

    kernel = tilelang.compile(_make_reduce_kernel(dtype, op), target="cuda")
    out = torch.empty((64,), dtype=torch_dtype, device="cuda")
    kernel(a, out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    tilelang.testing.main()
