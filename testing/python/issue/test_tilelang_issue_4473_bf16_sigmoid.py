"""Regression test for #4473: bf16 `T.sigmoid` fails to lower for sm_100a/sm_120.

With 256-bit vectorization a thread holds 16 bfloat16 lanes. The default
legalization of `tirx.sigmoid` (`1 / (1 + exp(-x))`) builds `exp` through the
`tirx::exp` C++ helper, which promotes a bfloat16 argument to float32 with
the same lane count, producing a `float32x16` expression. CUDA codegen has
no vector type for it:

    tvm.error.InternalError: Cannot convert type float32x16 to CUDA type

Post-fix, a wide (more than 8 lanes) bfloat16 sigmoid builds the `tirx.exp`
call directly on the bfloat16 value, so it lowers to `hexp` like a plain
`T.exp` does, while narrow vectors keep the default float32 evaluation.

The test has two parts:

1. **Codegen checks**. Lower the issue's kernel for explicit sm_100a /
   sm_120 targets and assert it lowers, uses `hexp` (not `expf`), and keeps
   the 256-bit vector store (`ulonglong4`). Requires a CUDA-enabled build
   (the CUDA target must exist to lower); pre-fix lowering raises the
   `Cannot convert type float32x16 to CUDA type` error. A control asserts
   that sm_90a (8 lanes per thread) still evaluates `exp` in float32.
2. **Runtime check** on the host GPU. Compiles + executes the same kernel
   and compares against `torch.sigmoid`.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _make_sigmoid_kernel(dtype="bfloat16"):
    @T.prim_func
    def main(A: T.Tensor((8, 64), dtype), O: T.Tensor((8, 64), dtype)):
        with T.Kernel(1, threads=32):
            for i, j in T.Parallel(8, 64):
                O[i, j] = T.sigmoid(A[i, j])

    return main


def _lower(arch, config=None):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with (
        tilelang.testing.without_kernel_cache(),
        tvm.transform.PassContext(opt_level=3, config=config or {}),
        target,
    ):
        return tilelang.lower(_make_sigmoid_kernel(), target=target)


# --------------------------------------------------------------------------
# Codegen checks (host-independent)
# --------------------------------------------------------------------------


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_100a", "sm_120"])
def test_bf16_sigmoid_wide_vector_lowers(arch):
    """A 16-lane bfloat16 sigmoid lowers via hexp instead of failing on
    float32x16, keeping the 256-bit vector store."""
    artifact = _lower(arch)
    assert "hexp" in artifact.kernel_source
    assert "expf" not in artifact.kernel_source
    assert "ulonglong4" in artifact.kernel_source


@tilelang.testing.requires_cuda
def test_bf16_sigmoid_narrow_vector_still_uses_float32_exp():
    """Control: with 8 lanes per thread (sm_90a) the default float32
    evaluation is preserved."""
    artifact = _lower("sm_90a")
    assert "expf" in artifact.kernel_source
    assert "hexp" not in artifact.kernel_source


# --------------------------------------------------------------------------
# Runtime check (requires GPU)
# --------------------------------------------------------------------------


@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_bf16_sigmoid_runs():
    """The issue's kernel compiles and matches torch.sigmoid."""
    kernel = tilelang.compile(_make_sigmoid_kernel(), out_idx=[1], target="cuda")
    a = (torch.randn(8, 64) * 3).to(torch.bfloat16).cuda()
    out = kernel(a)
    expected = torch.sigmoid(a.float()).to(torch.bfloat16)
    assert torch.allclose(out.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
