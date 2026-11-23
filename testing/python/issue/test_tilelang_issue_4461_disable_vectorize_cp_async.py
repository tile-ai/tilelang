"""Regression test for #4461: `tirx.disable_vectorize` breaks pipelined
narrow `T.copy` with an illegal 1-/2-byte cp.async.

With `tirx.disable_vectorize=True`, the later `VectorizeLoop` pass
serializes vectorized loops (`VectorizeSkipper`) instead of merging them.
The cp.async injector still sized each transfer by the vectorized loop
extent (e.g. 2 bytes x 8 lanes = 16 bytes for fp16), so each lane reached
CUDA codegen as its own 1-element `ptx_cp_async` of 1 or 2 bytes, which
`IsValidCPAsyncTransferBytes` rejects:

    T.ptx_cp_async(...) requires a final PTX byte width in {4, 8, 16},
    but got 2

Post-fix the injector counts a vectorized loop as a single lane when
vectorization is disabled, so a copy whose per-access width is not itself
a legal cp.async width falls back to a normal copy.

The test has two parts:

1. **Codegen check** (host-independent). Lowers the issue's kernel for an
   explicit sm_89 target and asserts it lowers without raising. Runs on
   any host; pre-fix it raises the `IsValidCPAsyncTransferBytes` error
   for float16, bfloat16, int8 and uint8.
2. **Runtime check** on the host GPU. Compiles + executes the same kernel
   with vectorization disabled and asserts exact results.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

M, K, BM, BK = 128, 256, 64, 64


def _make_pipelined_copy_kernel(dtype, num_stages=2):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((M, BK), dtype)):
        with T.Kernel(M // BM, threads=128) as bx:
            As = T.alloc_shared((BM, BK), dtype)
            acc = T.alloc_fragment((BM, BK), dtype)
            T.clear(acc)
            for k in T.Pipelined(K // BK, num_stages=num_stages):
                T.copy(A[bx * BM, k * BK], As)  # global -> shared
                for i, j in T.Parallel(BM, BK):
                    acc[i, j] += As[i, j]
            T.copy(acc, B[bx * BM, 0])

    return main


def _lower(dtype, disable_vectorize):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_89"})
    cfg = {"tirx.disable_vectorize": True} if disable_vectorize else {}
    with tilelang.testing.without_kernel_cache(), tvm.transform.PassContext(config=cfg), target:
        return tilelang.lower(_make_pipelined_copy_kernel(dtype), target=target)


# --------------------------------------------------------------------------
# Codegen checks (host-independent)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int8", "uint8"])
def test_disable_vectorize_pipelined_narrow_copy_lowers(dtype):
    """Narrow pipelined copies fall back to a normal copy instead of
    reaching codegen as illegal 1-/2-byte cp.async."""
    artifact = _lower(dtype, disable_vectorize=True)
    assert "cp_async" not in artifact.kernel_source


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_vectorized_pipelined_copy_still_uses_cp_async(dtype):
    """Control: with vectorization enabled the same copies still take the
    cp.async path."""
    artifact = _lower(dtype, disable_vectorize=False)
    assert "cp_async" in artifact.kernel_source


def test_disable_vectorize_wide_copy_still_uses_cp_async():
    """Control: 4-byte elements stay on the cp.async path even with
    vectorization disabled, since one element is already a legal width."""
    artifact = _lower("float32", disable_vectorize=True)
    assert "cp_async" in artifact.kernel_source


# --------------------------------------------------------------------------
# Runtime check (requires GPU)
# --------------------------------------------------------------------------


@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_disable_vectorize_pipelined_narrow_copy_runs(dtype):
    """The issue's kernel compiles and gives exact results with
    vectorization disabled."""
    kernel = tilelang.compile(
        _make_pipelined_copy_kernel(dtype),
        out_idx=[1],
        target="cuda",
        pass_configs={"tirx.disable_vectorize": True},
    )
    torch_dtype = getattr(torch, dtype)
    a = torch.randint(0, 5, (M, K), device="cuda").to(torch_dtype)
    out = kernel(a)
    expected = a.view(M, K // BK, BK).sum(1)
    assert (out - expected).abs().max().item() == 0


if __name__ == "__main__":
    tilelang.testing.main()
