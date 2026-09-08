"""Regression test for https://github.com/tile-ai/tilelang/issues/3021

Logical `T.Not` on a `bool` buffer inside an auto-vectorized `T.Parallel`
loop used to emit a scalar `!` on the whole vector carrier (e.g.
`!ushort4`), which nvcc rejects. The NotNode visitor must scalarize the
negation lane by lane.

The negated operand must be a plain `bool` buffer load: a NotNode over a
comparison (e.g. `T.Not(A[i] > 0)`) is rewritten by the simplifier before
codegen and would not exercise the vectorized path.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

N = 256


def _make_not_kernel(threads):
    @T.prim_func
    def main(A: T.Tensor((N,), "bool"), C: T.Tensor((N,), "bool")):
        with T.Kernel(1, threads=threads):
            for i in T.Parallel(N):
                C[i] = T.Not(A[i])

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "threads",
    [
        256,  # one element per thread: scalar `!x` (control)
        128,  # two elements per thread: boolx2
        64,  # four elements per thread: boolx4 (the reported crash)
    ],
)
def test_vectorized_bool_not(threads):
    kernel = tilelang.compile(_make_not_kernel(threads), out_idx=[1])

    a = torch.rand(N, device="cuda") > 0.5
    c = kernel(a)
    torch.testing.assert_close(c, ~a, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
