"""Regression tests for GitHub issue #2568.

This test asserts that the bool reductions compile, run, and produce results
identical to the equivalent logical reductions over the same data. The compile
path also exercises the wide-bool-vector handling that used to abort codegen
with ``Cannot convert type boolx8 to CUDA type``.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


M, N = 32, 32
THREADS = 128


def _make_reduce_kernel(dtype, reduce_fn):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M,), dtype)):
        with T.Kernel(1, 1, threads=THREADS) as (bx, by):
            A_fr = T.alloc_fragment((M, N), dtype)
            B_fr = T.alloc_fragment((M,), dtype)
            T.copy(A, A_fr)
            reduce_fn(A_fr, B_fr, dim=1)
            T.copy(B_fr, Out)

    return main


def _make_logical_kernel(logical_op):
    @T.prim_func
    def main(A: T.Tensor((M, N), "bool"), Out: T.Tensor((M,), "bool")):
        with T.Kernel(M, threads=THREADS) as bx:
            A_sh = T.alloc_shared((N,), "bool")
            for j in T.Parallel(N):
                A_sh[j] = A[bx, j]
            Out[bx] = logical_op(A_sh)

    return main


def _make_bool_tile():
    torch.manual_seed(0)
    a = torch.rand(M, N, device="cuda") < 0.5
    # Deterministic rows so both outcomes of the logical reductions occur:
    # row 0 is all-true (any_of/all_of -> True), row 1 is all-false (-> False).
    a[0] = True
    a[1] = False
    return a


@pytest.mark.parametrize(
    ("reduce_fn", "logical_op", "torch_reduce"),
    [
        (T.reduce_max, T.any_of, lambda a: a.any(dim=1)),
        (T.reduce_min, T.all_of, lambda a: a.all(dim=1)),
        (T.reduce_bitand, T.all_of, lambda a: a.all(dim=1)),
    ],
    ids=["reduce_max==any_of", "reduce_min==all_of", "reduce_bitand==all_of"],
)
@tilelang.testing.requires_cuda
def test_bool_reduce_matches_logical_reduction(reduce_fn, logical_op, torch_reduce):
    a = _make_bool_tile()

    out = tilelang.compile(_make_reduce_kernel("bool", reduce_fn), out_idx=[-1], target="cuda")(a)
    logical_out = tilelang.compile(_make_logical_kernel(logical_op), out_idx=[-1], target="cuda")(a)
    torch_ref = torch_reduce(a)

    assert out.dtype == torch.bool
    assert torch.equal(out, logical_out)
    assert torch.equal(out, torch_ref)


@pytest.mark.parametrize("dtype", ["int32", "float32"])
@tilelang.testing.requires_cuda
def test_non_bool_reduce_max_still_compiles(dtype):
    # Compile-time control from the issue: the int/float reduce path must be
    # unaffected by the bool identity fix.
    tilelang.compile(_make_reduce_kernel(dtype, T.reduce_max), out_idx=[-1], target="cuda")


if __name__ == "__main__":
    tilelang.testing.main()
