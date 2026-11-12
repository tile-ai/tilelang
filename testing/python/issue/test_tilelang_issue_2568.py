import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _bool_reduce_kernel(reduce_op, M, N, threads):
    @T.prim_func
    def main(A: T.Tensor((M, N), "bool"), B: T.Tensor((M,), "bool")):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((M, N), "bool")
            dst = T.alloc_fragment((M,), "bool")
            T.copy(A, src)
            reduce_op(src, dst, dim=1)
            T.copy(dst, B)

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    ("reduce_name", "reduce_op", "ref_fn"),
    [
        ("reduce_max", T.reduce_max, lambda a: a.any(dim=1)),
        ("reduce_min", T.reduce_min, lambda a: a.all(dim=1)),
        ("reduce_bitand", T.reduce_bitand, lambda a: a.all(dim=1)),
    ],
)
@pytest.mark.parametrize(("M", "N", "threads"), [(8, 8, 32), (4, 4, 32)])
def test_bool_reduce_matches_logical_reduce(reduce_name, reduce_op, ref_fn, M, N, threads):
    """bool reduce_max/min/bitand lower with the any_of/all_of identities (#2568).

    The bool case used to fall into the float identity branch of
    MakeInitValue, where make_const(bool, +/-INFINITY) crashed on the bool
    IntImm range check with INT64_MIN. With dedicated bool identities the
    reduce matches the intended logical reductions over the same data.

    Only width-1 fragment layouts are covered here: wider bool fragments
    still hit the vectorized-bool codegen gap tracked in #2206.
    """
    kernel = tilelang.compile(_bool_reduce_kernel(reduce_op, M, N, threads), out_idx=[-1])

    torch.manual_seed(0)
    A = torch.rand(M, N, device="cuda") < 0.5
    B = kernel(A)

    torch.testing.assert_close(B, ref_fn(A), check_dtype=True)
