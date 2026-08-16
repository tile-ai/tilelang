"""Tests for default (contiguous) stride construction in the Python frontend.

``construct_strides`` is the single source of truth: ``T.Tensor``, ``T.out``
(via ``OutTensor.strides``) and ``retrieve_stride`` all route through it, so the
rank invariant ``len(strides) == len(shape)`` must hold everywhere.
"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import tirx
from tilelang.language.eager.builder import OutTensor
from tilelang.language.eager.utils import construct_strides
from tilelang.utils.language import retrieve_stride


def test_construct_strides_row_major():
    assert construct_strides(()) == ()
    assert construct_strides((5,)) == (1,)
    assert construct_strides((2, 3)) == (3, 1)
    assert construct_strides((2, 3, 4)) == (12, 4, 1)
    # Accepts any sequence, not just tuples.
    assert construct_strides([2, 3, 4]) == (12, 4, 1)


def test_construct_strides_rank_is_preserved():
    for rank in range(5):
        shape = tuple(range(2, 2 + rank))
        assert len(construct_strides(shape)) == rank


def test_construct_strides_symbolic():
    n = tirx.Var("n", "int32")
    # A symbolic inner extent propagates into the outer strides; the innermost
    # stride is always a plain contiguous 1.
    strides = construct_strides((2, n))
    assert len(strides) == 2
    assert strides[1] == 1
    assert isinstance(strides[0], tirx.PrimExpr)
    assert tirx.analysis.expr_deep_equal(strides[0], n)

    # An outermost symbolic extent never reaches the strides at all.
    assert construct_strides((n, 4)) == (4, 1)


def test_construct_strides_disallow_prim_expr():
    n = tirx.Var("n", "int32")
    # Fully static shapes are unaffected by the flag.
    assert construct_strides((2, 3, 4), allow_prim_expr=False) == (12, 4, 1)
    assert construct_strides((), allow_prim_expr=False) == ()
    with pytest.raises(ValueError, match="allow_prim_expr"):
        construct_strides((2, n), allow_prim_expr=False)


def test_construct_strides_consumers_agree():
    """T.Tensor / OutTensor / retrieve_stride must not drift apart."""
    for shape in [(), (5,), (2, 3), (2, 3, 4)]:
        expected = construct_strides(shape)
        buf = tirx.decl_buffer(shape, "float32")
        assert tuple(retrieve_stride(buf)) == expected
        assert tuple(OutTensor(shape, T.float32).strides) == expected


@tilelang.testing.requires_cuda
def test_rank0_tensor_strides_match_shape_rank():
    """Regression: T.Tensor(()) used to yield strides=(1,) for a rank-0 shape."""

    @T.prim_func
    def main(S: T.Tensor((), "float32")):
        with T.Kernel(1, threads=1):
            S[()] = T.float32(1)

    for buf in main.buffer_map.values():
        assert len(buf.shape) == 0
        assert len(buf.strides) == len(buf.shape), f"rank-0 buffer got strides {buf.strides}"

    # T.Buffer already behaved correctly (it passes strides=None); keep them aligned.
    @T.prim_func
    def control(S: T.Buffer((), "float32")):
        with T.Kernel(1, threads=1):
            S[()] = T.float32(1)

    for buf in control.buffer_map.values():
        assert len(buf.strides) == len(buf.shape)


@tilelang.testing.requires_cuda
def test_rank0_tensor_kernel_runs():
    """A rank-0 T.Tensor output must compile and produce the right value."""

    @tilelang.jit
    def build(N):
        @T.prim_func
        def main(A: T.Tensor((N,), "float32"), S: T.Tensor((), "float32")):
            with T.Kernel(1, threads=1):
                acc = T.alloc_local((1,), "float32")
                acc[0] = T.float32(0)
                for i in T.serial(N):
                    acc[0] += A[i]
                S[()] = acc[0]

        return main

    kernel = build(8)
    a = torch.arange(8, dtype=torch.float32, device="cuda")
    s = torch.zeros((), dtype=torch.float32, device="cuda")
    kernel(a, s)
    torch.testing.assert_close(s, a.sum())


if __name__ == "__main__":
    tilelang.testing.main()
