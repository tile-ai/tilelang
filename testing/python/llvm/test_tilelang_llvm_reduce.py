"""LLVM reduce op tests (shares the cpu.Reduce registry with the `c` target).

Lightweight mirror of testing/python/cpu/test_tilelang_cpu_reduce.py covering
three representative ops (sum/max/bitand) + keepdim + clear=False, since the
`llvm` and `c` targets share the same CPU ReduceImpl (TargetIsCPU matches
kDLCPU). See src/cpu/op/reduce.cc.
"""

import functools

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _ref_reduce(A, op, dim):
    legal = dim if dim >= 0 else A.ndim + dim
    if op == "sum":
        return A.sum(dim=legal)
    if op == "max":
        return A.amax(dim=legal)
    if op == "bitand":
        return functools.reduce(
            lambda acc, i: torch.bitwise_and(acc, A.select(legal, i)),
            range(1, A.shape[legal]),
            A.select(legal, 0).clone(),
        )
    raise ValueError(op)


def _emit(T, op, src, dst, dim, clear):
    if op == "sum":
        T.reduce_sum(src, dst, dim=dim, clear=clear)
    elif op == "max":
        T.reduce_max(src, dst, dim=dim, clear=clear)
    elif op == "bitand":
        T.reduce_bitand(src, dst, dim=dim, clear=clear)
    else:
        raise ValueError(op)


@tilelang.testing.requires_llvm
@pytest.mark.parametrize("op,dtype", [("sum", "float32"), ("max", "float32"), ("bitand", "int32")])
def test_llvm_reduce_2d(op, dtype):
    M, N = 4, 8
    dim = 1
    dst_shape = (M,)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        Init: T.Tensor(dst_shape, dtype),
        B: T.Tensor(dst_shape, dtype),
    ):
        with T.Kernel(1):
            src = T.alloc_local((M, N), dtype)
            dst = T.alloc_local(dst_shape, dtype)
            for i, j in T.grid(M, N):
                src[i, j] = A[i, j]
            for i in T.grid(M):
                dst[i] = Init[i]
            _emit(T, op, src, dst, dim=dim, clear=True)
            for i in T.grid(M):
                B[i] = dst[i]

    with tvm.target.Target("llvm"):
        kernel = tilelang.compile(main, out_idx=[2], target="llvm", execution_backend="tvm_ffi")

    tdtype = getattr(torch, dtype)
    gen = torch.Generator(device="cpu").manual_seed(11)
    if op == "bitand":
        A = torch.randint(-50, 50, (M, N), dtype=tdtype, generator=gen)
    else:
        A = torch.randn((M, N), dtype=tdtype, generator=gen)
    Init = torch.zeros(dst_shape, dtype=tdtype)
    out = kernel(A, Init)
    expected = _ref_reduce(A, op, dim)
    if op == "bitand":
        torch.testing.assert_close(out, expected)
    else:
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@tilelang.testing.requires_llvm
def test_llvm_reduce_keepdim():
    """keepdim dst shape [M, 1] along dim=1."""
    M, N = 4, 8
    dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(1):
            src = T.alloc_local((M, N), dtype)
            dst = T.alloc_local((M, 1), dtype)
            for i, j in T.grid(M, N):
                src[i, j] = A[i, j]
            T.reduce_sum(src, dst, dim=1, clear=True)
            for i in T.grid(M):
                B[i, 0] = dst[i, 0]

    with tvm.target.Target("llvm"):
        kernel = tilelang.compile(main, out_idx=[1], target="llvm", execution_backend="tvm_ffi")

    A = torch.randn((M, N), dtype=torch.float32)
    out = kernel(A)
    expected = A.sum(dim=1, keepdim=True)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@tilelang.testing.requires_llvm
def test_llvm_reduce_clear_false_accumulates():
    M, N = 4, 8
    dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        Init: T.Tensor((M,), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(1):
            src = T.alloc_local((M, N), dtype)
            dst = T.alloc_local((M,), dtype)
            for i, j in T.grid(M, N):
                src[i, j] = A[i, j]
            for i in T.grid(M):
                dst[i] = Init[i]
            T.reduce_sum(src, dst, dim=1, clear=False)
            for i in T.grid(M):
                B[i] = dst[i]

    with tvm.target.Target("llvm"):
        kernel = tilelang.compile(main, out_idx=[2], target="llvm", execution_backend="tvm_ffi")

    A = torch.randn((M, N), dtype=torch.float32)
    Init = torch.full((M,), 1.0, dtype=torch.float32)
    out = kernel(A, Init)
    expected = A.sum(dim=1) + Init
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tilelang.testing.main()
