"""CPU reduce op tests (target="c", execution_backend="cython").

Covers the 8 reduce variants supported by src/cpu/op/reduce.cc on local
buffers, which is the only path the frontend emits on CPU.
"""

import functools

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang import tvm

_FLOAT_OPS = ("sum", "max", "min", "abssum", "absmax")
_BIT_OPS = ("bitand", "bitor", "bitxor")


def _torch_dtype(dtype: str) -> torch.dtype:
    return getattr(torch, dtype)


def _ref_reduce(A: torch.Tensor, op: str, dim: int) -> torch.Tensor:
    legal = dim if dim >= 0 else A.ndim + dim
    if op == "sum":
        return A.sum(dim=legal)
    if op == "max":
        return A.amax(dim=legal)
    if op == "min":
        return A.amin(dim=legal)
    if op == "abssum":
        return A.abs().sum(dim=legal)
    if op == "absmax":
        return A.abs().amax(dim=legal)
    if op in _BIT_OPS:
        binop = {
            "bitand": torch.bitwise_and,
            "bitor": torch.bitwise_or,
            "bitxor": torch.bitwise_xor,
        }[op]
        return functools.reduce(
            lambda acc, i: binop(acc, A.select(legal, i)),
            range(1, A.shape[legal]),
            A.select(legal, 0).clone(),
        )
    raise ValueError(op)


def _emit_reduce(T, op: str, src, dst, dim: int, clear: bool):
    if op == "sum":
        T.reduce_sum(src, dst, dim=dim, clear=clear)
    elif op == "max":
        T.reduce_max(src, dst, dim=dim, clear=clear)
    elif op == "min":
        T.reduce_min(src, dst, dim=dim, clear=clear)
    elif op == "abssum":
        T.reduce_abssum(src, dst, dim=dim)
    elif op == "absmax":
        T.reduce_absmax(src, dst, dim=dim, clear=clear)
    elif op == "bitand":
        T.reduce_bitand(src, dst, dim=dim, clear=clear)
    elif op == "bitor":
        T.reduce_bitor(src, dst, dim=dim, clear=clear)
    elif op == "bitxor":
        T.reduce_bitxor(src, dst, dim=dim, clear=clear)
    else:
        raise ValueError(op)


def _make_kernel_2d(op: str, M: int, N: int, dim: int, dtype: str, clear: bool):
    legal = dim if dim >= 0 else 2 + dim
    dst_shape = (N,) if legal == 0 else (M,)

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
            for i in T.grid(dst_shape[0]):
                dst[i] = Init[i]
            _emit_reduce(T, op, src, dst, dim=legal, clear=clear)
            for i in T.grid(dst_shape[0]):
                B[i] = dst[i]

    return main


def _make_kernel_3d(op: str, P: int, Q: int, R: int, dim: int, dtype: str):
    legal = dim if dim >= 0 else 3 + dim
    if legal == 0:
        dst_shape = (Q, R)
    elif legal == 1:
        dst_shape = (P, R)
    else:
        dst_shape = (P, Q)

    @T.prim_func
    def main(
        A: T.Tensor((P, Q, R), dtype),
        B: T.Tensor(dst_shape, dtype),
    ):
        with T.Kernel(1):
            src = T.alloc_local((P, Q, R), dtype)
            dst = T.alloc_local(dst_shape, dtype)
            for p, q, r in T.grid(P, Q, R):
                src[p, q, r] = A[p, q, r]
            _emit_reduce(T, op, src, dst, dim=legal, clear=True)
            if legal == 0:
                for q, r in T.grid(Q, R):
                    B[q, r] = dst[q, r]
            elif legal == 1:
                for p, r in T.grid(P, R):
                    B[p, r] = dst[p, r]
            else:
                for p, q in T.grid(P, Q):
                    B[p, q] = dst[p, q]

    return main


def _compile_c(func):
    with tvm.target.Target("c"):
        return tilelang.compile(
            func,
            out_idx=[2],
            target="c",
            target_host="c",
            execution_backend="cython",
        )


def _make_input(shape, dtype, op):
    tdtype = _torch_dtype(dtype)
    gen = torch.Generator(device="cpu").manual_seed(7)
    if op in _BIT_OPS:
        return torch.randint(-50, 50, shape, dtype=tdtype, generator=gen)
    return torch.randn(shape, dtype=tdtype, generator=gen)


@pytest.mark.parametrize("op", _FLOAT_OPS)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cpu_reduce_2d_float(op, dim):
    M, N = 4, 8
    dtype = "float32"
    func = _make_kernel_2d(op, M, N, dim, dtype, clear=True)
    kernel = _compile_c(func)

    A = _make_input((M, N), dtype, op)
    Init = torch.zeros((N if (dim if dim >= 0 else 2 + dim) == 0 else M,), dtype=_torch_dtype(dtype))
    out = kernel(A, Init)
    expected = _ref_reduce(A, op, dim)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op", _BIT_OPS)
@pytest.mark.parametrize("dim", [0, 1])
def test_cpu_reduce_2d_bit(op, dim):
    M, N = 4, 8
    dtype = "int32"
    func = _make_kernel_2d(op, M, N, dim, dtype, clear=True)
    kernel = _compile_c(func)

    A = _make_input((M, N), dtype, op)
    dst_shape = (N,) if (dim if dim >= 0 else 2 + dim) == 0 else (M,)
    Init = torch.zeros(dst_shape, dtype=_torch_dtype(dtype))
    out = kernel(A, Init)
    expected = _ref_reduce(A, op, dim)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("op", ["sum", "max", "bitand"])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_cpu_reduce_3d(op, dim):
    P, Q, R = 2, 3, 4
    dtype = "float32" if op != "bitand" else "int32"
    func = _make_kernel_3d(op, P, Q, R, dim, dtype)
    with tvm.target.Target("c"):
        kernel = tilelang.compile(
            func,
            out_idx=[1],
            target="c",
            target_host="c",
            execution_backend="cython",
        )

    A = _make_input((P, Q, R), dtype, op)
    out = kernel(A)
    expected = _ref_reduce(A, op, dim)
    if op == "bitand":
        torch.testing.assert_close(out, expected)
    else:
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_cpu_reduce_clear_false_accumulates(op):
    """clear=False must accumulate onto the pre-existing dst values."""
    M, N = 4, 8
    dtype = "float32"
    func = _make_kernel_2d(op, M, N, dim=1, dtype=dtype, clear=False)
    kernel = _compile_c(func)

    A = _make_input((M, N), dtype, op)
    Init = torch.full((M,), 1.0, dtype=_torch_dtype(dtype))
    out = kernel(A, Init)

    base = _ref_reduce(A, op, dim=1)
    if op == "sum":
        expected = base + Init
    elif op == "max":
        expected = torch.maximum(base, Init)
    else:  # min
        expected = torch.minimum(base, Init)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_cpu_reduce_non_divisible_extent():
    """Reduce extent of 17 (prime) exercises the kSerial loop tail."""
    M, N = 3, 17
    dtype = "float32"
    func = _make_kernel_2d("sum", M, N, dim=1, dtype=dtype, clear=True)
    kernel = _compile_c(func)

    A = _make_input((M, N), dtype, "sum")
    Init = torch.zeros((M,), dtype=_torch_dtype(dtype))
    out = kernel(A, Init)
    expected = _ref_reduce(A, "sum", dim=1)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
