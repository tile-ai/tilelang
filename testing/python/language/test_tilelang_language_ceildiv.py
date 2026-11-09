import pytest

import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm


@tilelang.jit(out_idx=[-1])
def _ceildiv_kernel(a: int, b: int):
    @T.prim_func
    def ceildiv_kernel(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1) as _:
            A[0] = T.ceildiv(T.int32(a), T.int32(b))

    return ceildiv_kernel


@tilelang.jit(out_idx=[-1])
def _align_up_kernel(a: int, b: int):
    @T.prim_func
    def align_up_kernel(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1) as _:
            A[0] = T.align_up(T.int32(a), T.int32(b))

    return align_up_kernel


def run_ceildiv(a=128, b=32):
    kernel = _ceildiv_kernel(a, b)
    A = kernel()
    print(kernel.get_kernel_source())
    print(A)


def test_ceildiv():
    run_ceildiv(a=128, b=32)
    run_ceildiv(a=1, b=32)
    run_ceildiv(a=-1, b=32)
    run_ceildiv(a=-2, b=32)


def run_align_up(a=128, b=32):
    kernel = _align_up_kernel(a, b)
    A = kernel()
    print(kernel.get_kernel_source())
    print(A)


def test_align_up():
    run_align_up(a=128, b=32)
    run_align_up(a=1, b=32)
    run_align_up(a=-1, b=32)
    run_align_up(a=-2, b=32)


@tilelang.jit
def _ceildiv_kernel_dyn(b: int):
    @T.prim_func
    def ceildiv_kernel(A: T.Tensor((1,), T.int32), a: T.int32):
        with T.Kernel(1, threads=1) as _:
            A[0] = T.ceildiv(T.int32(a), T.int32(b))

    return ceildiv_kernel


@tilelang.jit
def _align_up_kernel_dyn(b: int):
    @T.prim_func
    def align_up_kernel(A: T.Tensor((1,), T.int32), a: T.int32):
        with T.Kernel(1, threads=1) as _:
            A[0] = T.align_up(T.int32(a), T.int32(b))

    return align_up_kernel


def run_ceildiv_dyn(a=128, b=32):
    kernel = _ceildiv_kernel_dyn(b)
    A = torch.empty((1,), dtype=torch.int32, device="cuda")
    kernel(A, a)
    print(kernel.get_kernel_source())
    print(A)


@tilelang.testing.requires_cuda
def test_ceildiv_dyn():
    run_ceildiv_dyn(a=128, b=32)
    run_ceildiv_dyn(a=1, b=32)
    run_ceildiv_dyn(a=-1, b=32)
    run_ceildiv_dyn(a=-2, b=32)


def run_align_up_dyn(a=128, b=32):
    kernel = _align_up_kernel_dyn(b)
    A = torch.empty((1,), dtype=torch.int32, device="cuda")
    kernel(A, a)
    print(kernel.get_kernel_source())
    print(A)


@tilelang.testing.requires_cuda
def test_align_up_dyn():
    run_align_up_dyn(a=128, b=32)
    run_align_up_dyn(a=1, b=32)
    run_align_up_dyn(a=-1, b=32)
    run_align_up_dyn(a=-2, b=32)


@pytest.mark.parametrize("a,b", [(7, -3), (7, -7), (-31, -2), (0, -3), (-7, 2), (7, 3), (2147483647, 2), (-2147483648, -3)])
def test_ceildiv_constant(a, b):
    result = tvm.arith.Analyzer().simplify(T.ceildiv(a, b))
    assert int(result) == -(-a // b)


@pytest.mark.parametrize("a,b", [(9, 2), ((1 << 32) - 1, 1), ((1 << 32) - 1, 2)])
def test_ceildiv_unsigned_constants(a, b):
    lhs = tvm.tirx.const(a, "uint32")
    rhs = tvm.tirx.const(b, "uint32")
    result = T.ceildiv(lhs, rhs)
    assert isinstance(result, tvm.tirx.IntImm)
    assert result.dtype == "uint32"
    assert int(result) == -(-a // b)


def test_ceildiv_vector_expression():
    lhs = tvm.tirx.Var("lhs", "int32x4")
    rhs = tvm.tirx.Var("rhs", "int32x4")
    assert str(T.ceildiv(lhs, rhs).dtype) == "int32x4"
    assert str(T.cdiv(lhs, rhs).dtype) == "int32x4"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype,bits", [("int32", 32), ("int64", 64)])
def test_ceildiv_dynamic_signed(dtype, bits):
    max_value = (1 << (bits - 1)) - 1
    min_value = -(1 << (bits - 1))
    a_values = [7, 7, 32, -31, -7, -7, 7, 0, max_value, max_value, min_value, min_value]
    b_values = [-3, -7, -2, -2, 2, -2, 3, -3, 1, 2, 1, -3]
    n = len(a_values)

    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(1, threads=n):
                i = T.get_thread_binding()
                C[i] = T.ceildiv(A[i], B[i])

        return main

    a = torch.tensor(a_values, dtype=getattr(torch, dtype), device="cuda")
    b = torch.tensor(b_values, dtype=getattr(torch, dtype), device="cuda")
    expected = [-(-x // y) for x, y in zip(a_values, b_values)]
    assert kernel()(a, b).cpu().tolist() == expected


@tilelang.testing.requires_cuda
def test_ceildiv_call_operands_are_evaluated_once():
    @tilelang.jit(out_idx=[-1])
    def kernel():
        @T.prim_func
        def main(A: T.Tensor((2,), "int32"), C: T.Tensor((1,), "int32")):
            with T.Kernel(1, threads=1):
                C[0] = T.ceildiv(
                    T.call_extern("int32", "atomicAdd", T.address_of(A[0]), 1),
                    T.call_extern("int32", "atomicAdd", T.address_of(A[1]), 1),
                )

        return main

    values = torch.tensor([7, -10], dtype=torch.int32, device="cuda")
    result = kernel()(values)
    assert result.cpu().tolist() == [0]
    assert values.cpu().tolist() == [8, -9]


if __name__ == "__main__":
    tilelang.testing.main()
