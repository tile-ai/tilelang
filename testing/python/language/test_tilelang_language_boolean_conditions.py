import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tvm import IRModule, tirx
from tvm.ir import Op


DTYPES = ["int8", "uint8", "int32", "int64", "float32", "bool"]


def condition_kernel(dtype):
    def main(A: T.Tensor((8,), dtype), B: T.Tensor((14, 8), "int32")):
        with T.Kernel(1, threads=8):
            i = T.get_thread_binding()
            if A[i]:
                B[0, i] = 1
            else:
                B[0, i] = 0

            B[1, i] = 0
            if A[i]:
                B[1, i] = 1

            if not A[i]:
                B[2, i] = 1
            else:
                B[2, i] = 0

            B[3, i] = 1 if A[i] else 0

            B[4, i] = 0
            remaining = T.alloc_var(dtype, A[i])
            while remaining:
                B[4, i] = 1
                remaining = 0

            if A[i] != 0:
                B[5, i] = 1
            else:
                B[5, i] = 0

            j = (i + 1) % 8
            if A[i] and A[j]:
                B[6, i] = 1
            else:
                B[6, i] = 0

            if (A[i] != 0) and A[j]:
                B[7, i] = 1
            else:
                B[7, i] = 0

            if A[i] and (A[j] != 0):
                B[8, i] = 1
            else:
                B[8, i] = 0

            if A[i] or A[j]:
                B[9, i] = 1
            else:
                B[9, i] = 0

            if (A[i] != 0) or A[j]:
                B[10, i] = 1
            else:
                B[10, i] = 0

            if A[i] or (A[j] != 0):
                B[11, i] = 1
            else:
                B[11, i] = 0

            B[12, i] = T.if_then_else(A[i] and A[j], 1, 0)
            if not (A[i] and A[j]):
                B[13, i] = 1
            else:
                B[13, i] = 0

    return main


def make_prim_func(func, frontend, *args):
    if frontend == "prim_func":
        return T.prim_func(func)
    return tilelang.jit(func).get_tir(*args)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("frontend", ["prim_func", "jit"])
def test_numeric_conditions_ir(dtype, frontend):
    func = make_prim_func(condition_kernel(dtype), frontend, T.Tensor((8,), dtype), T.Tensor((14, 8), "int32"))
    conditions = []

    def visit(node):
        if isinstance(node, (tirx.IfThenElse, tirx.While)):
            conditions.append(node.condition)
        elif isinstance(node, tirx.Call) and node.op.same_as(Op.get("tirx.if_then_else")):
            conditions.append(node.args[0])

    tirx.stmt_functor.post_order_visit(func.body, visit)
    assert len(conditions) == 14
    assert all(cond.dtype == "bool" for cond in conditions)

    # The original if/else crashed in the negative-index analyzer. The
    # then-only form also needs a boolean predicate during buffer flattening.
    mod = tilelang.transform.LegalizeNegativeIndex()(IRModule({"main": func}))
    tilelang.transform.FlattenBuffer()(mod)


@pytest.mark.parametrize("dtype", ["int32", "float32", "bool"])
@pytest.mark.parametrize("frontend", ["prim_func", "jit"])
def test_numeric_assert_ir(dtype, frontend):
    def main(A: T.Tensor((8,), dtype), B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=8):
            i = T.get_thread_binding()
            assert A[i], "Expected a nonzero value"
            B[i] = 1

    func = make_prim_func(main, frontend, T.Tensor((8,), dtype), T.Tensor((8,), "int32"))
    assertions = []
    tirx.stmt_functor.post_order_visit(func.body, lambda node: assertions.append(node) if isinstance(node, tirx.AssertStmt) else None)
    assert len(assertions) == 1
    assert assertions[0].condition.dtype == "bool"
    tilelang.transform.LegalizeNegativeIndex()(IRModule({"main": func}))


@pytest.mark.parametrize("condition", [0, 2, -1, 0.0, 0.5, False, True])
def test_compile_time_conditions(condition):
    @T.prim_func
    def main(B: T.Tensor((1,), "int32")):
        if condition:
            B[0] = 1
        else:
            B[0] = 0

    assert isinstance(main.body, tirx.BufferStore)
    assert int(main.body.value) == int(bool(condition))


def test_compile_time_boolop_short_circuit():
    false_condition, true_condition = False, True

    def unexpected_rhs():
        raise AssertionError("A short-circuited RHS must not be evaluated")

    @T.prim_func
    def main(B: T.Tensor((2,), "int32")):
        if false_condition and unexpected_rhs():
            B[0] = -1
        else:
            B[0] = 0
        if true_condition or unexpected_rhs():
            B[1] = 1
        else:
            B[1] = -1

    stores = []
    tirx.stmt_functor.post_order_visit(main.body, lambda node: stores.append(node) if isinstance(node, tirx.BufferStore) else None)
    assert [int(store.value) for store in stores] == [0, 1]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("frontend", ["prim_func", "jit"])
def test_numeric_conditions_cuda(dtype, frontend):
    if dtype == "float32":
        values = [0.0, -0.0, 0.5, -0.5, float("nan"), float("inf"), -float("inf"), 2.0]
    elif dtype == "uint8":
        values = [0, 1, 2, 127, 128, 255, 0, 4]
    elif dtype == "int8":
        values = [-128, -2, -1, 0, 1, 2, 127, 0]
    elif dtype == "int64":
        values = [0, 1, -1, 2**32, -(2**32), 2**40, -(2**40), 0]
    else:
        values = [0, 1, 2, -1, 0, 4, -3, 0]

    a = torch.tensor(values, device="cuda", dtype=getattr(torch, dtype))
    b = torch.full((14, 8), -9999, device="cuda", dtype=torch.int32)
    func = condition_kernel(dtype)
    kernel = tilelang.compile(T.prim_func(func), target="cuda") if frontend == "prim_func" else tilelang.jit(func, target="cuda")
    kernel(a, b)

    nonzero = (a != 0).to(torch.int32)
    next_nonzero = torch.roll(nonzero, shifts=-1)
    both = nonzero & next_nonzero
    either = nonzero | next_nonzero
    expected = torch.stack(
        [nonzero, nonzero, 1 - nonzero, nonzero, nonzero, nonzero, both, both, both, either, either, either, both, 1 - both]
    )
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
