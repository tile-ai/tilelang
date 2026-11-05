"""Capture and lifetime contracts of Python tile-op implementation functions."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.language._native import buffer
from tilelang.tileop._lowering import close_lowered_function
from tilelang.testing.packed import compile_packed_function


def test_close_lowered_function_includes_type_only_shape_and_stride_captures():
    n = T.dynamic("n")
    stride = T.dynamic("stride")
    offset = T.dynamic("offset")
    a = ir.Var("A", buffer((n,), "float32", strides=(stride,), elem_offset=offset).as_type())
    out = ir.Var("out", T.Tensor((1,), "float32").as_type())

    @T.prim_func
    def lowering():
        out[0] = a[0]

    closed, captures = close_lowered_function(lowering)
    assert tirx.analysis.check_well_formed(closed)
    assert isinstance(closed.body, ir.SeqStmt)
    assert all(any(capture.same_as(expected) for capture in captures) for expected in (n, stride, offset, a, out)), (
        [value.name for value in captures],
        {value.name: any(capture.same_as(value) for capture in captures) for value in (n, stride, offset, a, out)},
    )
    a_index = next(i for i, capture in enumerate(captures) if capture.same_as(a))
    for expected in (n, stride, offset):
        assert next(i for i, capture in enumerate(captures) if capture.same_as(expected)) < a_index
    assert all(parameter.same_as(argument) for parameter, argument in zip(closed.params, captures, strict=True))


@pytest.mark.parametrize(
    "target",
    ["c", pytest.param("llvm", marks=pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM support is not built"))],
)
def test_close_lowered_function_retains_same_name_distinct_captures_and_executes(target):
    a = ir.Var("value", T.Tensor((8,), "int32").as_type())
    b = ir.Var("value", T.Tensor((8,), "int32").as_type())
    delta = ir.Var("value", "int32")

    @T.prim_func
    def lowering():
        for i in T.serial(8):
            b[i] = a[i] + delta

    closed, captures = close_lowered_function(lowering)
    assert len(captures) == 3
    assert len(set(captures)) == 3
    inputs = {
        a: tvm_tirx.runtime.tensor(np.arange(8, dtype="int32"), tvm_tirx.cpu()),
        b: tvm_tirx.runtime.tensor(np.zeros(8, dtype="int32"), tvm_tirx.cpu()),
        delta: 7,
    }
    executable = compile_packed_function(closed, tilelang_owned=False) if target == "c" else tirx.compile(closed, target="llvm")
    executable(*[inputs[capture] for capture in captures])
    np.testing.assert_array_equal(inputs[b].numpy(), np.arange(7, 15, dtype="int32"))


def test_closed_lowering_can_renew_allocations_without_changing_external_arguments():
    a = ir.Var("A", T.Tensor((4,), "int32").as_type())

    @T.prim_func
    def lowering():
        tmp = T.alloc_local((4,), "int32")
        for i in T.serial(4):
            tmp[i] = a[i]
            a[i] = tmp[i] + 1

    closed, captures = close_lowered_function(lowering)
    left, right = tirx.renew_def(closed), tirx.renew_def(closed)
    assert len(captures) == 1 and captures[0].same_as(a)
    assert not left.params[0].same_as(right.params[0])
    allocations = []
    for function in (left, right):
        local = []
        tvm_ffi.structural_walk(
            function,
            (
                ir.Bind,
                lambda node, local=local: (
                    local.append(node.var) if isinstance(node.value, ir.Call) and node.value.op.name == "tirx.alloc_tensor" else None
                ),
            ),
            order="pre",
        )
        assert len(local) == 1
        allocations.extend(local)
        assert tirx.analysis.check_well_formed(function)
    assert not allocations[0].same_as(allocations[1])


def test_local_allocation_type_dependencies_remain_captures():
    n = T.dynamic("n")

    @T.prim_func
    def lowering():
        tmp = T.alloc_local((n,), "int32")
        tmp[0] = 0

    closed, captures = close_lowered_function(lowering)
    assert len(captures) == 1 and captures[0].same_as(n)
    assert tirx.analysis.check_well_formed(closed)


def test_lowerer_rejects_explicit_parameters_and_nonvoid_return():
    parameter = ir.Var("value", "int32")
    with pytest.raises(ValueError, match="explicit parameters"):
        close_lowered_function(tirx.Function([parameter], [], ir.PrimType("void")))
    with pytest.raises(TypeError, match="return void"):
        close_lowered_function(tirx.Function([], [], ir.PrimType("int32")))
    # Native Function requires SeqStmt: an absent body is rejected by its
    # constructor before the TileLang lowerer can receive a declaration.
    with pytest.raises(TypeError, match="Expected `ir.SeqStmt`"):
        tirx.Function([], None, ir.PrimType("void"))
