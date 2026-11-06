"""Explicit T.ret constructs native control flow and evaluates exactly once."""

from tilelang.language import _construction
from tvm_tirx.script.ir_builder import base as _builder_base

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tilelang.testing.packed import compile_packed_function
from tvm_tirx.script.ir_builder import IRBuilder

import tilelang.language as T


def returns(function):
    result = []
    tvm_ffi.structural_walk(function, (ir.Return, result.append))
    return result


@pytest.mark.parametrize("value_type", [ir.PrimType("int32"), ir.PrimType("float32"), ir.PrimType("float32x4"), ir.PtrType("void")])
def test_ret_constructs_native_statement_with_exact_operand_and_location(value_type):
    value = ir.Var("value", value_type)
    loc = ir.SourceLoc(ir.SourceName("return_input.py"), 7, 3, 7, 21)
    result = T.ret(value, span=loc)
    assert isinstance(result, ir.Return)
    assert result.value.same_as(value)
    assert result.loc.same_as(loc)
    assert tvm_ffi.structural_equal(ir.load_json(ir.save_json(result)), result, map_free_vars=True)


def test_ret_construction_does_not_emit_in_active_builder():
    with IRBuilder() as builder, _construction.prim_func():
        T.func_name("construct_only")
        pending = T.ret(17)
        T.evaluate(0)
    assert isinstance(pending, ir.Return)
    assert returns(builder.get()) == []


def test_evaluate_return_receipt_emits_one_statement():
    with IRBuilder() as builder, _construction.prim_func():
        T.func_name("emit_once")
        T.func_ret(ir.PrimType("int32"))
        pending = T.ret(17)
        receipt = T.evaluate(pending)
        assert isinstance(receipt, _builder_base.AlreadyEmitted)
        assert receipt.value.same_as(pending)
        assert T.evaluate(receipt) is receipt
    function = builder.get()
    assert len(returns(function)) == 1
    assert returns(function)[0].same_as(pending)
    assert tirx.analysis.check_well_formed(function)


@pytest.mark.parametrize("explicit_evaluate", [False, True])
def test_public_source_return_in_branch_is_native_control_flow(explicit_evaluate):
    def source(value: T.int32):
        T.func_ret(ir.PrimType("int32"))
        if value > 0:
            if explicit_evaluate:
                T.evaluate(T.ret(value))
            else:
                T.ret(value)
        T.ret(0)

    function = T.prim_func(source)
    statements = returns(function)
    assert len(statements) == 2
    assert statements[0].value.same_as(function.params[0])
    assert int(statements[1].value) == 0
    assert tirx.analysis.check_well_formed(function)
    assert all(isinstance(statement.loc, ir.SourceLoc) for statement in statements)


@pytest.mark.parametrize("value", [None, "name", (1, 2), ir.Var("tensor", tirx.TensorType((4,), "int32"))])
def test_ret_rejects_nonprimitive_values(value):
    with pytest.raises(TypeError):
        T.ret(value)


def test_return_statement_cannot_be_embedded_in_primitive_arithmetic():
    with pytest.raises(TypeError):
        ir.const(1) + T.ret(2)


def test_native_cpu_return_exits_the_branch_before_the_fallback():
    def source(value: T.int32):
        T.func_ret(ir.PrimType("int32"))
        if value > 0:
            T.ret(value + 7)
        T.ret(-19)

    function = T.prim_func(source)
    executable = compile_packed_function(function, tilelang_owned=False)
    assert executable(5) == 12
    assert executable(0) == -19
    assert executable(-3) == -19
