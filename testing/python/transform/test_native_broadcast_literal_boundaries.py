"""A cast is a hoistable literal only when its entire operand chain is literal."""

import pytest
import tilelang
import tilelang.language as T
from tvm_tirx import ir, tirx


@T.prim_func
def from_var(B: T.Tensor((4,), T.float8_e4m3fn), scalar: T.float32):
    B[T.Ramp(0, 1, 4)] = T.Broadcast(T.float8_e4m3fn(scalar), 4)


@T.prim_func
def from_load(A: T.Tensor((1,), T.float32), B: T.Tensor((4,), T.float8_e4m3fn)):
    B[T.Ramp(0, 1, 4)] = T.Broadcast(T.float8_e4m3fn(A[0]), 4)


@T.prim_func
def from_call(B: T.Tensor((4,), T.float8_e4m3fn), scalar: T.float32):
    B[T.Ramp(0, 1, 4)] = T.Broadcast(T.float8_e4m3fn(T.call_extern("float32", "opaque_float", scalar)), 4)


@pytest.mark.parametrize("function", [from_var, from_load, from_call], ids=["var", "load", "call"])
def test_nonliteral_cast_is_not_hoisted(function):
    original = function.with_attr("global_symbol", "main")
    result = tilelang.transform.HoistBroadcastValues()(ir.IRModule.from_expr(original))["main"]
    tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(original, result)
