"""Source Bind spelling and signed-width literals stay native and executable."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

import tilelang
import tilelang.language as T
from tilelang.language import _native


@pytest.mark.parametrize(
    "value,dtype", [(0, "int32"), ((1 << 31) - 1, "int32"), (1 << 31, "int64"), (1 << 32, "int64"), ((1 << 63) - 1, "int64")]
)
def test_tensor_shape_literals_choose_the_source_signed_width(value, dtype):
    annotation = T.Tensor((value,), "float16")
    assert annotation.shape[0].ty == ir.PrimType(dtype)
    assert int(annotation.shape[0].value) == value
    assert int(annotation.strides[0].value) == 1
    assert annotation.strides[0].ty == ir.PrimType(dtype)


@pytest.mark.parametrize("value,dtype", [(-(1 << 31), "int32"), (-(1 << 31) - 1, "int64"), (-(1 << 63), "int64"), (True, "bool")])
def test_loop_literal_conversion_does_not_narrow_or_retype_bool(value, dtype):
    expression = _native._expr(value)
    assert expression.ty == ir.PrimType(dtype)
    assert int(expression.value) == int(value)


def test_callable_let_statement_keeps_explicit_var_and_native_execution():
    @T.prim_func
    def increment(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        for i in T.serial(4):
            temporary = T.int32()
            T.let(temporary, A[i] + 3)
            B[i] = temporary

    bindings = []
    stores = []
    tvm_ffi.structural_map(increment, (ir.Bind, lambda binding: bindings.append(binding) or binding))
    tvm_ffi.structural_map(increment, (ir.TensorStore, lambda store: stores.append(store) or store))
    assert len(bindings) == 1
    assert stores[0].value.same_as(bindings[0].var)
    assert tirx.analysis.check_well_formed(increment)
    function = tilelang.compile(increment, target="c", execution_backend="tvm_ffi")
    source = np.array([-7, 0, 4, 123], dtype="int32")
    output = np.full(4, -99, dtype="int32")
    function(source, output)
    np.testing.assert_array_equal(output, source + 3)


def test_callable_let_expression_has_a_separate_expression_scope():
    value = ir.Var("value", "int32")
    with pytest.warns(DeprecationWarning, match="T.Let"):
        expression = T.let(value, 5, value * 3)
    assert isinstance(expression, ir.prim.Let)
    assert expression.var.same_as(value)
    assert int(expression.value.value) == 5
    assert int(sym.Analyzer().simplify(expression).value) == 15
