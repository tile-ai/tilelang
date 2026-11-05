"""C compute emits typed predicate lanes and evaluates vector operands once."""

import operator

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize("comparison", ["eq", "ne", "lt", "le", "gt", "ge"])
@pytest.mark.parametrize("dtype", ["int32", "uint32", "float32"])
def test_c_vector_predicate_preserves_dtype(comparison, dtype):
    lhs = ir.Var("lhs", tirx.TensorType([4], dtype))
    rhs = ir.Var("rhs", tirx.TensorType([4], dtype))
    output = ir.Var("output", tirx.TensorType([4], dtype))
    lanes = ir.prim.Ramp(0, 1, 4)
    a, b = ir.TensorLoad(lhs, [lanes]), ir.TensorLoad(rhs, [lanes])
    condition = ir.Var("condition", "boolx4")
    compare = getattr(operator, comparison)
    body = [ir.Bind(condition, compare(a, b)), ir.TensorStore(output, [lanes], ir.prim.Select(condition, a, b))]
    function = tirx.Function([lhs, rhs, output], body).with_attr("global_symbol", "main")
    executable = compile_packed_function(function, c_codegen="compute")
    if dtype == "uint32":
        left, right = [0, 1 << 31, (1 << 32) - 1, 7], [1, 7, (1 << 32) - 1, 1 << 31]
    elif dtype == "int32":
        left, right = [-(1 << 31), -1, (1 << 31) - 1, 7], [1, -1, -7, 8]
    else:
        left, right = [np.nan, -0.0, np.inf, -np.inf], [1.0, 0.0, np.inf, 2.0]
    left, right = np.asarray(left, dtype=dtype), np.asarray(right, dtype=dtype)
    expected = np.where(compare(left, right), left, right)
    actual = runtime.tensor(np.zeros(4, dtype=dtype))
    executable(runtime.tensor(left), runtime.tensor(right), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("uniform", [False, True])
def test_c_vector_select_evaluates_both_arms_once(uniform):
    state = ir.Var("state", tirx.TensorType([2], "float32"))
    output = ir.Var("output", tirx.TensorType([4], "float32"))
    flags = ir.Var("flags", tirx.TensorType([4], "int32"))
    lanes = ir.prim.Ramp(0, 1, 4)

    def take(index):
        pointer = ir.Call("tirx.address_of", [ir.TensorLoad(state, [index])], ty=ir.PtrType(ir.PrimType("float32")))
        return ir.prim.Broadcast(tirx.call_extern("select_take", pointer, ty="float32"), 4)

    condition = ir.TensorLoad(flags, [0]) > 0 if uniform else ir.TensorLoad(flags, [lanes]) > ir.prim.Broadcast(ir.const(0), 4)
    if not uniform:
        condition = ir.prim.Not(condition)
    selected = ir.prim.Select(condition, take(0), take(1))
    source = 'extern "C" float select_take(float* value) { float result = *value; *value += 1; return result; }'
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": "pragma_import_c", "value": source})
    body = ir.RegionStmt(
        "tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, [ir.TensorStore(output, [lanes], selected)]
    )
    function = tirx.Function([state, output, flags], body).with_attr("global_symbol", "main")
    executable = compile_packed_function(function, c_codegen="compute")
    for values in [[0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]]:
        state_value = runtime.tensor(np.array([10, 20], dtype="float32"))
        actual = runtime.tensor(np.zeros(4, dtype="float32"))
        executable(state_value, actual, runtime.tensor(np.array(values, dtype="int32")))
        np.testing.assert_array_equal(state_value.numpy(), np.array([11, 21], dtype="float32"))
        predicate = bool(values[0]) if uniform else ~np.array(values, dtype=bool)
        expected = np.where(predicate, 10, 20) + np.zeros(4)
        np.testing.assert_array_equal(actual.numpy(), expected.astype("float32"))
