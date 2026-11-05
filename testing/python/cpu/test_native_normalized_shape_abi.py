"""Exact shape decoding consumes TileLang source defaults at both ABI boundaries."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
from tilelang.jit.abi import FunctionABIPlan
from tilelang.jit.adapter.abi import SourceABIPlan
from tilelang.testing.packed import compile_packed_function


def make_kernel(*, marked=True, affine=False):
    n = T.int32(is_size_var=marked)
    dimension = 2 * T.max(n, 0) + 1 if affine else T.max(n, 0)

    @T.prim_func
    def kernel(A: T.Tensor((dimension,), "int32"), B: T.Tensor((dimension,), "int32")):
        for i in T.serial(A.shape[0]):
            B[i] = A[i] + 3

    return kernel, n


def compile_kernel(function, consumer):
    if consumer == "tilelang_jit":
        return tilelang.compile(function, target="c", target_host="c", execution_backend="tvm_ffi")
    return compile_packed_function(function.with_attr("global_symbol", "main"))


@pytest.mark.parametrize("consumer", ["tilelang_packed", "tilelang_jit"])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("n", [0, 1, 7])
def test_normalized_carriers_execute_exact_numerical_oracle(consumer, affine, n):
    function, _ = make_kernel(affine=affine)
    executable = compile_kernel(function, consumer)
    extent = 2 * n + 1 if affine else n
    data = np.arange(extent, dtype="int32") * 5 - 4
    output = np.full(extent, -991, dtype="int32")
    executable(tvm_ffi.from_dlpack(data), tvm_ffi.from_dlpack(output))
    np.testing.assert_array_equal(output, data + 3)


@pytest.mark.parametrize("consumer", ["tilelang_packed", "tilelang_jit"])
def test_normalized_affine_carrier_keeps_divisibility_guard(consumer):
    function, _ = make_kernel(affine=True)
    executable = compile_kernel(function, consumer)
    data = tvm_ffi.from_dlpack(np.zeros(8, dtype="int32"))
    output = tvm_ffi.from_dlpack(np.zeros(8, dtype="int32"))
    with pytest.raises((ValueError, RuntimeError), match="divisible"):
        executable(data, output)


@pytest.mark.parametrize("consumer", ["tilelang_packed", "tilelang_jit"])
def test_unmarked_noninjective_shape_has_no_invented_inverse(consumer):
    function, _ = make_kernel(marked=False)
    with pytest.raises(Exception, match="(undefined|Undefined|not been defined)"):
        compile_kernel(function, consumer)


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("n", [0, 1, 7])
def test_caller_output_plan_decodes_same_original_variable(affine, n):
    function, variable = make_kernel(affine=affine)
    extent = 2 * n + 1 if affine else n
    plan = FunctionABIPlan(function, [1])
    arguments, bindings = plan.bind_inputs([np.zeros(extent, dtype="int32")])
    assert arguments[1] is None
    assert bindings.evaluate(variable, "captured dimension") == n
    assert plan.output_shape(1, bindings) == (extent,)


def test_caller_output_plan_does_not_solve_noninjective_unmarked_shape():
    function, _ = make_kernel(marked=False)
    with pytest.raises(ValueError, match="Cannot resolve"):
        FunctionABIPlan(function, [1]).bind_inputs([np.zeros(3, dtype="int32")])


def test_caller_output_plan_keeps_divisibility_and_cross_carrier_checks():
    function, _ = make_kernel(affine=True)
    with pytest.raises(ValueError, match="divisible"):
        FunctionABIPlan(function, [1]).bind_inputs([np.zeros(8, dtype="int32")])
    with pytest.raises(ValueError, match="(conflicts|expected)"):
        FunctionABIPlan(function).bind_inputs([np.zeros(7, dtype="int32"), np.zeros(9, dtype="int32")])


def test_explicit_scalar_value_precedes_default_for_python_binding():
    variable = T.int32(is_size_var=True)
    dimension = tirx.max(variable, 0)
    tensor = ir.Var("A", tirx.TensorType([dimension], "int32", layout=None))
    function = tvm_ffi.get_global_func("tl.AttachSizeVarDefaults")(tirx.Function([tensor, variable], []))
    _, bindings = FunctionABIPlan(function).bind_inputs([np.zeros(0, dtype="int32"), -2])
    assert bindings.evaluate(variable, "explicit scalar") == -2
    assert bindings.evaluate(dimension, "explicit dimension") == 0


@pytest.mark.parametrize("extent", [0, 1, 7])
def test_source_adapter_consumes_the_same_function_default_carrier(extent):
    function, variable = make_kernel(affine=True)
    arrays = [np.zeros(2 * extent + 1, dtype="int32") for _ in range(2)]
    bindings = SourceABIPlan(function).bind_arguments(arrays)
    assert bindings.evaluate(variable, "source adapter dimension") == extent


@pytest.mark.parametrize("consumer", ["tilelang_packed", "tilelang_jit"])
def test_explicit_negative_scalar_precedes_default_in_executable(consumer):
    variable = T.int32(is_size_var=True)
    tensor = ir.Var("A", tirx.TensorType([tirx.max(variable, 0)], "int32", align=4))
    output = ir.Var("B", tirx.TensorType([1], "int32", align=4))
    function = tirx.Function([tensor, output, variable], [ir.TensorStore(output, [0], variable)])
    function = tvm_ffi.get_global_func("tl.AttachSizeVarDefaults")(function.with_attr("global_symbol", "negative"))
    executable = compile_kernel(function, consumer)
    source = np.zeros(0, dtype="int32")
    result = np.full(1, -991, dtype="int32")
    executable(tvm_ffi.from_dlpack(source), tvm_ffi.from_dlpack(result), -2)
    np.testing.assert_array_equal(result, [-2])


def test_nullable_bare_carrier_precedes_affine_inverse_checks():
    n = T.int32(is_size_var=True)
    bare = ir.Var("A", tirx.TensorType([n], "int32", align=4))
    affine = ir.Var("B", tirx.TensorType([2 * tirx.max(n, 0) + 1], "int32", align=4))
    output = ir.Var("C", tirx.TensorType([1], "int32", align=4))
    function = tirx.Function([bare, affine, output], [ir.TensorStore(output, [0], n)])
    function = tvm_ffi.get_global_func("tl.AttachSizeVarDefaults")(function.with_attr("global_symbol", "nullable"))
    executable = compile_packed_function(function)
    result = np.full(1, -991, dtype="int32")
    direct = tvm_ffi.from_dlpack(np.zeros(3, dtype="int32"))
    valid = tvm_ffi.from_dlpack(np.zeros(7, dtype="int32"))
    invalid = tvm_ffi.from_dlpack(np.zeros(8, dtype="int32"))
    out = tvm_ffi.from_dlpack(result)
    executable(direct, valid, out)
    np.testing.assert_array_equal(result, [3])
    executable(None, valid, out)
    np.testing.assert_array_equal(result, [3])
    # A live higher-priority bare carrier supplies n. Check the original B
    # shape without applying the unused affine inverse's divisibility rule.
    result.fill(-991)
    with pytest.raises(RuntimeError, match=r"Invalid B\.shape\[0\] on argument #1") as error:
        executable(direct, invalid, out)
    assert "divisible" not in str(error.value)
    np.testing.assert_array_equal(result, [-991])
    with pytest.raises(RuntimeError, match="divisible"):
        executable(None, invalid, out)
    np.testing.assert_array_equal(result, [-991])
    with pytest.raises(RuntimeError, match="non-null"):
        executable(None, None, out)
    np.testing.assert_array_equal(result, [-991])
