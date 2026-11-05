"""Native argument plans preserve shape identity and runtime equality checks."""

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import S

from tilelang.jit.abi import FunctionABIPlan


def tensor(name, shape, *, strides=None, dtype="float32", offset=0):
    layout = tirx.TileLayout(S[shape:strides] + offset) if strides is not None else "default"
    return ir.Var(name, tirx.TensorType(shape, dtype, layout=layout))


def function(*params):
    return tirx.Function(params, [], ir.NoneType()).with_attr("global_symbol", "main")


def test_output_before_input_and_late_scalar_bind_linear_output():
    n = ir.Var("n", "int64")
    output = tensor("B", [2 * n + 1])
    source = tensor("A", [n])
    plan = FunctionABIPlan(function(output, source, n), [0])
    for extent in (1, 17, 257):
        array = torch.empty(extent)
        arguments, bindings = plan.bind_inputs([array, extent])
        assert arguments[0] is None and arguments[1] is array and arguments[2] == extent
        assert plan.output_shape(0, bindings) == (2 * extent + 1,)
    with pytest.raises(ValueError, match="conflicts"):
        plan.bind_inputs([torch.empty(17), 18])


def test_linear_input_decoder_matches_native_packed_binder():
    n = ir.Var("n", "int64")
    plan = FunctionABIPlan(function(tensor("B", [n]), tensor("A", [3 * n + 2])), [0])
    _, bindings = plan.bind_inputs([np.empty(17, dtype="float32")])
    assert plan.output_shape(0, bindings) == (5,)
    with pytest.raises(ValueError, match="divisible by 3"):
        plan.bind_inputs([np.empty(18, dtype="float32")])


@pytest.mark.parametrize("round_trip", [False, True])
def test_same_name_variables_keep_distinct_identity(round_trip):
    n, other_n = ir.Var("n", "int64"), ir.Var("n", "int64")
    source = function(tensor("B", [n, other_n]), tensor("A", [n]), tensor("C", [other_n]))
    if round_trip:
        source = ir.load_json(ir.save_json(source))
    plan = FunctionABIPlan(source, [0])
    _, bindings = plan.bind_inputs([torch.empty(3), torch.empty(7)])
    assert plan.output_shape(0, bindings) == (3, 7)
    assert len(bindings.values) == 2


def test_shape_and_stride_share_a_real_variable():
    n, stride = ir.Var("n", "int64"), ir.Var("stride", "int64")
    source = tensor("A", [n, 4], strides=[stride, 1])
    plan = FunctionABIPlan(function(tensor("B", [stride]), source), [0])
    view = torch.empty(3, 8)[:, ::2]
    with pytest.raises(ValueError, match=r"strides\[1\].*expected 1, got 2"):
        plan.bind_inputs([view])
    source = tensor("A", [n, 4], strides=[stride, 2])
    plan = FunctionABIPlan(function(tensor("B", [stride]), source), [0])
    _, bindings = plan.bind_inputs([view])
    assert plan.output_shape(0, bindings) == (8,)


def test_offset_view_shape_does_not_add_storage_offset_again():
    n = ir.Var("n", "int64")
    plan = FunctionABIPlan(function(tensor("B", [n]), tensor("A", [n])), [0])
    view = torch.arange(21, dtype=torch.float32)[4:17]
    arguments, bindings = plan.bind_inputs([view])
    assert arguments[1] is view
    assert plan.output_shape(0, bindings) == (13,)


@pytest.mark.parametrize("kind", ["torch", "numpy", "ffi"])
def test_subbyte_logical_shape_and_storage_divisibility(kind):
    n = ir.Var("n", "int64")
    plan = FunctionABIPlan(function(tensor("B", [n], dtype="float4_e2m1fn"), tensor("A", [n], dtype="float4_e2m1fn")), [0])
    value = torch.empty(8, dtype=torch.uint8) if kind == "torch" else np.empty(8, dtype="uint8")
    if kind == "ffi":
        value = tvm_ffi.from_dlpack(value)
    _, bindings = plan.bind_inputs([value])
    assert plan.output_shape(0, bindings) == (16,)
    assert plan.output_shape(0, bindings, storage_bits=8) == (8,)
    odd = FunctionABIPlan(function(tensor("B", [3], dtype="float4_e2m1fn")), [0])
    _, bindings = odd.bind_inputs([])
    with pytest.raises(ValueError, match="not divisible"):
        odd.output_shape(0, bindings, storage_bits=8)


def test_rank_zero_input_is_a_tensor_and_output_indices_are_validated():
    scalar = tensor("A", [])
    plan = FunctionABIPlan(function(scalar))
    arguments, _ = plan.bind_inputs([torch.tensor(3.0)])
    assert 0 in plan.tensors and arguments[0].ndim == 0
    with pytest.raises(ValueError, match="not a tensor"):
        FunctionABIPlan(function(ir.Var("n", "int64")), [0])
    with pytest.raises(ValueError, match="duplicate"):
        FunctionABIPlan(function(scalar), [0, -1])


def test_nonlinear_or_narrowing_only_binding_is_not_guessed():
    n = ir.Var("n", "uint32")
    plan = FunctionABIPlan(function(tensor("B", [n]), tensor("A", [ir.prim.Cast("uint8", n)])), [0])
    with pytest.raises(ValueError, match="Cannot resolve"):
        plan.bind_inputs([torch.empty(3)])


def test_boolean_storage_is_one_byte_per_logical_element():
    n = ir.Var("n", "int64")
    plan = FunctionABIPlan(function(tensor("B", [n], dtype="bool"), tensor("A", [n], dtype="bool")), [0])
    _, bindings = plan.bind_inputs([torch.empty(11, dtype=torch.bool)])
    assert plan.output_shape(0, bindings, storage_bits=8) == (11,)


@pytest.mark.parametrize("shape", [(3, 4), (0, 4)])
def test_ffi_tensor_strides_are_elements_and_offset_is_not_reapplied(shape):
    n, stride = ir.Var("n", "int64"), ir.Var("stride", "int64")
    source = tensor("A", [n, 4], strides=[stride, 2])
    plan = FunctionABIPlan(function(tensor("B", [stride]), source), [0])
    array = np.arange(40, dtype="float32").reshape(5, 8)[1 : 1 + shape[0], ::2]
    value = tvm_ffi.from_dlpack(array)
    assert value.strides == (8, 2)
    arguments, bindings = plan.bind_inputs([value])
    assert arguments[1] is value
    assert plan.output_shape(0, bindings) == (8,)
    assert bindings.evaluate(n, "n") == shape[0]
    wrong = function(tensor("A", [n, 4], strides=[stride, 1]))
    with pytest.raises(ValueError, match=r"strides\[1\].*expected 1, got 2"):
        FunctionABIPlan(wrong).bind_inputs([value])


def test_public_cpu_jit_accepts_runtime_tensor_inputs_and_outputs():
    import tilelang
    import tilelang.cpu.language as T
    from tvm_tirx import runtime

    @T.prim_func
    def add(A: T.Tensor((5,), "int32"), B: T.Tensor((5,), "int32")):
        for i in T.serial(5):
            B[i] = A[i] + 1

    target = {"kind": "c"}
    kernel = tilelang.compile(add, target=target, target_host=target, execution_backend="tvm_ffi")
    source = runtime.tensor(np.arange(5, dtype="int32"))
    destination = runtime.tensor(np.zeros(5, dtype="int32"))
    kernel(source, destination)
    np.testing.assert_array_equal(destination.numpy(), np.arange(5, dtype="int32") + 1)
