import numpy as np
import pytest
import tvm_ffi

from tvm_tirx import ir, runtime, tirx
from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def atomic(name, tensor, index, value=None, order=0, lanes=1):
    pointer = ir.Call("tl.access_ptr", [tensor.load(index), ir.const(lanes), ir.const(3)])
    arguments = [pointer] if value is None else [pointer, value]
    arguments.append(ir.const(order))
    result_type = f"{tensor.dtype}x{lanes}" if lanes != 1 else tensor.dtype
    return ir.Call(f"tl.atomic_{name}_elem_op", arguments, ty=result_type)


def lower(body, parameters, target="llvm"):
    function = tirx.Function(parameters, body).with_attr("global_symbol", "atomic_test")
    module = tvm_ffi.get_global_func("tl.cpu.transform.LowerCPUAtomics")()(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    calls = []
    stmt_functor.post_order_visit(module, lambda node: calls.append(node) if isinstance(node, ir.Call) else None)
    assert not any(isinstance(call.op, ir.Op) and call.op.name.startswith("tl.atomic_") for call in calls)
    assert module["main"].attrs["tl.cpu_had_atomics"]
    return tirx.compile(module, target=target)


@pytest.mark.parametrize("operation,expected", [("add", 12), ("max", 7), ("min", 5), ("or", 7), ("store", 7)])
@pytest.mark.parametrize("target", ["c", "llvm"])
def test_scalar_cpu_atomic_updates(operation, expected, target):
    tensor = TensorView.create("tensor", [1], "int32")
    executable = lower(ir.Evaluate(atomic(operation, tensor, 0, ir.const(7))), [tensor.tensor], target)
    actual = runtime.tensor(np.array([5], dtype="int32"))
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [expected])


@pytest.mark.parametrize("operation,expected", [("add_ret", 12), ("max_ret", 7), ("min_ret", 5)])
def test_returned_value_precedes_the_atomic_update(operation, expected):
    tensor = TensorView.create("tensor", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    executable = lower(output.store(0, atomic(operation, tensor, 0, ir.const(7))), [tensor.tensor, output.tensor])
    actual, previous = runtime.tensor(np.array([5], dtype="int32")), runtime.empty((1,), "int32")
    executable(actual, previous)
    np.testing.assert_array_equal(actual.numpy(), [expected])
    np.testing.assert_array_equal(previous.numpy(), [5])


def test_cpu_atomic_expression_preserves_left_to_right_read_snapshots():
    tensor = TensorView.create("tensor", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    value = tensor.load(0) + atomic("add_ret", tensor, 0, ir.const(3)) + atomic("add_ret", tensor, 0, ir.const(5))
    executable = lower(output.store(0, value), [tensor.tensor, output.tensor])
    actual, result = runtime.tensor(np.array([2], dtype="int32")), runtime.empty((1,), "int32")
    executable(actual, result)
    np.testing.assert_array_equal(actual.numpy(), [10])
    np.testing.assert_array_equal(result.numpy(), [9])


def test_atomic_load_and_nested_atomic_arguments():
    tensor = TensorView.create("tensor", [1], "int32")
    output = TensorView.create("output", [2], "int32")
    nested = atomic("add_ret", tensor, 0, atomic("add_ret", tensor, 0, ir.const(1)))
    body = [output.store(0, nested), output.store(1, atomic("load", tensor, 0))]
    executable = lower(body, [tensor.tensor, output.tensor])
    actual, result = runtime.tensor(np.array([2], dtype="int32")), runtime.empty((2,), "int32")
    executable(actual, result)
    np.testing.assert_array_equal(result.numpy(), [3, 5])


def test_lazy_conditional_only_executes_the_selected_atomic():
    flag = ir.Var("flag", "bool")
    tensor = TensorView.create("tensor", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    value = ir.Call(
        "prim.if_then_else", [flag, atomic("add_ret", tensor, 0, ir.const(3)), atomic("add_ret", tensor, 0, ir.const(9))], ty="int32"
    )
    executable = lower(output.store(0, value), [flag, tensor.tensor, output.tensor])
    for selected, expected in [(True, 5), (False, 11)]:
        actual, previous = runtime.tensor(np.array([2], dtype="int32")), runtime.empty((1,), "int32")
        executable(selected, actual, previous)
        np.testing.assert_array_equal(actual.numpy(), [expected])
        np.testing.assert_array_equal(previous.numpy(), [2])


def test_while_condition_atomic_executes_again_after_continue():
    counter = TensorView.create("counter", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    body = ir.If(counter.load(0) == 2, ir.Continue(), output.store(0, output.load(0) + 1))
    loop = ir.While(atomic("add_ret", counter, 0, ir.const(1)) < 3, body)
    executable = lower(loop, [counter.tensor, output.tensor])
    actual, result = runtime.tensor(np.array([0], dtype="int32")), runtime.tensor(np.array([0], dtype="int32"))
    executable(actual, result)
    np.testing.assert_array_equal(actual.numpy(), [4])
    np.testing.assert_array_equal(result.numpy(), [2])


@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("return_previous", [False, True])
def test_vector_atomic_values_and_returns(lanes, return_previous):
    tensor = TensorView.create("tensor", [lanes], "float32")
    source = TensorView.create("source", [lanes], "float32")
    output = TensorView.create("output", [lanes], "float32")
    value = source.load(ir.prim.Ramp(0, 1, lanes))
    operation = f"addx{lanes}" + ("_ret" if return_previous else "")
    call = atomic(operation, tensor, 0, value, lanes=lanes)
    body = output.store(ir.prim.Ramp(0, 1, lanes), call) if return_previous else ir.Evaluate(call)
    if return_previous:
        with pytest.raises(ValueError, match="CPU vector atomics do not support return_prev"):
            lower(body, [tensor.tensor, source.tensor, output.tensor])
        return
    executable = lower(body, [tensor.tensor, source.tensor, output.tensor])
    initial = np.arange(lanes, dtype="float32") + 2
    values = np.arange(lanes, dtype="float32") + 5
    actual, result = runtime.tensor(initial), runtime.empty((lanes,), "float32")
    executable(actual, runtime.tensor(values), result)
    np.testing.assert_array_equal(actual.numpy(), initial + values)


def test_vector_atomic_pointer_source_is_read_before_overlapping_updates():
    tensor = TensorView.create("tensor", [5], "float32")
    source = ir.Call("tl.access_ptr", [tensor.load(0), ir.const(4), ir.const(1)])
    executable = lower(ir.Evaluate(atomic("addx4", tensor, 1, source, lanes=4)), [tensor.tensor])
    actual = runtime.tensor(np.arange(1, 6, dtype="float32"))
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [1, 3, 5, 7, 9])


def test_atomic_destination_indices_are_captured_before_argument_updates():
    indices = TensorView.create("indices", [1], "int32")
    values = TensorView.create("values", [3], "int32")
    call = atomic("add", values, indices.load(0), atomic("add_ret", indices, 0, ir.const(1)))
    executable = lower(ir.Evaluate(call), [indices.tensor, values.tensor])
    index, actual = runtime.tensor(np.array([1], dtype="int32")), runtime.tensor(np.array([5, 7, 9], dtype="int32"))
    executable(index, actual)
    np.testing.assert_array_equal(actual.numpy(), [5, 8, 9])


def test_concurrent_scalar_atomic_lowering_is_rejected():
    tensor = TensorView.create("tensor", [1], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 2, 1, ir.ForKind.PARALLEL, ir.Evaluate(atomic("add", tensor, 0, ir.const(1))))
    with pytest.raises(ValueError, match="concurrent loop"):
        lower(loop, [tensor.tensor])
