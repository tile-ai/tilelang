"""Empty DLPack tensors have no addressed element and may carry arbitrary strides."""

import tvm_tirx

import numpy as np
import pytest
import torch
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.tirx.layout import S

import tilelang
import tilelang.cpu.language as C
from tilelang.jit.abi import FunctionABIPlan
from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_python_and_native_binder_agree_on_empty_strided_tensor(kind):
    source = ir.Var("source", tirx.TensorType((0, 4), "float32"))
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    function = tirx.Function([source, output], [ir.TensorStore(output, [0], 42)]).with_attr({"global_symbol": "main"})
    value = np.empty((0, 8), dtype="float32")[:, ::2] if kind == "numpy" else torch.empty((0, 8))[:, ::2]
    assert value.shape == (0, 4)
    compiled = compile_packed_function(function)
    actual = runtime.tensor(np.zeros(1, dtype="int32"))
    compiled(value, actual)
    np.testing.assert_array_equal(actual.numpy(), [42])

    # The Python plan should accept the same metadata before caller output
    # allocation. It must still obtain a concrete shape for the output tensor.
    plan = FunctionABIPlan(function, [1], tensor_error_type=RuntimeError)
    _, bindings = plan.bind_inputs([value])
    assert plan.output_shape(1, bindings) == (1,)


def test_empty_compact_tensor_still_binds_and_checks_every_shape():
    rows, columns = ir.Var("rows", "int64"), ir.Var("columns", "int64")
    source = ir.Var("source", tirx.TensorType((rows, columns), "float32"))
    output = ir.Var("output", tirx.TensorType((columns,), "float32"))
    function = tirx.Function([source, output, rows], [])
    plan = FunctionABIPlan(function, [1], tensor_error_type=RuntimeError)
    _, bindings = plan.bind_inputs([np.empty((0, 7), dtype="float32"), 0])
    assert plan.output_shape(1, bindings) == (7,)
    with pytest.raises(RuntimeError, match="conflicts"):
        plan.bind_inputs([np.empty((0, 7), dtype="float32"), 1])


@pytest.mark.parametrize("linear", [False, True])
def test_empty_explicit_layout_retains_symbolic_stride_provider(linear):
    stride = ir.Var("stride", "int64")
    outer_stride = 2 * stride if linear else stride
    source = ir.Var("source", tirx.TensorType((0, 4), "float32", layout=tirx.TileLayout(S[(0, 4) : (outer_stride, 2)])))
    output = ir.Var("output", tirx.TensorType((stride,), "int32"))
    function = tirx.Function([source, output], [ir.TensorStore(output, [0], 42)]).with_attr("global_symbol", "main")
    value = torch.empty((0, 8))[:, ::2]
    plan = FunctionABIPlan(function, [1], tensor_error_type=RuntimeError)
    _, bindings = plan.bind_inputs([value])
    expected_stride = 4 if linear else 8
    assert plan.output_shape(1, bindings) == (expected_stride,)
    compiled = compile_packed_function(function)
    actual = runtime.tensor(np.zeros(expected_stride, dtype="int32"))
    compiled(value, actual)
    assert actual.numpy()[0] == 42


@pytest.mark.parametrize("shape", [(0, 4), (1, 4), (2, 4)])
def test_explicit_layout_fixed_strides_remain_checked(shape):
    source = ir.Var("source", tirx.TensorType(shape, "float32", layout=tirx.TileLayout(S[shape : (4, 1)])))
    function = tirx.Function([source], [])
    plan = FunctionABIPlan(function, tensor_error_type=RuntimeError)
    value = torch.empty((shape[0], 8))[:, ::2]
    with pytest.raises(RuntimeError, match="strides"):
        plan.bind_inputs([value])


@pytest.mark.parametrize("kind", ["llvm", "c"])
@pytest.mark.parametrize("extent", [0, 1, 7])
@pytest.mark.parametrize("allocate", [False, True])
def test_dsl_compact_slices_preserve_empty_shapes_and_offset_addresses(kind, extent, allocate):
    if kind == "llvm" and not tvm_tirx.runtime.enabled("llvm"):
        pytest.skip("Optional LLVM code generation is disabled")
    size = C.dynamic("size", "int64")

    @C.prim_func
    def copy(A: C.Tensor((2, size), "float32"), B: C.Tensor((2, size), "float32")):
        with C.Kernel(1):
            for row in C.serial(2):
                for column in C.serial(size):
                    B[row, column] = A[row, column] + 1

    target = {"kind": kind, **({"mcpu": "generic"} if kind == "llvm" else {})}
    kernel = tilelang.compile(copy, target=target, target_host=target, execution_backend="tvm_ffi", out_idx=1 if allocate else None)
    # Compact storage may begin at an element offset inside the owning Torch
    # allocation. A zero-width tensor has no addressed stride constraint.
    backing = torch.arange(2 * extent + 2, dtype=torch.float32)
    value = backing[1 : 1 + 2 * extent].view(2, extent)
    if allocate:
        actual = kernel(value)
    else:
        storage = torch.full((2 * extent + 2,), -100, dtype=torch.float32)
        actual = storage[1 : 1 + 2 * extent].view(2, extent)
        kernel(value, actual)
        assert storage[0] == -100 and storage[-1] == -100
    assert actual.shape == (2, extent)
    torch.testing.assert_close(actual, value + 1, rtol=0, atol=0)


def test_explicit_parameter_alignment_remains_a_checked_promise():
    @C.prim_func
    def copy(A: C.Buffer((4,), "float32", align=16), B: C.Tensor((4,), "float32")):
        with C.Kernel(1):
            for i in C.Parallel(4):
                B[i] = A[i]

    kernel = tilelang.compile(copy, target="c", target_host="c", execution_backend="tvm_ffi")
    source = torch.arange(8, dtype=torch.float32)
    output = torch.empty(4, dtype=torch.float32)
    kernel(source[:4], output)
    torch.testing.assert_close(output, source[:4], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="alignment"):
        kernel(source[1:5], output)
