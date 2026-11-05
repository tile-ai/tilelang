"""Keep TileLang tensor-slice errors separate from native IR and Python indexing."""

import numpy as np
import pytest

import tilelang.language as T
from tilelang.language.eager.ast import BaseBuilder, mutate


@pytest.mark.parametrize("form", ["load", "augment"])
@pytest.mark.parametrize("step", [1, 2, -1, 0])
def test_tensor_explicit_slice_step_keeps_runtime_error(form, step):
    with pytest.raises(RuntimeError, match=r"^Buffer slicing with step is not supported\.$"):

        @T.prim_func
        def kernel(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
            if form == "load":
                T.copy(A[0:4:step], B[0:4])
            else:
                A[0:4:step] += 1


def test_tensor_explicit_none_step_is_a_region():
    @T.prim_func
    def kernel(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
        T.copy(A[0:4:None], B[0:4:None])

    # Parsing succeeds without converting this region into a scalar load.
    assert len(kernel.params) == 2


@pytest.mark.parametrize("step", [None, 1, 2])
def test_tensor_store_slice_has_vector_index_semantics(step):
    import tilelang
    import torch

    stride = 1 if step is None else step

    @T.prim_func
    def kernel(A: T.Tensor((8,), "int32"), B: T.Tensor((8 // stride,), "int32")):
        A[0:8:step] = B[0 : 8 // stride]

    target = {"kind": "c"}
    executable = tilelang.compile(kernel, target=target, target_host=target, execution_backend="tvm_ffi")
    actual = torch.full((8,), -99, dtype=torch.int32)
    source = torch.arange(8 // stride, dtype=torch.int32) + 10
    expected = actual.clone()
    expected[0:8:step] = source
    executable(actual, source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class PythonBuilder(BaseBuilder):
    def set_fileline(self, filename, lineno, name):
        pass


@pytest.mark.parametrize("step", [None, 1, 2, -1, -2])
def test_plain_python_slicing_keeps_values_and_assignment(step):
    def function():
        values = list(range(8))
        first = values[::step]
        values[::step] = first
        grid = np.arange(12).reshape(3, 4)
        second = grid[::step, 1::2]
        grid[::step, 1::2] += 10
        return first, second, grid

    expected = function()
    actual = mutate(function).gen(PythonBuilder())()
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])


def test_user_getitem_evaluates_each_operand_once():
    events = []

    class Receiver:
        def __getitem__(self, index):
            events.append(("getitem", index.start, index.stop, index.step))
            return 37

    def receiver():
        events.append("receiver")
        return Receiver()

    def index():
        events.append("index")
        return slice(7, 1, -2)

    def function():
        return receiver()[index()]

    assert mutate(function).gen(PythonBuilder())() == 37
    assert events == ["receiver", "index", ("getitem", 7, 1, -2)]


def test_direct_native_subscript_keeps_native_error_type():
    from tvm_tirx import ir, tirx

    tensor = ir.Var("A", tirx.TensorType((8,), "int32", layout="default"))
    assert tensor[0:4:1].ty is not None
    with pytest.raises(ValueError, match="non-unit step"):
        tensor[0:4:2]
