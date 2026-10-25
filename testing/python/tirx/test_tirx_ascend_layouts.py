import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts")


@pytest.mark.parametrize("kind", ["major_k", "major_mn", "l0c", "sf"])
def test_fractal_layout_native_shape_kind_and_serialization(kind):
    tensor = TensorView.create("operand", [3, 32, 64], "float16")
    make = tvm_ffi.get_global_func(f"tl.make_ascend_{kind}_layout")
    layout = make(tensor.tensor, 1) if kind in ("major_k", "major_mn") else make(tensor.tensor)
    cloned = ir.load_json(ir.save_json(layout))
    ir.assert_structural_equal(layout, cloned)
    assert (
        tuple(int(value) for value in cloned(2, 17, 33))
        == {
            "major_k": (2, 2, 1, 1, 1),
            "major_mn": (2, 1, 2, 1, 1),
            "l0c": (2, 2, 1, 1, 1),
            "sf": (2, 1, 33, 1, 0),
        }[kind]
    )
    info = tvm_ffi.get_global_func("tl.try_extract_fractal_layout")(cloned, tensor.tensor)
    assert int(info["kind"]) == {"major_k": 0, "major_mn": 3, "l0c": 1, "sf": 2}[kind]
    assert int(info["rows"]) == 32 and int(info["cols"]) == 64


def test_native_strided_slice_preserves_existing_origin_and_executes(native_layouts):
    storage = TensorView.create("storage", [160], "int32")
    source = TensorView.create("source", [8, 16], "int32", strides=[20, 1], elem_offset=3)
    output = TensorView.create("output", [2, 4], "int32")
    layout = native_layouts.Layout([8, 16], lambda row, column: [row, column])
    declaration = tvm_ffi.get_global_func("tl.make_strided_slice")(source.tensor, layout, [ir.Range(2, 4), ir.Range(5, 9)])
    view = TensorView(ir.Var("slice", declaration.ty))
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    body = [
        source.declare(storage.data),
        ir.Bind(view.tensor, declaration),
        ir.For(
            row,
            0,
            2,
            1,
            ir.ForKind.DEFAULT,
            ir.For(column, 0, 4, 1, ir.ForKind.DEFAULT, output.store([row, column], view.load([row, column]))),
        ),
    ]
    function = tirx.Function([storage.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "slice_test")
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    executable = tirx.compile(function, target="c")
    values = np.arange(160, dtype="int32")
    actual = runtime.empty((2, 4), "int32")
    backing = runtime.tensor(values)
    executable(backing, actual)
    np.testing.assert_array_equal(actual.numpy(), values[np.arange(2)[:, None] * 20 + np.arange(4)[None, :] + 48])
