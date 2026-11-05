"""Real native lowering/execution of partial asynchronous-copy fallbacks."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.transform as tr
from tilelang.testing.packed import compile_packed_function


def integer(value):
    return ir.IntImm("int64", value)


def access(tensor, start, count, mask, index_dtype="int64"):
    return ir.Call("tl.access_ptr", [ir.TensorLoad(tensor, [ir.IntImm(index_dtype, start)]), integer(count), ir.IntImm("int32", mask)])


def partial_copy(count, start, padding, predicate, unit, dest_extent=None, dest_dtype="float32", source_extent=3, index_dtype="int64"):
    source = ir.Var("source", tirx.TensorType([source_extent], "float32", data_alignment=4))
    dest = ir.Var("dest", tirx.TensorType([count if dest_extent is None else dest_extent], dest_dtype, data_alignment=4))
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"safe_value_map": {source: padding}})
    note = ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))
    name, amount = ("tl.ptx_cp_async", count) if unit == "elements" else ("tl.ptx_cp_async_bytes", count * 4)
    args = [access(dest, 0, count, 2), access(source, start, count, 1, index_dtype), integer(amount), ir.IntImm("bool", predicate)]
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"tag": "retained", "unit": unit})
    copy = ir.Call(
        name,
        [*args, payload],
        ty_args=[ir.PrimType("void")],
        attrs={
            "schema_version": 1,
            "operand_count": len(args),
            "annotation_schema": schema,
        },
    )
    function = tirx.Function([source, dest], [note, ir.Evaluate(copy)], ir.NoneType()).with_attr("global_symbol", "main")
    return tr.LegalizeSafeMemoryAccess()(ir.IRModule({"main": function}))


def calls(module):
    result = []
    tvm_ffi.structural_walk(module, (ir.Call, result.append), order="pre")
    return result


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("start", [-1, 2])
@pytest.mark.parametrize("padding", [0.0, -0.0, 3.0])
@pytest.mark.parametrize("predicate", [True, False])
@pytest.mark.parametrize("unit", ["elements", "bytes"])
def test_partial_transfer_preserves_valid_elements_and_predicate(count, start, padding, predicate, unit):
    module = partial_copy(count, start, padding, predicate, unit)
    assert tirx.analysis.check_well_formed(module)
    retained = next(call for call in calls(module) if call.op.name.startswith("tl.ptx_cp_async"))
    metadata = tvm_ffi.get_global_func("tl.DecodeAnnotations")(retained.args[-1], retained.attrs["annotation_schema"])
    assert dict(metadata) == {"tag": "retained", "unit": unit}
    assert int(retained.args[2]) == count * (4 if unit == "bytes" else 1)
    # Constants select the partial-copy branch. Compile and execute that real
    # generated IR through the C packed host; no mock implements the fallback or its padding.
    module = tirx.transform.StmtSimplify()(module)
    assert not any(call.op.name.startswith("tl.ptx_cp_async") for call in calls(module))
    module = tr.LowerAccessPtr()(module)
    executable = compile_packed_function(module, tilelang_owned=False)
    source = np.array([7, 11, 19], dtype="float32")
    output = np.full(count, 123, dtype="float32")
    executable(tvm_ffi.from_dlpack(source), tvm_ffi.from_dlpack(output))
    expected = (
        np.array([source[start + lane] if 0 <= start + lane < len(source) else padding for lane in range(count)], dtype="float32")
        if predicate
        else np.zeros(count, dtype="float32")
    )
    # Bitwise comparison distinguishes negative-zero padding and reinterpret.
    np.testing.assert_array_equal(output.view("uint32"), expected.view("uint32"))


def test_destination_tail_is_not_written():
    module = partial_copy(4, 2, 3.0, True, "elements", dest_extent=2)
    module = tr.LowerAccessPtr()(tirx.transform.StmtSimplify()(module))
    executable = compile_packed_function(module, tilelang_owned=False)
    source = np.array([7, 11, 19], dtype="float32")
    backing = np.full(6, -91, dtype="float32")
    executable(tvm_ffi.from_dlpack(source), tvm_ffi.from_dlpack(backing[2:4]))
    np.testing.assert_array_equal(backing, [-91, -91, 19, 3, -91, -91])


def test_equal_width_cross_dtype_copy_preserves_bits():
    module = partial_copy(2, 2, -0.0, True, "elements", dest_dtype="int32")
    module = tr.LowerAccessPtr()(tirx.transform.StmtSimplify()(module))
    executable = compile_packed_function(module, tilelang_owned=False)
    source = np.array([7, 11, 19], dtype="float32")
    output = np.zeros(2, dtype="int32")
    executable(tvm_ffi.from_dlpack(source), tvm_ffi.from_dlpack(output))
    expected = np.array([19, -0.0], dtype="float32").view("int32")
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize(
    "dtype,start,size,expected",
    [
        ("uint8", 255, 256, [255, -3]),
        ("int8", 127, 128, [127, -3]),
        ("uint32", (1 << 32) - 1, 3, [-3, -3]),
        ("uint64", (1 << 64) - 1, 3, [-3, -3]),
        ("int64", (1 << 63) - 1, 3, [-3, -3]),
    ],
)
def test_transfer_index_does_not_wrap_a_tail_back_to_element_zero(dtype, start, size, expected):
    module = partial_copy(2, start, -3.0, True, "elements", source_extent=size, index_dtype=dtype)
    module = tr.LowerAccessPtr()(tirx.transform.StmtSimplify()(module))
    assert not any(call.op.name.startswith("tl.ptx_cp_async") for call in calls(module))
    executable = compile_packed_function(module, tilelang_owned=False)
    values = np.arange(size, dtype="float32")
    output = np.zeros(2, dtype="float32")
    executable(tvm_ffi.from_dlpack(values), tvm_ffi.from_dlpack(output))
    np.testing.assert_array_equal(output, expected)
