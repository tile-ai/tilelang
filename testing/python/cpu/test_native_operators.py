"""Real TileLang tile operators through native inference, lowering and C.

The test-only C++ bridge calls the production operator virtual methods. The
owning LowerTileOp pass and the complete public DSL/JIT are tested separately.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.testing.packed import compile_packed_function


def _integer(value, dtype="int64"):
    return ir.IntImm(dtype, value)


def _tensor(name, shape, dtype="float32", scope="global"):
    return ir.Var(
        name,
        tirx.TensorType(
            [value if isinstance(value, ir.Expr) else _integer(value) for value in shape],
            dtype,
            scope=scope,
            layout="default",
        ),
    )


def _region(tensor, begins, extents):
    ranges = []
    for begin, extent, dimension in zip(begins, extents, tensor.ty.shape):
        dtype = dimension.ty.dtype
        begin = begin if isinstance(begin, ir.Expr) else _integer(begin, dtype)
        extent = extent if isinstance(extent, ir.Expr) else _integer(extent, dtype)
        ranges.append(ir.Range.from_min_extent(begin, extent))
    return ir.TensorRegion(tensor, ranges, ir.TensorRegionType())


def _call(name, *operands, hints=None):
    hints = hints or {}
    schema = ["map"]
    for key, value in hints.items():
        schema.append([["str", key], ["bool" if isinstance(value, bool) else "int", value]])
    call = ir.Call(
        "tl.tileop." + name,
        [*operands, ir.Tuple([])],
        attrs={"schema_version": 1, "operand_count": len(operands), "annotation_schema": schema},
    )
    call.validate()
    return call


def _lower(call):
    target = tvm_tirx.target.Target("c")
    with target:
        result = tvm_ffi.get_global_func("tl.testing.LowerNativeOperator")(call, target, {})
    return result


def _alloc(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _compile(name, params, statements):
    function = tirx.Function(params, statements, ir.NoneType()).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)
    artifact_root = os.environ.get("TILELANG_OPS_ARTIFACT_DIR")
    if artifact_root:
        path = Path(artifact_root)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(ir.save_json(function))
        (path / (name + ".py")).write_text(function.script())
    executable = compile_packed_function(function, c_codegen="compute")
    if artifact_root:
        (Path(artifact_root) / (name + ".cc")).write_text(executable.mod.inspect_source("c"))
    return executable


def _run(executable, *inputs):
    arguments = [tvm_tirx.runtime.tensor(x) if isinstance(x, np.ndarray) else x for x in inputs]
    executable(*arguments)
    return [x.numpy() if hasattr(x, "numpy") else x for x in arguments]


@pytest.mark.parametrize("extent", [1, 17, 257])
def test_copy_dynamic(extent):
    n = ir.Var("n", "int64")
    source, destination = _tensor("A", [n]), _tensor("B", [n])
    lowered = _lower(_call("copy", source, destination))
    assert len(lowered["reads"]) == len(lowered["writes"]) == 1
    executable = _compile("copy_dynamic_" + str(extent), [n, source, destination], [lowered["statement"]])
    values = np.arange(extent, dtype="float32") - np.float32(3.5)
    result = _run(executable, extent, values, np.full(extent, -99, dtype="float32"))[-1]
    np.testing.assert_array_equal(result, values)


def test_copy_regions_and_dtype_conversion():
    source = _tensor("A", [13], "float32")
    destination = _tensor("B", [17], "int32")
    lowered = _lower(_call("copy", _region(source, [3], [7]), _region(destination, [5], [7])))
    executable = _compile("copy_region_cast", [source, destination], [lowered["statement"]])
    values = np.arange(13, dtype="float32") + np.float32(0.25)
    initial = np.full(17, -99, dtype="int32")
    expected = initial.copy()
    expected[5:12] = values[3:10].astype("int32")
    result = _run(executable, values, initial)[-1]
    np.testing.assert_array_equal(result, expected)


def test_copy_out_of_bounds_load_zero_fills():
    source, destination = _tensor("A", [5]), _tensor("B", [9])
    lowered = _lower(_call("copy", _region(source, [-2], [9]), destination))
    executable = _compile("copy_oob_zero", [source, destination], [lowered["statement"]])
    values = np.arange(5, dtype="float32") + 1
    result = _run(executable, values, np.full(9, -99, dtype="float32"))[-1]
    np.testing.assert_array_equal(result, np.pad(values, (2, 2)))


@pytest.mark.parametrize("shape", [[], [17], [3, 5]])
def test_fill_full_and_scalar(shape):
    destination = _tensor("B", shape, "int32")
    lowered = _lower(_call("fill", destination, _integer(7, "int32")))
    assert not lowered["reads"] and len(lowered["writes"]) == 1
    executable = _compile("fill_" + str(len(shape)), [destination], [lowered["statement"]])
    initial = np.zeros(shape, dtype="int32")
    result = _run(executable, initial)[0]
    np.testing.assert_array_equal(result, np.full(shape, 7, dtype="int32"))


def test_fill_slice_does_not_touch_neighbors():
    destination = _tensor("B", [4, 7])
    lowered = _lower(_call("fill", _region(destination, [1, 2], [2, 3]), ir.FloatImm("float32", 2.5)))
    executable = _compile("fill_slice", [destination], [lowered["statement"]])
    initial = np.arange(28, dtype="float32").reshape(4, 7)
    expected = initial.copy()
    expected[1:3, 2:5] = 2.5
    np.testing.assert_array_equal(_run(executable, initial)[0], expected)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("kind", ["sum", "max", "min", "abssum", "absmax"])
def test_reduce_local_copy_chain(dim, kind):
    shape = [3, 5]
    output_shape = [shape[1 - dim]]
    source, output = _tensor("A", shape), _tensor("B", output_shape)
    local = _tensor("local", shape, scope="local")
    reduced = _tensor("reduced", output_shape, scope="local")
    reduction = _lower(_call("reduce", local, reduced, ir.StringImm(kind), _integer(dim, "int32"), _integer(1, "bool")))
    assert len(reduction["reads"]) == 1 and len(reduction["writes"]) == 1
    statements = [
        _alloc(local),
        _alloc(reduced),
        _lower(_call("copy", source, local))["statement"],
        reduction["statement"],
        _lower(_call("copy", reduced, output))["statement"],
    ]
    executable = _compile(f"reduce_{kind}_{dim}", [source, output], statements)
    values = (np.arange(15, dtype="float32") - 8).reshape(shape)
    oracle_input = np.abs(values) if kind.startswith("abs") else values
    oracle = {"sum": np.sum, "max": np.max, "min": np.min, "abssum": np.sum, "absmax": np.max}[kind]
    expected = oracle(oracle_input, axis=dim)
    actual = _run(executable, values, np.zeros(output_shape, dtype="float32"))[-1]
    np.testing.assert_array_equal(actual, expected)


def test_reduce_clear_false_and_keepdim():
    source, output = _tensor("A", [3, 5]), _tensor("B", [3, 1])
    local, reduced = _tensor("local", [3, 5], scope="local"), _tensor("reduced", [3, 1], scope="local")
    reduction = _lower(_call("reduce", local, reduced, ir.StringImm("sum"), _integer(1, "int32"), _integer(0, "bool")))
    assert len(reduction["reads"]) == 2
    statements = [
        _alloc(local),
        _alloc(reduced),
        _lower(_call("copy", source, local))["statement"],
        _lower(_call("fill", reduced, ir.FloatImm("float32", 7)))["statement"],
        reduction["statement"],
        _lower(_call("copy", reduced, output))["statement"],
    ]
    executable = _compile("reduce_no_clear_keepdim", [source, output], statements)
    values = np.arange(15, dtype="float32").reshape(3, 5)
    result = _run(executable, values, np.zeros((3, 1), dtype="float32"))[-1]
    np.testing.assert_array_equal(result, np.sum(values, axis=1, keepdims=True) + 7)


@pytest.mark.parametrize("begins,extents", [([-1], [3]), ([4], [5]), ([0], [-1])])
def test_fill_rejects_invalid_regions(begins, extents):
    destination = _tensor("B", [8])
    with pytest.raises((ValueError, RuntimeError)):
        _lower(_call("fill", _region(destination, begins, extents), ir.FloatImm("float32", 1)))


def test_reduce_rejects_shape_and_scope():
    source, output = _tensor("A", [3, 5]), _tensor("B", [3])
    with pytest.raises((ValueError, RuntimeError), match="local"):
        _lower(_call("reduce", source, output, ir.StringImm("sum"), _integer(1, "int32"), _integer(1, "bool")))
    local, wrong = _tensor("local", [3, 5], scope="local"), _tensor("wrong", [4], scope="local")
    with pytest.raises((ValueError, RuntimeError), match="shape mismatch"):
        _lower(_call("reduce", local, wrong, ir.StringImm("sum"), _integer(1, "int32"), _integer(1, "bool")))


def test_ptr_access_retains_pointer_interval_and_rejects_wrong_type():
    ty = ir.PtrType(ir.PrimType("float32"), "global")
    pointer = ir.Var("pointer", ty)
    call = ir.Call("tl.ptr_access", [pointer, _integer(3), _integer(7), _integer(3, "int32")])
    call.validate()
    assert tvm_ffi.structural_equal(call.ty, ty)
    assert tuple(int(x) for x in call.args[1:]) == (3, 7, 3)
    for mask in [-1, 4]:
        invalid = ir.Call("tl.ptr_access", [pointer, _integer(0), _integer(1), _integer(mask, "int32")])
        with pytest.raises(ValueError, match="rw_mask"):
            invalid.validate()
    invalid = ir.Call("tl.ptr_access", call.args, ty=ir.PtrType(ir.PrimType("int32"), "global"))
    with pytest.raises(TypeError, match="preserve pointer"):
        invalid.validate()
