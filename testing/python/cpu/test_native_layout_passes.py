"""Production LayoutInference and LowerTileOp compose over one native IR graph."""

import os
from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.testing.packed import compile_packed_function


def walk(value, callback):
    def visit(obj):
        callback(obj)
        return tvm_ffi.WalkResult.ADVANCE

    return tvm_ffi.structural_walk(value, (object, visit), order="post")


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, shape, scope="global", dtype="float32", layout="default"):
    return ir.Var(name, tirx.TensorType([integer(x) if isinstance(x, int) else x for x in shape], dtype, scope=scope, layout=layout))


def allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def tileop(name, *operands, metadata=None):
    return ir.Evaluate(
        tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
            ir.PrimType("void"), ir.Op.get("tl.tileop." + name), list(operands), metadata or {}, ir.UnknownLoc()
        )
    )


def annotation(values):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def native_function(params, body, target=None):
    target = target or tvm_tirx.target.Target("c")
    return tirx.Function(params, body, ir.NoneType()).with_attr("global_symbol", "main").with_attr("target", target)


def run_passes(function):
    module = ir.IRModule({"main": function})
    target = function.attrs["target"]
    with target:
        inferred = tvm_ffi.get_global_func("tl.transform.LayoutInference")()(module)
        lowered = tvm_ffi.get_global_func("tl.transform.LowerTileOp")()(inferred)
    result = lowered["main"]
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    calls = []
    walk(result, lambda node: calls.append(node.op.name) if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) else None)
    assert not any(name.startswith("tl.tileop.") for name in calls)
    return inferred["main"], result


def run_numeric(function, arrays, name):
    artifact_dir = os.environ.get("TILELANG_OPS_ARTIFACT_DIR")
    if artifact_dir:
        directory = Path(artifact_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (name + ".json")).write_text(ir.save_json(function))
    module = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")("c")(ir.IRModule({"main": function}))
    executable = compile_packed_function(module, c_codegen="compute")
    args = [tvm_tirx.runtime.tensor(x) if isinstance(x, np.ndarray) else x for x in arrays]
    executable(*args)
    return [x.numpy() if hasattr(x, "numpy") else x for x in args]


@pytest.mark.parametrize("extent", [1, 17, 257])
def test_production_copy_dynamic(extent):
    n = ir.Var("n", "int32")
    a, b = tensor("a", [n]), tensor("b", [n])
    function = native_function([n, a, b], [tileop("copy", a, b)])
    inferred, lowered = run_passes(function)
    assert not any(isinstance(stmt, ir.Bind) for stmt in inferred.body)
    inputs = np.arange(extent, dtype="float32") - 3.5
    actual = run_numeric(lowered, [extent, inputs, np.full(extent, -99, dtype="float32")], f"copy-{extent}")[-1]
    np.testing.assert_array_equal(actual, inputs)


@pytest.mark.parametrize("kind", ["sum", "max", "min", "abssum", "absmax"])
def test_production_copy_reduce_copy(kind):
    a, b = tensor("a", [3, 5]), tensor("b", [3])
    temporary, reduced = tensor("temporary", [3, 5], "local"), tensor("reduced", [3], "local")
    function = native_function(
        [a, b],
        [
            allocate(temporary),
            allocate(reduced),
            tileop("copy", a, temporary),
            tileop("reduce", temporary, reduced, ir.StringImm(kind), integer(1), integer(1, "bool")),
            tileop("copy", reduced, b),
        ],
    )
    _, lowered = run_passes(function)
    values = (np.arange(15, dtype="float32") - 8).reshape(3, 5)
    source = abs(values) if kind.startswith("abs") else values
    operation = {"sum": np.sum, "max": np.max, "min": np.min, "abssum": np.sum, "absmax": np.max}[kind]
    actual = run_numeric(lowered, [values, np.zeros(3, dtype="float32")], "reduction-" + kind)[-1]
    np.testing.assert_array_equal(actual, operation(source, axis=1))


def test_production_parallel_loop_executes_native_partition():
    a, b = tensor("a", [17]), tensor("b", [17])
    i = ir.Var("i", "int32")
    loop = ir.For(
        i,
        integer(0),
        integer(17),
        integer(1),
        ir.ForKind.PARALLEL,
        [ir.TensorStore(b, [i], ir.TensorLoad(a, [i]) + ir.FloatImm("float32", 2.0))],
    )
    inferred, lowered = run_passes(native_function([a, b], [loop]))
    assert "parallel_loop_layout" in inferred.script()
    values = np.arange(17, dtype="float32")
    actual = run_numeric(lowered, [values, np.zeros(17, dtype="float32")], "parallel-17")[-1]
    np.testing.assert_array_equal(actual, values + 2)


def test_native_fragment_definition_all_uses_and_safe_values_are_remapped():
    from tilelang.layout import Fragment

    fragment = tensor("fragment", [8], "local.fragment")
    output = tensor("output", [8])
    i = ir.Var("i", "int32")
    layout = Fragment([8], forward_fn=lambda x: (0, x))
    function = native_function(
        [output],
        [
            allocate(fragment),
            annotation({"layout_map": {fragment: layout}, "safe_value_map": {fragment: ir.FloatImm("float32", 7.0)}}),
            ir.For(
                i,
                integer(0),
                integer(8),
                integer(1),
                ir.ForKind.DEFAULT,
                [ir.TensorStore(fragment, [i], ir.FloatImm("float32", 4.0)), ir.TensorStore(output, [i], ir.TensorLoad(fragment, [i]))],
            ),
        ],
    )
    before = ir.save_json(function)
    inferred, lowered = run_passes(function)
    assert ir.save_json(function) == before
    physical = next(stmt.var for stmt in lowered.body if isinstance(stmt, ir.Bind) and isinstance(stmt.var.ty, tirx.TensorType))
    assert physical.ty.scope.name == "local"
    assert not physical.same_as(fragment)
    calls = []
    walk(
        lowered, lambda x: calls.append(x) if isinstance(x, ir.Call) and isinstance(x.op, ir.Op) and x.op.name == "tl.annotation" else None
    )
    safe_maps = [tvm_ffi.get_global_func("tl.DecodeAnnotations")(x.args[0], x.attrs["annotation_schema"]) for x in calls]
    safe_map = next(values["safe_value_map"] for values in safe_maps if "safe_value_map" in values)
    assert physical in safe_map and fragment not in safe_map
    restored = ir.load_json(ir.save_json(lowered))
    assert tirx.analysis.check_well_formed(restored)
    assert not tirx.analysis.undefined_vars(restored)


def test_parameter_layout_uses_a_body_view_and_keeps_logical_abi():
    from tilelang.layout import Layout

    a = tensor("a", [2, 4])
    layout = Layout([2, 4], lambda i, j: [j, i])
    function = native_function([a], [annotation({"layout_map": {a: layout}}), ir.Evaluate(ir.TensorLoad(a, [1, 2]))])
    _, lowered = run_passes(function)
    assert lowered.params[0].same_as(a)
    view = lowered.body[0]
    assert isinstance(view, ir.Bind) and view.value.op.name == "tirx.decl_tensor"
    assert tuple(map(int, view.var.ty.shape)) == (4, 2)
    assert not view.var.same_as(a)
    loads = []
    walk(lowered, lambda x: loads.append(x) if isinstance(x, ir.TensorLoad) else None)
    assert len(loads) == 1 and loads[0].source.same_as(view.var)
    assert tuple(map(int, loads[0].indices)) == (2, 1)


def test_shared_layout_rejects_symbolic_extents_with_precise_reason():
    from tilelang.layout import Layout

    n = ir.Var("n", "int32")
    shared = tensor("shared", [n], "shared")
    layout = Layout([n], lambda i: i)
    function = native_function([n], [allocate(shared), annotation({"layout_map": {shared: layout}})])
    with pytest.raises(
        ValueError,
        match=r"Shared buffer `shared` has symbolic extent n; applying a layout requires constant extents",
    ):
        run_passes(function)


def test_cpu_reduction_applies_annotated_input_layout_once():
    from tilelang.layout import Layout

    source, output = tensor("source", [2, 3]), tensor("output", [2])
    local, reduced = tensor("local", [2, 3], "local"), tensor("reduced", [2], "local")
    function = native_function(
        [source, output],
        [
            allocate(local),
            allocate(reduced),
            annotation({"layout_map": {local: Layout([2, 3], lambda i, j: [j, i])}}),
            tileop("copy", source, local),
            tileop("reduce", local, reduced, ir.StringImm("sum"), integer(1), integer(1, "bool")),
            tileop("copy", reduced, output),
        ],
    )
    _, lowered = run_passes(function)
    values = np.arange(6, dtype="float32").reshape(2, 3)
    actual = run_numeric(lowered, [values, np.zeros(2, dtype="float32")], "reduction-transposed-layout")[-1]
    np.testing.assert_array_equal(actual, values.sum(axis=1))


@pytest.mark.parametrize("kind", ["atomic_add", "atomic_max", "atomic_min"])
def test_cpu_atomic_applies_annotated_destination_layout_once(kind):
    from tilelang.layout import Layout

    source, output = tensor("source", [2, 3]), tensor("output", [2, 3])
    local = tensor("local", [2, 3], "local")
    function = native_function(
        [source, output],
        [
            allocate(local),
            annotation({"layout_map": {local: Layout([2, 3], lambda i, j: [j, i])}}),
            tileop("fill", local, ir.FloatImm("float32", 2.0)),
            tileop({"atomic_add": "atomicadd", "atomic_max": "atomicmax", "atomic_min": "atomicmin"}[kind], source, local),
            tileop("copy", local, output),
        ],
    )
    _, lowered = run_passes(function)
    values = (np.arange(6, dtype="float32") - 1).reshape(2, 3)
    operation = {"atomic_add": lambda x: x + 2, "atomic_max": lambda x: np.maximum(2, x), "atomic_min": lambda x: np.minimum(2, x)}[kind]
    actual = run_numeric(lowered, [values, np.zeros((2, 3), dtype="float32")], kind + "-transposed-layout")[-1]
    np.testing.assert_array_equal(actual, operation(values))
