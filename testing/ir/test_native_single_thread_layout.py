"""One-thread loop planning preserves typed axes and physical-storage checks."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, sym, tirx

import tilelang
from tilelang.analysis._visitor import tilelang_metadata
from test_native_dynamic_stride_vectorize import (
    _program,
    test_dynamic_strided_parallel_layout_lowering_and_cpu_values as _check_dynamic_values,
)


@pytest.mark.parametrize("cost_model", ["register-count", "io-aware"])
@pytest.mark.parametrize("size", [1, 3, 7])
@pytest.mark.parametrize("stride", [4, 5])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_dynamic_stride_values_under_both_cost_models(cost_model, size, stride, index_dtype):
    # Reuse the original program, ABI, exact NumPy oracle and all IR assertions.
    with tvm_tirx.transform.PassContext(config={"tl.layout_cost_model": cost_model}):
        _check_dynamic_values(size, stride, index_dtype)


def _loop_layout(module):
    found = []

    def collect(node):
        metadata = tilelang_metadata(node)
        if metadata is not None and metadata.get("key") == "tl.loop.annotations":
            metadata = metadata["value"]
        if metadata is not None and "parallel_loop_layout" in metadata:
            found.append(metadata["parallel_loop_layout"])

    tvm_ffi.structural_walk(module, ((ir.Call, ir.RegionStmt), collect), order="pre")
    assert len(found) == 1
    return found[0]


@pytest.mark.parametrize("cost_model", ["register-count", "io-aware"])
@pytest.mark.parametrize("row", [0, 7, 2**31, 2**62])
def test_single_thread_wide_coordinates_do_not_need_a_flat_capacity(cost_model, row):
    value = _program(True, "int64").with_attr("target", tvm_tirx.target.Target("c"))
    original = ir.save_json(value)
    with tvm_tirx.transform.PassContext(config={"tl.layout_cost_model": cost_model}):
        inferred = tilelang.transform.LayoutInference()(ir.IRModule.from_expr(value))
    layout = _loop_layout(inferred)
    coordinate = [ir.const(row, "int64"), ir.const(1, "int64")]
    analyzer = sym.Analyzer()
    # The final point needs more than int64 if flattened; the logical axes
    # themselves remain representable. This is an IR mapping check, not a
    # claim that a tensor of that total size was allocated or executed.
    assert [int(analyzer.simplify(x)) for x in layout.map_forward_index(coordinate)] == [row, 1]
    assert [int(analyzer.simplify(x)) for x in layout.map_forward_thread(coordinate)] == [0]
    assert [str(x.ty) for x in layout.get_output_shape()] == ["int64", "int64"]
    assert ir.save_json(value) == original
    assert tirx.analysis.check_well_formed(inferred)


def _symbolic_fragment():
    size = ir.Var("size", "int64")
    shape = [size, ir.const(2, "int64")]
    source = ir.Var("source", tirx.TensorType(shape, "float32"))
    fragment = ir.Var("fragment", tirx.TensorType(shape, "float32", scope="local.fragment"))
    layout = tilelang.Fragment(
        shape,
        forward_thread_fn=lambda row, col: 0,
        forward_index_fn=lambda row, col: ir.prim.Cast("int64", row) * ir.const(2, "int64") + col,
    )
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"layout_map": {fragment: layout}})
    annotation = ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))
    row, col = ir.Var("row", "int64"), ir.Var("col", "int64")
    body = ir.TensorStore(fragment, [row, col], ir.TensorLoad(source, [row, col]))
    for var, extent in [(col, shape[1]), (row, size)]:
        body = ir.For(var, ir.const(0, "int64"), extent, ir.const(1, "int64"), ir.ForKind.PARALLEL, [body])
    return (
        tirx.Function(
            [size, source],
            [ir.Bind(fragment, ir.Call("tirx.alloc_tensor", [], ty_args=[fragment.ty])), annotation, body],
            ir.NoneType(),
        )
        .with_attr("target", tvm_tirx.target.Target("c"))
        .with_attr("global_symbol", "main")
    )


@pytest.mark.parametrize("cost_model", ["register-count", "io-aware"])
@pytest.mark.parametrize("entry", ["LayoutInference", "LowerTileOp"])
def test_symbolic_fragment_allocation_still_requires_a_safe_capacity(cost_model, entry):
    value = _symbolic_fragment()
    original = ir.save_json(value)
    assert tirx.analysis.check_well_formed(value)
    with (
        tvm_tirx.transform.PassContext(config={"tl.layout_cost_model": cost_model}),
        pytest.raises(ValueError, match="Cannot infer a safe output extent"),
    ):
        getattr(tilelang.transform, entry)()(ir.IRModule.from_expr(value))
    assert ir.save_json(value) == original


@pytest.mark.parametrize("cost_model", ["register-count", "io-aware"])
@pytest.mark.parametrize("index_dtype,static_extent", [("int32", None), ("int32", 4), ("int64", 4)])
def test_static_and_narrow_single_thread_plans_retain_flat_coordinates(cost_model, index_dtype, static_extent):
    value = _program(True, index_dtype, static_extent=static_extent).with_attr("target", tvm_tirx.target.Target("c"))
    with tvm_tirx.transform.PassContext(config={"tl.layout_cost_model": cost_model}):
        inferred = tilelang.transform.LayoutInference()(ir.IRModule.from_expr(value))
    layout = _loop_layout(inferred)
    analyzer = sym.Analyzer()
    # Keep the established static/narrow planning shape, including the
    # fusion opportunity; only the unbounded wide case needs another form.
    assert len(layout.get_output_shape()) == 1
    # LayoutInference fuses a static nest before planning, while a symbolic
    # nest retains two input axes. Query each actual logical input domain.
    assert len(layout.get_input_shape()) == (2 if static_extent is None else 1)
    for row in range(4):
        for col in range(2):
            coordinates = [row, col] if static_extent is None else [row * 2 + col]
            assert [int(analyzer.simplify(x)) for x in layout.map_forward_index(coordinates)] == [row * 2 + col]
            assert [int(analyzer.simplify(x)) for x in layout.map_forward_thread(coordinates)] == [0]
