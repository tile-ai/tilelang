"""Annotation owners materialize semantics and retire only their own carriers."""

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

import tilelang  # noqa: F401
import tilelang.ascend  # noqa: F401
from tilelang.layout import Layout
from tilelang.testing.ir import collect_nodes


def _annotation(fields):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(fields)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def _attribute(key, value, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def _ascend(name, fields, body):
    args, attrs = tvm_ffi.get_global_func("tl.EncodeAscendBlock")(name, fields)
    return ir.RegionStmt("tl.ascend_block", args, [], attrs, body)


def _scope(carrier, fields, body):
    if carrier == "statement":
        return ir.SeqStmt([_annotation(fields), *body])
    if carrier == "source_after":
        return _attribute(
            "tl.sblock",
            {"name": "ordinary"},
            [*body, _annotation({"tl.source_block_annotation": True, **fields})],
        )
    if carrier == "attribute":
        ((key, value),) = fields.items()
        return _attribute(key, value, body)
    if carrier == "kernel":
        return _attribute("tl.kernel", fields, body)
    if carrier == "block":
        return _attribute("tl.sblock", {"name": "ordinary", "annotations": fields}, body)
    assert carrier == "ascend"
    return _ascend("VECTOR", fields, body)


def _function(params, body, target="c"):
    function = tirx.Function(params, body)
    return function if target is None else function.with_attr("target", Target(target))


def _run(name, function):
    before = ir.save_json(function)
    assert tirx.analysis.check_well_formed(function)
    transform = tvm_ffi.get_global_func("tl.transform." + name)()
    result = transform(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def _metadata(function):
    decode = tvm_ffi.get_global_func("tl.DecodeAnnotations")
    values = []
    for call in collect_nodes(function, ir.Call):
        if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation":
            values.append(dict(decode(call.args[0], call.attrs["annotation_schema"])))
    for region in collect_nodes(function, ir.RegionStmt):
        if region.op.name == "tl.ascend_block":
            fields = dict(region.attrs["annotations"])
            if region.args:
                fields.update(dict(decode(region.args[0], region.attrs["annotation_schema"])))
            values.append(fields)
        elif region.op.name == "tl.attr":
            fields = decode(region.args[0], region.attrs["annotation_schema"])
            key, value = str(fields["key"]), fields["value"]
            if key == "tl.sblock":
                values.append(dict(value.get("annotations", {})))
            elif key in {"tl.kernel", "tl.loop.annotations"}:
                values.append(dict(value))
            else:
                values.append({key: value})
    values.extend(dict(loop.annotations) for loop in collect_nodes(function, ir.For))
    return values


_CARRIERS = ["statement", "source_after", "attribute", "kernel", "block", "ascend"]


@pytest.mark.parametrize("carrier", _CARRIERS)
def test_layout_owner_materializes_each_carrier_before_retiring_it(carrier):
    scratch = ir.Var("scratch", tirx.TensorType([2, 4], "int32", scope="local"))
    output = ir.Var("output", tirx.TensorType([1], "int32", scope="global"))
    allocate = ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty]))
    body = [
        ir.TensorStore(scratch, [1, 2], ir.const(7)),
        ir.TensorStore(output, [0], ir.TensorLoad(scratch, [1, 2])),
    ]
    layout = Layout([2, 4], lambda i, j: i * 4 + j)
    scope = _scope(carrier, {"layout_map": {scratch: layout}}, body)
    function = _function([output], [allocate, scope], "ascend" if carrier == "ascend" else "c")
    result = _run("LowerTileOp", function)
    assert "tl.tile_ops_lowered" not in result.attrs
    assert all("layout_map" not in fields for fields in _metadata(result))
    allocations = [
        bind for bind in collect_nodes(result, ir.Bind) if isinstance(bind.value, ir.Call) and bind.value.op.name == "tirx.alloc_tensor"
    ]
    assert len(allocations) == 1
    assert [int(extent) for extent in allocations[0].var.ty.shape] == [8]
    stores = [store for store in collect_nodes(result, ir.TensorStore) if store.dest.name == "scratch"]
    assert len(stores) == 1 and len(stores[0].indices) == 1
    assert int(sym.Analyzer().simplify(stores[0].indices[0])) == 6


def test_loop_hints_are_retired_at_their_original_definition_carriers():
    index = ir.Var("i", "int32")
    width = ir.Var("width", "int32")
    body = [
        ir.Bind(width, ir.const(4)),
        _annotation(
            {
                "coalesced_width": width,
                "parallel_loop_predicate": width > 0,
                "parallel_loop_requires_padding_guard": True,
                "tl.unconsumed_loop_test": width,
            }
        ),
        ir.Evaluate(index),
    ]
    loop = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        body,
        annotations={
            "prefer_async": 1,
            "parallel_prefer_async": True,
            "parallel_async_without_async_commit_wait": True,
        },
    )
    function = _function([], [_attribute("tl.loop.annotations", {"vectorize": 1}, [loop])])
    result = _run("LowerTileOp", function)
    fields = _metadata(result)
    consumed_keys = {
        "coalesced_width",
        "prefer_async",
        "vectorize",
        "parallel_prefer_async",
        "parallel_async_without_async_commit_wait",
        "parallel_loop_predicate",
        "parallel_loop_requires_padding_guard",
    }
    assert all(not consumed_keys.intersection(metadata) for metadata in fields)
    remaining = [metadata["tl.unconsumed_loop_test"] for metadata in fields if "tl.unconsumed_loop_test" in metadata]
    assert len(remaining) == 1 and remaining[0].same_as(width)
    assert len(collect_nodes(result, ir.Bind)) == 1


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("carrier", _CARRIERS)
def test_safety_owner_handles_each_carrier_with_and_without_guards(disabled, carrier):
    source = ir.Var("source", tirx.TensorType([4], "int32", scope="global"))
    output = ir.Var("output", tirx.TensorType([1], "int32", scope="global"))
    index = ir.Var("index", "int32")
    store = ir.TensorStore(output, [0], ir.TensorLoad(source, [index]))
    scope = _scope(carrier, {"safe_value_map": {source: -7}}, [store])
    function = _function([source, output, index], [scope], "ascend" if carrier == "ascend" else "c")
    with PassContext(config={"tl.disable_safe_memory_legalize": disabled}):
        result = _run("LegalizeSafeMemoryAccess", function)
    assert all("safe_value_map" not in fields for fields in _metadata(result))
    guards = [call for call in collect_nodes(result, ir.Call) if isinstance(call.op, ir.Op) and call.op.name == "prim.if_then_else"]
    if disabled:
        assert not guards
        (actual_store,) = collect_nodes(result, ir.TensorStore)
        ir.assert_structural_equal(actual_store, store)
    else:
        assert guards and all(int(call.args[-1]) == -7 for call in guards)


@pytest.mark.parametrize("disabled", [False, True])
def test_safety_owner_preserves_unrelated_dynamic_block_metadata(disabled):
    source = ir.Var("source", tirx.TensorType([4], "int32"))
    index = ir.Var("index", "int32")
    region = _scope(
        "block",
        {"safe_value_map": {source: -7}, "keep": index},
        [ir.Evaluate(ir.TensorLoad(source, [index]))],
    )
    with PassContext(config={"tl.disable_safe_memory_legalize": disabled}):
        result = _run("LegalizeSafeMemoryAccess", _function([source, index], [region]))
    (metadata,) = _metadata(result)
    assert set(metadata) == {"keep"} and metadata["keep"].same_as(index)


def _source_allocation(target, marker=1):
    scratch = ir.Var("scratch", tirx.TensorType([4], "int32", scope="local"))
    allocate = ir.Bind(
        scratch,
        ir.Call(
            "tirx.alloc_tensor",
            [],
            attrs={"tl.source_block_alloc": ir.const(marker), "keep": "origin-test"},
            ty_args=[scratch.ty],
        ),
    )
    scope = ir.RegionStmt("tl.lexical_scope", [], [], {}, [allocate, ir.TensorStore(scratch, [0], ir.const(7))])
    return _function([], [scope], target)


def test_non_cpu_source_origin_retirement_preserves_allocation_scope_and_type():
    function = _source_allocation("ascend")
    result = _run("LowerSourceAllocations", function)
    (scope,) = result.body.seq
    assert scope.op.name == "tl.lexical_scope"
    (allocation,) = collect_nodes(scope, ir.Bind)
    assert "tl.source_block_alloc" not in allocation.value.attrs
    original = collect_nodes(function, ir.Bind)[0]
    assert allocation.var.same_as(original.var)
    assert dict(allocation.value.attrs) == {"keep": "origin-test"}
    ir.assert_structural_equal(allocation.value.args, original.value.args)
    ir.assert_structural_equal(allocation.value.ty, original.value.ty)
    ir.assert_structural_equal(allocation.value.ty_args, original.value.ty_args)


@pytest.mark.parametrize(
    "target,pattern",
    [(None, "bound target"), ("c", "LowerCPUKernelLaunch"), ("llvm", "LowerCPUKernelLaunch")],
)
def test_source_origin_cannot_bypass_its_cpu_or_target_owner(target, pattern):
    with pytest.raises(ValueError, match=pattern):
        _run("LowerSourceAllocations", _source_allocation(target))


@pytest.mark.parametrize("marker", [0, 2])
def test_source_origin_rejects_non_source_markers(marker):
    with pytest.raises(ValueError, match="source allocation marker"):
        _run("LowerSourceAllocations", _source_allocation("ascend", marker))


def test_ascend_annotation_owner_runs_without_the_optional_scheduler():
    region = _ascend("SIMT_VF", {"tl.vf_source_index": 7, "tl.vf_latency": 33}, [ir.Evaluate(ir.const(1))])
    result = _run("LowerAscendAnnotations", _function([], [region], "ascend"))
    (retained,) = result.body.seq
    assert retained.op.name == "tl.ascend_block"
    assert dict(retained.attrs["annotations"]) == {"tl.vf_source_index": 7}
    ir.assert_structural_equal(retained.body, region.body)


@pytest.mark.parametrize(
    "fields,pattern",
    [
        ({"tl.vf_source_index": -1}, "nonnegative"),
        ({"tl.unconsumed_hint": 1}, "Unconsumed structured block annotation"),
    ],
)
def test_ascend_header_errors_are_owned_by_the_backend(fields, pattern):
    with pytest.raises(ValueError, match=pattern):
        _run("LowerAscendAnnotations", _function([], [_ascend("SIMT_VF", fields, [])], "ascend"))


def test_ascend_dynamic_latency_is_validated_before_retirement():
    latency = ir.Var("latency", "int32")
    region = _ascend("SIMT_VF", {"tl.vf_latency": latency}, [])
    with pytest.raises(ValueError, match="constant integer"):
        _run("LowerAscendAnnotations", _function([latency], [region], "ascend"))


@pytest.mark.parametrize("scope", [False, True])
def test_ascend_region_and_scope_require_their_backend(scope):
    region = _attribute("tl.simtvf_scope", 1, []) if scope else _ascend("SIMT_VF", {}, [])
    with pytest.raises(ValueError, match="requires an Ascend target"):
        _run("LowerAscendAnnotations", _function([], [region], "c"))
