"""Native CUDA metadata retains address provenance and lexical initialization."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, sym, tirx
from tvm_tirx.tirx.layout import S, TileLayout


PREFIX = "tl.cuda.transform."


def tensor(name, dtype="float32", shape=(32,), scope="global", layout=None):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout=layout))


def alloc(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def annotation(values):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def copy(source, destination, name="copy", fields=None):
    from tilelang.language._native import call_intrin

    return ir.Evaluate(call_intrin("handle", "tl.tileop." + name, source, destination, annotations=fields))


def launch(body, thread=None):
    thread = ir.Var("thread", "int32") if thread is None else thread
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.IntImm("int32", 32)], [thread], None, body)


def nodes(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def calls(value, name):
    return [node for node in nodes(value, ir.Call) if node.op.name == name]


def apply(name, params, statements, tmp_path, *, config=None, attrs=None):
    function = tirx.Function(params, statements).with_attr("global_symbol", "main")
    for key, value in (attrs or {}).items():
        function = function.with_attr(key, value)
    before = ir.save_json(function)
    assert tirx.analysis.check_well_formed(function)
    operation = tvm_ffi.get_global_func(PREFIX + name)()
    with tvm_tirx.transform.PassContext(config=config or {}):
        result = operation(tvm_tirx.IRModule({"main": function}))
        tvm_tirx.ir.assert_structural_equal(operation(result), result)
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    (tmp_path / "before.json").write_text(before)
    (tmp_path / "after.json").write_text(ir.save_json(result))
    return result["main"]


def metadata(call):
    return tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[-1], call.attrs["annotation_schema"])


@pytest.mark.parametrize("name", ["copy", "async_copy", "tma_copy"])
@pytest.mark.parametrize("origin", ["parameter", "alias", "device_offset", "device_shape", "global_workspace"])
def test_copy_base_classification_uses_typed_backing_provenance(name, origin, tmp_path):
    source = tensor("source")
    shared = tensor("shared", scope="shared")
    thread = ir.Var("thread", "int32")
    body = [alloc(shared)]
    operand = source
    if origin in ("alias", "device_offset", "device_shape"):
        pointer = source.data
        if origin == "device_offset":
            pointer = ir.Call("tirx.ptr_byte_offset", [pointer, thread * 4])
        shape = (thread + 32,) if origin == "device_shape" else (32,)
        operand = tensor("view", shape=shape)
        body.append(ir.Bind(operand, ir.Call("tirx.decl_tensor", [pointer], ty_args=[operand.ty])))
    elif origin == "global_workspace":
        operand = tensor("workspace")
        body.append(alloc(operand))
    body.append(copy(operand, shared, name))
    result = apply("AnnotateDeviceBoundTmaCopies", [source], [launch(body, thread)], tmp_path)
    (call,) = calls(result, "tl.tileop." + name)
    hints = metadata(call)
    assert ("tma_descriptor_base_is_device_bound" in hints) == (origin in ("device_offset", "device_shape"))


def test_parameter_backed_address_lvalue_is_not_a_tensor_read(tmp_path):
    source = tensor("source")
    view = tensor("view", shape=(16,))
    shared = tensor("shared", shape=(16,), scope="shared")
    pointer = ir.Call("tirx.address_of", [ir.TensorLoad(source, [7])])
    body = [alloc(shared), ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])), copy(view, shared)]
    result = apply("AnnotateDeviceBoundTmaCopies", [source], [launch(body)], tmp_path)
    assert "tma_descriptor_base_is_device_bound" not in metadata(calls(result, "tl.tileop.copy")[0])


def test_explicit_copy_annotation_is_preserved(tmp_path):
    source, shared = tensor("source"), tensor("shared", scope="shared")
    body = [alloc(shared), copy(source, shared, fields={"tma_descriptor_base_is_device_bound": 1, "coalesced_width": 4})]
    result = apply("AnnotateDeviceBoundTmaCopies", [source], [launch(body)], tmp_path)
    fields = metadata(calls(result, "tl.tileop.copy")[0])
    assert int(fields["tma_descriptor_base_is_device_bound"]) == 1
    assert int(fields["coalesced_width"]) == 4


def test_l2_dynamic_window_and_other_annotations_survive(tmp_path):
    size = ir.Var("size", "int64")
    source = tensor("source", "float16", (size, 8))
    statement = annotation({"l2_hit_ratio_map": {source: ir.FloatImm("float64", 0.75)}, "preserve": 7})
    result = apply("LowerL2Persistent", [size, source], [statement], tmp_path, attrs={"tl.l2_policy_materialized": True})
    ratio, byte_count = result.attrs["l2_persistent_map"]["source"]
    assert float(ratio.value) == 0.75
    assert sym.Analyzer().can_prove_equal(byte_count, size * 16)
    assert "tl.l2_policy_materialized" not in result.attrs
    (remaining,) = calls(result, "tl.annotation")
    assert int(metadata(remaining)["preserve"]) == 7
    assert "l2_hit_ratio_map" not in metadata(remaining)


def test_l2_annotation_connects_to_host_origin_consumer(tmp_path):
    source = tensor("source", "float32", (64,), layout=TileLayout(S[64:1] + 7))
    result = apply(
        "LowerL2Persistent", [source], [annotation({"l2_hit_ratio_map": {source: ir.FloatImm("float64", 0.5)}}), launch([])], tmp_path
    )
    lowered = tvm_ffi.get_global_func(PREFIX + "LowerHopperIntrin")()(tvm_tirx.IRModule({"main": result}))["main"]
    windows = [call for call in calls(lowered, "tirx.call_packed") if call.args[0].value == "__tvm_cuda_stream_set_access_policy_window"]
    assert len(windows) == 1
    assert windows[0].args[1].op.name == "tirx.handle_add_byte_offset"
    assert int(sym.Analyzer().simplify(windows[0].args[1].args[1])) == 28
    assert int(sym.Analyzer().simplify(windows[0].args[2])) == 256


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_l2_rejects_out_of_range_hit_ratio(ratio, tmp_path):
    source = tensor("source")
    with pytest.raises(ValueError, match="hit ratio"):
        apply("LowerL2Persistent", [source], [annotation({"l2_hit_ratio_map": {source: ir.FloatImm("float64", ratio)}})], tmp_path)


@pytest.mark.parametrize("disable", [False, True])
def test_barrier_initializers_share_one_fence_and_respect_election(disable, tmp_path):
    first = tensor("first", "uint64", (2,), "shared.barrier")
    second = tensor("second", "uint64", (1,), "shared.barrier")
    thread = ir.Var("thread", "int32")
    body = [
        alloc(first),
        annotation({"barrier_init": {first: [ir.IntImm("int32", 1), ir.IntImm("int32", 128)]}}),
        alloc(second),
        annotation({"barrier_init": {second: [ir.IntImm("int32", 32)]}}),
        ir.Evaluate(first.data),
    ]
    result = apply("LowerSharedBarrier", [], [launch(body, thread)], tmp_path, config={"tl.disable_shuffle_elect": disable})
    init = calls(result, "tl.ptx_init_barrier_thread_count")
    assert [int(call.args[1]) for call in init] == [1, 128, 32]
    assert len(calls(result, "tl.ptx_fence_barrier_init")) == 1
    assert [call.args[0].value for call in calls(result, "tirx.gpu_storage_sync")] == ["shared"]
    assert not calls(result, "tl.annotation")
    condition = nodes(result, ir.If)[0].condition
    if disable:
        tvm_tirx.ir.assert_structural_equal(condition, thread == 0)
    else:
        assert condition.op.name == "tl.tl_shuffle_elect"
    assert all(bool(binding.value.attrs["tl.barrier_initialized"]) for binding in nodes(result, ir.Bind))


def test_cluster_barrier_state_does_not_leak_to_sibling_scope(tmp_path):
    regions = []
    for name, scope in [("cluster", "shared.cluster_barrier"), ("ordinary", "shared.barrier")]:
        barrier = tensor(name, "uint64", (1,), scope)
        regions.append(
            launch([alloc(barrier), annotation({"barrier_init": {barrier: [ir.IntImm("int32", 1)]}}), ir.Evaluate(barrier.data)])
        )
    result = apply("LowerSharedBarrier", [], regions, tmp_path)
    assert [call.args[0].value for call in calls(result, "tirx.gpu_storage_sync")] == ["cluster", "shared"]


@pytest.mark.parametrize("error", ["missing", "length", "zero"])
def test_barrier_metadata_is_validated_before_codegen(error, tmp_path):
    barrier = tensor("barrier", "uint64", (2,), "shared.barrier")
    statements = [alloc(barrier)]
    if error != "missing":
        counts = [ir.IntImm("int32", 1)] if error == "length" else [ir.IntImm("int32", 1), ir.IntImm("int32", 0)]
        statements.append(annotation({"barrier_init": {barrier: counts}}))
    with pytest.raises(ValueError):
        apply("LowerSharedBarrier", [], [launch(statements)], tmp_path)
