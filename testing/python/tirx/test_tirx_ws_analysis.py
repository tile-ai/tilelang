import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, sym, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def native(name, *arguments):
    return tvm_ffi.get_global_func(f"tl.cuda.analysis.{name}")(*arguments)


def nodes(value, kind):
    result = []
    stmt_functor.post_order_visit(value, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


@pytest.mark.parametrize("reads_shared", [False, True])
def test_simt_producer_classification_uses_native_accesses(reads_shared):
    source = TensorView.create("source", [128], "float32")
    shared = TensorView.create("shared", [128], "float32", scope="shared")
    index = ir.Var("index", "int32")
    value = source.load(index) + shared.load(index) if reads_shared else source.load(index)
    body = ir.For(index, 0, 128, 1, ir.ForKind.PARALLEL, shared.store(index, value))
    assert native("ClassifyStmt", body, Target("cuda")) == (6 if reads_shared else 3)


def test_tensor_remapping_updates_allocations_pointers_and_regions_and_executes():
    original = TensorView.create("original", [4], "int32", scope="local")
    replacement = TensorView.create("replacement", [8], "int32", scope="local")
    output = TensorView.create("output", [4], "int32")
    index = ir.Var("index", "int32")
    body = ir.SeqStmt(
        [
            original.allocate(),
            ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, [original.store(index, index + 10), output.store(index, original.load(index))]),
        ]
    )
    remapped = native("RemapBuffers", body, {original.tensor: replacement.tensor})
    assert not any(variable.same_as(original.tensor) for variable in nodes(remapped, ir.Var))
    function = tirx.Function([output.tensor], remapped, ret_type=ir.TupleType([])).with_attr("global_symbol", "remap_test")
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((4,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [10, 11, 12, 13])


@pytest.mark.parametrize("dtype,bits", [("uint8", 8), ("uint16", 16), ("uint32", 32)])
def test_pipeline_alias_versions_use_owner_storage_stride(dtype, bits):
    owner = TensorView.create("owner", [96], "uint8", scope="shared", data_alignment=32)
    alias = TensorView.create("alias", [4], dtype, scope="shared", elem_offset=1)
    pointer = ir.Call("tirx.reinterpret", [owner.data], ty=ir.PtrType(alias.tensor.ty.dtype, alias.tensor.ty.scope))
    body = ir.SeqStmt([owner.allocate(), alias.declare(pointer)])
    assert native("StorageAliases", body)[alias.tensor].same_as(owner.tensor)
    expanded = native("ExpandPipelineTensor", owner.tensor, 3)
    view = native("ExpandPipelineAlias", alias.tensor, owner.tensor, expanded)
    assert [int(extent) for extent in view.ty.shape] == [3, 4]
    origin = view.ty.layout.apply(0, 0, shape=view.ty.shape)["m"]
    following = view.ty.layout.apply(ir.const(1, "int64"), 0, shape=view.ty.shape)["m"]
    analyzer = sym.Analyzer()
    assert int(analyzer.simplify(origin)) == 1
    assert int(analyzer.simplify(following - origin)) == 96 * 8 // bits
    rewritten = native("RewriteBufferVersions", ir.Evaluate(alias.load(0)), {alias.tensor: view}, {alias.tensor: [ir.const(2, "int64")]})
    assert rewritten.value.source.same_as(view)
    assert [int(index) for index in rewritten.value.indices] == [2, 0]


def test_manual_warp_specialization_is_lexical_metadata():
    body = ir.RegionStmt("tirx.scope", [], [], {"warp_specialize": True}, ir.Evaluate(0))
    assert native("HasManualWarpSpecialization", body)
    assert not native("HasManualWarpSpecialization", ir.Evaluate(0))


@pytest.mark.parametrize("kind", [ir.ForKind.DEFAULT, ir.ForKind.UNROLLED, ir.ForKind.PARALLEL, ir.ForKind.VECTORIZED])
def test_scheduler_treats_parallel_and_vectorized_loops_as_atomic_operations(kind):
    from tilelang._compat.ws import _Program

    output = TensorView.create("output", [32], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 32, 1, kind, output.store(index, index + 1))
    normalized = native("PreprocessIR", ir.SeqStmt([loop]), Target("cuda"))
    assert normalized is not None
    program = _Program(normalized)
    nested = kind in (ir.ForKind.DEFAULT, ir.ForKind.UNROLLED)
    assert len(program.scopes) == 1 + int(nested)
    assert len(program.operations) == 1 + int(nested)


@pytest.mark.parametrize("index_scope", ["shared", "local"])
@pytest.mark.parametrize("worker_reads_index", [False, True])
def test_role_scheduler_tracks_transitive_memory_definitions(index_scope, worker_reads_index):
    from tilelang._compat.ws import _Program, _Schedule, _role_based

    source = TensorView.create("source", [64], "float32")
    output = TensorView.create("output", [32], "float32")
    shared = TensorView.create("shared", [32], "float32", scope="shared")
    indices = TensorView.create("indices", [1], "int32", scope=index_scope)
    offset = ir.Var("offset", "int32")
    index = ir.Var("index", "int32")
    value = shared.load(index)
    if worker_reads_index:
        value = value + indices.load(0).astype("float32")
    body = ir.SeqStmt(
        [
            shared.allocate(),
            indices.allocate(),
            indices.store(0, 32),
            ir.Bind(offset, indices.load(0)),
            ir.For(index, 0, 32, 1, ir.ForKind.PARALLEL, shared.store(index, source.load(offset + index))),
            ir.For(index, 0, 32, 1, ir.ForKind.PARALLEL, output.store(index, value)),
        ]
    )
    target = Target("cuda")
    body = native("PreprocessIR", body, target)
    if worker_reads_index and index_scope == "shared":
        with pytest.raises(ValueError, match="Cross-role memory dependencies"):
            _role_based(None, body, 128, target)
        return
    schedule = _role_based(None, body, 128, target)
    assert schedule is not None
    program = _Program(body)
    checked = _Schedule(program, schedule)
    expected = [{"Load", "Worker"}, {"Load"}, {"Load"}, {"Worker"}] if worker_reads_index else [{"Load"}, {"Load"}, {"Load"}, {"Worker"}]
    assert [{role for role, _ in checked.placements[operation.identifier]} for operation in program.operations.values()] == expected
    assert len(schedule.pipelines) == 1


def test_preprocessor_preserves_scalar_and_tensor_binding_scopes():
    storage = TensorView.create("storage", [4], "int32", scope="local")
    output = TensorView.create("output", [4], "int32")
    scalar = ir.Var("scalar", "int32")
    index = ir.Var("index", "int32")
    body = ir.SeqStmt(
        [
            storage.allocate(),
            ir.Bind(scalar, ir.const(9)),
            ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, [storage.store(index, scalar), output.store(index, storage.load(index))]),
        ]
    )
    result = native("PreprocessIR", body, Target("cuda"))
    assert isinstance(result.seq[0], ir.Bind) and result.seq[0].value.op.name == "tirx.alloc_tensor"
    assert isinstance(result.seq[1], ir.Bind) and result.seq[1].var.same_as(scalar)
    assert result.seq[1].value.op.name == "tl.ws_value"
    assert native("GetWSOpId", result.seq[1]) == "scalar_0"
    function = tirx.Function([output.tensor], result, ret_type=ir.TupleType([]))
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    ir.assert_structural_equal(result, native("PreprocessIR", result, Target("cuda")))


def test_preprocessor_preserves_existing_ids_and_avoids_collisions():
    first = ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("first")], ty="void", attrs={"tl.ws_op_id": "scalar_0"}))
    scalar = ir.Var("scalar", "int32")
    result = native("PreprocessIR", ir.SeqStmt([first, ir.Bind(scalar, ir.const(3))]), Target("cuda"))
    assert native("GetWSOpId", result.seq[0]) == "scalar_0"
    assert native("GetWSOpId", result.seq[1]) == "scalar_1"


@pytest.mark.parametrize("exit_type", [ir.Break, ir.Continue, ir.Return])
def test_auto_schedule_declines_explicit_control_flow_exits(exit_type):
    statement = ir.Return(ir.Tuple([])) if exit_type is ir.Return else exit_type()
    assert native("PreprocessIR", statement, Target("cuda")) is None


def test_auto_schedule_declines_hand_managed_barrier_protocols():
    body = ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared")]))
    assert native("PreprocessIR", body, Target("cuda")) is None


@pytest.mark.parametrize("manual_protocol", [False, True])
@pytest.mark.parametrize("identified", [False, True])
def test_automatic_raw_copy_completion_requires_an_unmanaged_protocol(manual_protocol, identified):
    source = TensorView.create("source", [8], "float16")
    shared = TensorView.create("shared", [8], "float16", scope="shared")
    source_ptr = ir.Call("tl.access_ptr", [source.load(0), 8, 1])
    shared_ptr = ir.Call("tl.access_ptr", [shared.load(0), 8, 2])
    attributes = {"tl.ws_op_id": "raw_copy"} if identified else None
    copy = ir.Evaluate(ir.Call("tl.ptx_cp_async", [shared_ptr, source_ptr, 8], attrs=attributes, ty="void"))
    assert native("ClassifyStmt", copy, Target("cuda")) == 2
    statements = [shared.allocate(), copy]
    if manual_protocol:
        statements.append(ir.Evaluate(ir.Call("tirx.ptx.cp_async_commit_group", ["async", "commit_group", ""])))
    result = native("PreprocessIR", ir.SeqStmt(statements), Target("cuda"))
    if manual_protocol:
        assert result is None
    else:
        assert native("ClassifyStmt", result.seq[1], Target("cuda")) == 1
        if identified:
            assert native("GetWSOpId", result.seq[1]) == "raw_copy"
        function = tirx.Function([source.tensor], result)
        assert tirx.analysis.check_well_formed(function)
        ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
        ir.assert_structural_equal(result, native("PreprocessIR", result, Target("cuda")))


def test_while_scope_and_conditional_stores_keep_unique_ids():
    condition = ir.Var("condition", "bool")
    output = TensorView.create("output", [1], "int32")
    body = ir.While(condition, ir.If(condition, output.store(0, 2), output.store(0, 3)))
    result = native("PreprocessIR", body, Target("cuda"))
    function = tirx.Function([condition, output.tensor], result, ret_type=ir.TupleType([]))
    assert tirx.analysis.check_well_formed(function)
    identifiers = [native("GetWSOpId", statement) for statement in nodes(result, ir.Stmt)]
    identifiers = [identifier for identifier in identifiers if identifier is not None]
    assert len(identifiers) == len(set(identifiers)) == 2
    ir.assert_structural_equal(result, native("PreprocessIR", result, Target("cuda")))
