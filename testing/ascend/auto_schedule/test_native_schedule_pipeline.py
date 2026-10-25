import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang.ascend.language as T
from tilelang._compat.tensor import TensorView
from tilelang._compat.stmt_functor import post_order_visit


def _nodes(root, kind):
    result = []
    post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _module(body, params=(), attrs=None, sid=None):
    root = ir.RegionStmt("tl.scope", [], [], {"name_hint": "tilelang_root", **(attrs or {})}, body)
    if sid is not None:
        root = ir.RegionStmt("tl.launch_grid", ["cthread", 2], [sid], {}, root)
    function = tirx.Function(list(params), root, ret_type=ir.TupleType([]))
    return ir.IRModule({"main": function.with_attr("target", Target("ascend")).with_attr("global_symbol", "main")})


def _prefix(module):
    assert tirx.analysis.check_well_formed(module)
    snapshots = {}
    for name in [
        "MaterializeScheduleUnits",
        "AssignCore",
        "AnnotateMultiBufferEligible",
        "EstimateLatency",
        "AutoSchedule",
        "PrepareMultiBuffer",
        "ResolveCore",
        "InsertSync",
    ]:
        module = tvm_ffi.get_global_func("tl.transform." + name)()(module)
        assert tirx.analysis.check_well_formed(module, assert_mode=False), name
        ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
        snapshots[name] = module
    return snapshots


def _lower(module):
    result = _prefix(module)["InsertSync"]
    for name in ["MaterializeMultiBuffer", "LowerScheduledTIR"]:
        result = tvm_ffi.get_global_func("tl.transform." + name)()(result)
        assert tirx.analysis.check_well_formed(result), name
        ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


def _walk_statements(statement, ancestors=()):
    yield statement, ancestors
    if isinstance(statement, ir.SeqStmt):
        children = statement.seq
    elif isinstance(statement, (ir.RegionStmt, ir.For, ir.While)):
        children = [statement.body]
    elif isinstance(statement, ir.If):
        children = [statement.then_case, *([statement.else_case] if statement.else_case is not None else [])]
    else:
        children = []
    for child in children:
        yield from _walk_statements(child, (*ancestors, statement))


@pytest.mark.parametrize("begin,end,step", [(0, 4, 1), (6, -2, -2)])
@pytest.mark.parametrize("mode", ["counter", "iteration"])
def test_native_copy_ring_preserves_storage_through_schedule_prefix(begin, end, step, mode):
    source = TensorView.create("source", [8, 64], "float32")
    output = TensorView.create("output", [8, 64], "float32")
    shared = TensorView.create("shared", [64], "float32", scope="shared.dyn")
    index = ir.Var("index", "int32")
    region = [ir.Range.from_min_extent(index, 1), ir.Range(0, 64)]
    loop = ir.For(
        index,
        begin,
        end,
        step,
        ir.ForKind.DEFAULT,
        [ir.Evaluate(T.copy(source.region(region), shared.tensor)), ir.Evaluate(T.copy(shared.tensor, output.region(region)))],
        annotations={"num_stages": 2, "enable_offset": mode == "iteration"},
    )
    snapshots = _prefix(
        _module(
            [shared.allocate(), loop],
            [source.tensor, output.tensor],
            {"tl.buffer_versions_map": {shared.tensor: 2}, "tl.buffer_version_mode": {shared.tensor: mode}},
        )
    )
    for module in snapshots.values():
        allocations = [
            binding
            for binding in _nodes(module, ir.Bind)
            if isinstance(binding.value, ir.Call)
            and binding.value.op.name == "tirx.alloc_tensor"
            and binding.var.ty.scope.name == "shared.dyn"
        ]
        assert len(allocations) == 1
    materialized = tvm_ffi.get_global_func("tl.transform.MaterializeMultiBuffer")()(snapshots["InsertSync"])
    assert tirx.analysis.check_well_formed(materialized)
    ir.assert_structural_equal(materialized, ir.load_json(ir.save_json(materialized)))
    shared_allocations = [
        binding.var
        for binding in _nodes(materialized, ir.Bind)
        if isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.alloc_tensor" and binding.var.ty.scope.name == "shared.dyn"
    ]
    assert len(shared_allocations) == 1
    assert [int(axis) for axis in shared_allocations[0].ty.shape] == [2, 64]
    lowered = tvm_ffi.get_global_func("tl.transform.LowerScheduledTIR")()(materialized)
    assert tirx.analysis.check_well_formed(lowered)
    ir.assert_structural_equal(lowered, ir.load_json(ir.save_json(lowered)))
    assert not any(region.op.name == "tl.ascend.schedule_scope" for region in _nodes(lowered, ir.RegionStmt))


def test_native_dynamic_allocation_retains_lexical_shape_definition():
    extent = ir.Var("extent", "int32")
    index = ir.Var("index", "int32")
    size = ir.Var("size", "int32")
    shared = TensorView.create("shared", [size], "float32", scope="shared.dyn")
    loop = ir.For(
        index, 0, extent, 1, ir.ForKind.DEFAULT, [ir.Bind(size, index + 1), shared.allocate(), ir.Evaluate(T.fill(shared.tensor, 0))]
    )
    snapshots = _prefix(_module(loop, [extent]))
    for module in snapshots.values():
        allocations = [
            binding
            for binding in _nodes(module, ir.Bind)
            if isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.alloc_tensor"
        ]
        assert len(allocations) == 1
        assert any(variable.name.startswith("size") for extent in allocations[0].var.ty.shape for variable in _nodes(extent, ir.Var))


def test_guarded_scalar_load_is_never_evaluated_outside_its_condition():
    predicate = ir.Var("predicate", "bool")
    source = TensorView.create("source", [1], "float32")
    output = TensorView.create("output", [1], "float32")
    value = ir.Var("value", "float32")
    conditional = ir.If(predicate, [ir.Bind(value, source.load(0)), output.store(0, value + ir.const(1.0))], None)
    result = _lower(_module(conditional, [predicate, source.tensor, output.tensor]))
    loads = []
    for statement, ancestors in _walk_statements(result["main"].body):
        if isinstance(statement, (ir.Bind, ir.TensorStore, ir.Evaluate)):
            for load in _nodes(statement, ir.TensorLoad):
                if load.source.same_as(source.tensor):
                    loads.append(load)
                    assert any(isinstance(parent, ir.If) for parent in ancestors)
    assert len(loads) == 1


def test_guarded_dynamic_allocation_stays_with_its_consumers():
    predicate = ir.Var("predicate", "bool")
    sizes = TensorView.create("sizes", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    size = ir.Var("size", "int32")
    storage = TensorView.create("dynamic", [size], "int32", scope="local.var")
    conditional = ir.If(
        predicate, [ir.Bind(size, sizes.load(0)), storage.allocate(), storage.store(0, 7), output.store(0, storage.load(0))], None
    )
    result = _lower(_module(conditional, [predicate, sizes.tensor, output.tensor]))
    allocations = []
    for statement, ancestors in _walk_statements(result["main"].body):
        if (
            isinstance(statement, ir.Bind)
            and isinstance(statement.value, ir.Call)
            and statement.value.op.name == "tirx.alloc_tensor"
            and statement.var.name.startswith("dynamic")
        ):
            allocations.append(statement)
            assert any(isinstance(parent, ir.If) for parent in ancestors)
    assert len(allocations) == 1


@pytest.mark.parametrize("begin,end,step", [(0, 4, 1), (6, -2, -2)])
def test_staged_dynamic_allocation_keeps_its_iteration_lifetime(begin, end, step):
    index = ir.Var("index", "int32")
    output = TensorView.create("output", [8], "int32")
    delayed = TensorView.create("delayed", [8], "int32")
    storage = TensorView.create("dynamic", [index + 1], "int32", scope="local.var")
    first = ir.RegionStmt(
        "tl.scope", [], [], {"tl.ascend_stage": 0}, [storage.allocate(), storage.store(0, index), output.store(index, storage.load(0))]
    )
    second = ir.RegionStmt("tl.scope", [], [], {"tl.ascend_stage": 1}, delayed.store(index, index + 10))
    loop = ir.For(index, begin, end, step, ir.ForKind.DEFAULT, [first, second], annotations={"num_stages": 2, "enable_offset": True})
    result = _lower(_module(loop, [output.tensor, delayed.tensor]))
    allocations = [
        statement
        for statement in _nodes(result, ir.Bind)
        if isinstance(statement.value, ir.Call)
        and statement.value.op.name == "tirx.alloc_tensor"
        and statement.var.name.startswith("dynamic")
    ]
    assert len(allocations) == 1
    assert len(allocations[0].var.ty.shape) == 1


def test_guarded_tensor_view_preserves_conditional_pointer_evaluation():
    predicate = ir.Var("predicate", "bool")
    source = TensorView.create("source", [16], "float32")
    output = TensorView.create("output", [1], "float32")
    view = TensorView.create("view", [8], "float32")
    pointer = tirx.ptr_byte_offset(source.data, 32, ty=source.data.ty)
    conditional = ir.If(predicate, [view.declare(pointer), output.store(0, view.load(0))], None)
    result = _lower(_module(conditional, [predicate, source.tensor, output.tensor]))
    pointer_offsets = []
    for statement, ancestors in _walk_statements(result["main"].body):
        if isinstance(statement, ir.Bind):
            for call in _nodes(statement.value, ir.Call):
                if call.op.name == "tirx.ptr_byte_offset":
                    pointer_offsets.append(call)
                    assert any(isinstance(parent, ir.If) for parent in ancestors)
    assert len(pointer_offsets) == 1


def test_mixed_core_lowering_keeps_physical_allocations_shared_and_sid_vector_local():
    source = TensorView.create("source", [2, 16, 16], "float16")
    output = TensorView.create("output", [2, 16, 16], "float16")
    shared = TensorView.create("shared", [16, 16], "float16", scope="shared.dyn")
    level_one = TensorView.create("level_one", [16, 16], "float16", scope="shared.l1")
    sid = ir.Var("sid", "int32")
    ranges = [ir.Range.from_min_extent(sid, 1), ir.Range(0, 16), ir.Range(0, 16)]
    cube_ranges = [ir.Range(0, 1), ir.Range(0, 16), ir.Range(0, 16)]
    body = [
        shared.allocate(),
        level_one.allocate(),
        ir.Evaluate(T.copy(source.region(cube_ranges), level_one.tensor)),
        ir.Evaluate(T.copy(source.region(ranges), shared.tensor)),
        ir.Evaluate(T.copy(shared.tensor, output.region(ranges))),
    ]
    result = _lower(_module(body, [source.tensor, output.tensor], sid=sid))
    scopes = [region.attrs.dict.get("name_hint") for region in _nodes(result, ir.RegionStmt)]
    assert "CUBE" in scopes and "VECTOR" in scopes
    for statement, ancestors in _walk_statements(result["main"].body):
        if isinstance(statement, ir.RegionStmt) and statement.op.name == "tirx.launch_thread" and statement.args[0].value == "cthread":
            assert any(isinstance(parent, ir.RegionStmt) and parent.attrs.dict.get("name_hint") == "VECTOR" for parent in ancestors)
        if isinstance(statement, ir.Bind) and isinstance(statement.value, ir.Call) and statement.value.op.name == "tirx.alloc_tensor":
            assert not any(
                isinstance(parent, ir.RegionStmt) and parent.attrs.dict.get("name_hint") in ["CUBE", "VECTOR"] for parent in ancestors
            )
