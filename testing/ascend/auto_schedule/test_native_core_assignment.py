import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.target import Target

import tilelang.ascend.language as T
from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


def _nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _module(body, params=(), sid=None, attrs=None, prefix=()):
    root = ir.RegionStmt("tl.scope", [], [], {"name_hint": "tilelang_root", **(attrs or {})}, body)
    if sid is not None:
        root = ir.RegionStmt("tl.launch_grid", ["cthread", 2], [sid], {}, [*prefix, root])
    function = tirx.Function(list(params), root, ret_type=ir.TupleType([]))
    function = function.with_attr("target", Target("ascend")).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    return ir.IRModule({"main": function})


def _assign(module):
    for name in ["MaterializeScheduleUnits", "AssignCore"]:
        module = tvm_ffi.get_global_func("tl.transform." + name)()(module)
        assert tirx.analysis.check_well_formed(module)
        ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def _task_copies(module):
    result = []
    for region in _nodes(module, ir.RegionStmt):
        metadata = region.attrs.dict.get("tl.ascend_task")
        if metadata is not None:
            for call in _nodes(region.body, ir.Call):
                if call.op.name in ["tl.tileop.copy", "tl.tileop.ascend_copy"]:
                    result.append((metadata, call))
    return result


def _root(module):
    return next(region for region in _nodes(module, ir.RegionStmt) if region.attrs.dict.get("name_hint") == "tilelang_root")


def test_assigns_vector_and_cube_dma_to_their_own_cores():
    source = TensorView.create("source", [64], "float32")
    output = TensorView.create("output", [64], "float32")
    shared = TensorView.create("shared", [64], "float32", scope="shared.dyn")
    level_one = TensorView.create("level_one", [64], "float32", scope="shared.l1")
    body = [
        shared.allocate(),
        level_one.allocate(),
        ir.Evaluate(T.copy(source.tensor, shared.tensor)),
        ir.Evaluate(T.copy(shared.tensor, output.tensor)),
        ir.Evaluate(T.copy(source.tensor, level_one.tensor)),
    ]
    result = _assign(_module(body, [source.tensor, output.tensor]))
    assert [int(metadata["core_mask"]) for metadata, _ in _task_copies(result)] == [5, 5, 2]


def test_uniform_mixed_vector_work_does_not_force_projection():
    source = TensorView.create("source", [64], "float32")
    shared = TensorView.create("shared", [64], "float32", scope="shared.dyn")
    sid = ir.Var("sid", "int32")
    result = _assign(_module([shared.allocate(), ir.Evaluate(T.copy(source.tensor, shared.tensor))], [source.tensor], sid))
    assert "tl.split_subcores" not in _root(result).attrs.dict
    assert [int(metadata["core_mask"]) for metadata, _ in _task_copies(result)] == [5]


def test_subcore_guards_split_storage_without_losing_scalar_definitions():
    source = TensorView.create("source", [128], "float32")
    output = TensorView.create("output", [128], "float32")
    shared = TensorView.create("shared", [64], "float32", scope="shared.dyn")
    sid = ir.Var("sid", "int32")
    offset = ir.Var("offset", "int32")
    source_region = source.region([ir.Range.from_min_extent(offset, 64)])
    output_region = output.region([ir.Range.from_min_extent(offset, 64)])
    producer = ir.Evaluate(T.copy(source_region, shared.tensor))
    body = [
        shared.allocate(),
        ir.Bind(offset, sid * 64),
        ir.If(sid == 0, producer, producer),
        ir.Evaluate(T.copy(shared.tensor, output_region)),
    ]
    result = _assign(_module(body, [source.tensor, output.tensor], sid, {"tl.buffer_versions_map": {shared.tensor: 2}}))
    root_attrs = _root(result).attrs.dict
    assert root_attrs["tl.split_subcores"]
    assert sorted(int(value) for value in root_attrs["tl.aiv_storage_map"].values()) == [0, 1]
    assert len(root_attrs["tl.buffer_versions_map"]) == 2
    assert all(int(value) == 2 for value in root_attrs["tl.buffer_versions_map"].values())
    copies = _task_copies(result)
    assert len(copies) == 4
    assert sorted(int(metadata["core_mask"]) for metadata, _ in copies) == [1, 1, 4, 4]
    assert not any(variable.same_as(sid) for variable in _nodes(_root(result).body, ir.Var))


def test_projected_loops_and_thread_regions_get_fresh_binders():
    shared = TensorView.create("shared", [32], "float32", scope="shared.dyn")
    sid = ir.Var("sid", "int32")
    index = ir.Var("index", "int32")
    thread = ir.Var("thread", "int32")
    launch = ir.RegionStmt("tirx.launch_thread", ["threadIdx.x", 32], [thread], {}, shared.store(thread, ir.const(1.0)))
    vector = ir.RegionStmt("tl.scope", [], [], {"name_hint": "SIMT_VF"}, launch)
    loop = ir.For(index, 4, sid, -1, ir.ForKind.DEFAULT, vector)
    result = _assign(_module([shared.allocate(), loop], sid=sid))
    loops = _nodes(result, ir.For)
    assert len(loops) == 2
    assert not loops[0].var.same_as(loops[1].var)
    threads = [region.body_params[0] for region in _nodes(result, ir.RegionStmt) if region.op.name == "tirx.launch_thread"]
    assert len(threads) == 2
    assert not threads[0].same_as(threads[1])


@pytest.mark.parametrize("route,count", [(None, 2), (0, 1)])
def test_fixpipe_projection_keeps_cube_execution_and_aiv_routing(route, count):
    accum = TensorView.create("accum", [16, 16], "float32", scope="shared.l0c")
    shared = TensorView.create("shared", [16, 16], "float32", scope="shared.dyn")
    sid = ir.Var("sid", "int32")
    copy = ir.Evaluate(T.copy(accum.tensor, shared.tensor, sub_blockid=sid if route is None else route))
    result = _assign(_module([accum.allocate(), shared.allocate(), copy], sid=sid))
    copies = _task_copies(result)
    assert len(copies) == count
    assert all(int(metadata["core_mask"]) == 2 for metadata, _ in copies)
    assert sorted(int(call.attrs.dict["sub_blockid"]) for _, call in copies) == list(range(count))


def test_projected_tensor_types_use_their_subcore_shape():
    sid = ir.Var("sid", "int32")
    shared = TensorView.create("shared", [16 + sid], "float32", scope="shared.dyn")
    fill = ir.Evaluate(T.fill(shared.tensor, 1.0))
    result = _assign(_module([shared.allocate(), ir.If(sid == 0, fill, fill)], sid=sid))
    allocations = [
        binding
        for binding in _nodes(result, ir.Bind)
        if isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.alloc_tensor"
    ]
    analyzer = sym.Analyzer()
    assert sorted(int(analyzer.simplify(binding.var.ty.shape[0])) for binding in allocations) == [16, 17]


def test_projected_external_tensor_views_have_explicit_declarations():
    source = TensorView.create("source", [64], "float32")
    shared = TensorView.create("shared", [16], "float32", scope="shared.dyn")
    sid = ir.Var("sid", "int32")
    view = TensorView.create("view", [16 + sid], "float32")
    declaration = ir.Bind(view.tensor, ir.Call("tirx.decl_tensor", [source.data], ty_args=[view.tensor.ty]))
    copy = ir.Evaluate(T.copy(view.region([ir.Range(0, 16)]), shared.tensor))
    result = _assign(_module([shared.allocate(), ir.If(sid == 0, copy, copy)], [source.tensor], sid, prefix=[declaration]))
    declarations = [
        binding
        for binding in _nodes(_root(result).body, ir.Bind)
        if isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.decl_tensor"
    ]
    assert len(declarations) == 2
    assert all(binding.var.name.startswith("view_aiv") for binding in declarations)


def test_fixpipe_projection_rejects_replayed_external_side_effects():
    accum = TensorView.create("accum", [16, 16], "float32", scope="shared.l0c")
    shared = TensorView.create("shared", [16, 16], "float32", scope="shared.dyn")
    output = TensorView.create("output", [1], "float32")
    sid = ir.Var("sid", "int32")
    task = ir.RegionStmt(
        "tl.scope",
        [],
        [],
        {"tl.ascend_task": {}},
        [ir.Evaluate(T.copy(accum.tensor, shared.tensor, sub_blockid=sid)), output.store(0, ir.const(1.0))],
    )
    with pytest.raises(Exception, match="external side effects"):
        _assign(_module([accum.allocate(), shared.allocate(), task], [output.tensor], sid))


def test_assign_core_rejects_reapplication():
    source = TensorView.create("source", [16], "float32")
    shared = TensorView.create("shared", [16], "float32", scope="shared.dyn")
    result = _assign(_module([shared.allocate(), ir.Evaluate(T.copy(source.tensor, shared.tensor))], [source.tensor]))
    with pytest.raises(Exception, match="must run once"):
        tvm_ffi.get_global_func("tl.transform.AssignCore")()(result)


def _binding_core(module, name):
    return next(
        int(region.attrs["tl.ascend_task"]["core_mask"])
        for region in _nodes(module, ir.RegionStmt)
        if "tl.ascend_task" in region.attrs
        for binding in _nodes(region.body, ir.Bind)
        if binding.var.name.startswith(name)
    )


def test_register_pointer_availability_does_not_depend_on_stored_contents():
    registers = TensorView.create("registers", [1], "int32", scope="local.var")
    cube = TensorView.create("cube", [1], "int32", scope="shared.l1")
    pointer = ir.Var("pointer", registers.data.ty)
    result = _assign(_module([registers.allocate(), cube.allocate(), registers.store(0, cube.load(0)), ir.Bind(pointer, registers.data)]))
    assert _binding_core(result, "pointer") == 7


@pytest.mark.parametrize("through_alias", [False, True])
def test_register_content_availability_uses_canonical_storage(through_alias):
    registers = TensorView.create("registers", [1], "int32", scope="local.var")
    alias = TensorView.create("alias", [1], "int32", scope="local.var")
    cube = TensorView.create("cube", [1], "int32", scope="shared.l1")
    value = ir.Var("loaded_value", "int32")
    result = _assign(
        _module(
            [
                registers.allocate(),
                cube.allocate(),
                alias.declare(registers.data),
                registers.store(0, cube.load(0)),
                ir.Bind(value, (alias if through_alias else registers).load(0)),
            ]
        )
    )
    assert _binding_core(result, "loaded_value") == 2


def test_computed_alias_pointer_keeps_its_own_core_availability():
    registers = TensorView.create("registers", [4], "int32", scope="local.var")
    alias = TensorView.create("alias", [1], "int32", scope="local.var")
    cube = TensorView.create("cube", [1], "int32", scope="shared.l1")
    value = ir.Var("loaded_value", "int32")
    pointer = tirx.ptr_byte_offset(registers.data, cube.load(0) * 4, ty=registers.data.ty)
    result = _assign(_module([registers.allocate(), cube.allocate(), alias.declare(pointer), ir.Bind(value, alias.load(0))]))
    assert _binding_core(result, "alias") == 2
    assert _binding_core(result, "loaded_value") == 2
