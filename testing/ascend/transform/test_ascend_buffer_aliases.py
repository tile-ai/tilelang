"""Manual alias inference uses completion edges on the accessing hardware pipe."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang._compat import stmt_functor


def _nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _tensor(name, scope="shared.dyn"):
    return ir.Var(name, tirx.TensorType([64], "float32", scope=scope))


def _allocate(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _pointer(tensor, mask):
    return ir.Call(
        "tirx.ptr_byte_offset",
        [ir.Call("tirx.tensor_data_ptr", [tensor]), ir.const(0, "int64")],
        ty=ir.PtrType(tensor.ty.dtype, tensor.ty.scope),
        attrs={"tl.access_mask": mask, "tl.access_extent": 64},
    )


def _call(name, *args):
    return ir.Evaluate(
        ir.Call("tl." + name, [ir.StringImm(value) if isinstance(value, str) else value for value in args], ty=ir.TupleType([]))
    )


def _load(global_tensor, local_tensor):
    return _call("ascend_copy_gm_to_ubuf", _pointer(local_tensor, 2), _pointer(global_tensor, 1), 0, 1, 256, 0, 0, 0, 0, 0, 0)


def _store(local_tensor, global_tensor):
    return _call("ascend_copy_ubuf_to_gm", _pointer(global_tensor, 2), _pointer(local_tensor, 1), 0, 1, 256, 0, 0, 0)


def _flags(pipe, event_id=0):
    return [_call("ascend_set_flag", pipe, event_id), _call("ascend_wait_flag", pipe, event_id)]


def _scope(name, body):
    return ir.RegionStmt("tl.scope", [], [], {"name_hint": name}, body)


def _infer(body, parameters=(), disable_reuse=False):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([])).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    with ir.transform.PassContext(config={"tl.disable_shared_memory_reuse": disable_reuse}):
        result = tvm_ffi.get_global_func("tl.transform.InferBufferAliases")()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    restored = ir.load_json(ir.save_json(result))
    ir.assert_structural_equal(result, restored)
    assert tirx.analysis.check_well_formed(restored)
    return result


def _module(tensors, body, global_tensor, parameters=()):
    return _infer([*[_allocate(tensor) for tensor in tensors], *body], [global_tensor, *parameters])


def _may_alias(function, first, second):
    attrs = (
        [function.attrs]
        + [region.attrs for region in _nodes(function, ir.RegionStmt)]
        + [loop.annotations for loop in _nodes(function, ir.For)]
    )
    contracts = [annotation["tl.buffer_alias_map"] for annotation in attrs if "tl.buffer_alias_map" in annotation]
    assert contracts
    return any(first in contract and any(other.same_as(second) for other in contract[first]) for contract in contracts)


@pytest.mark.parametrize("release", ["none", "acquire-only", "MTE3_MTE2", "PIPE_ALL"])
def test_dma_read_requires_a_completion_edge_before_reuse(release):
    global_tensor, first, second = _tensor("global", "global"), _tensor("first"), _tensor("second")
    body = [_load(global_tensor, first), *_flags("MTE2_MTE3"), _store(first, global_tensor)]
    if release == "acquire-only":
        body.append(_call("ascend_wait_flag", "MTE3_MTE2", 0))
    elif release == "MTE3_MTE2":
        body.extend(_flags(release))
    elif release == "PIPE_ALL":
        body.append(_call("ascend_pipe_barrier", "PIPE_ALL"))
    body.extend([_load(global_tensor, second), *_flags("MTE2_MTE3"), _store(second, global_tensor)])
    after = _module([first, second], body, global_tensor)
    assert _may_alias(after, first, second) == (release in ("MTE3_MTE2", "PIPE_ALL"))


@pytest.mark.parametrize("release_pipe", ["V_MTE2", "S_MTE2"])
def test_pad_value_operand_is_scalar_even_inside_vector_block(release_pipe):
    global_tensor, first, second = _tensor("global", "global"), _tensor("first"), _tensor("second")
    body = [
        _load(global_tensor, first),
        *_flags("MTE2_S"),
        _call("ascend_set_copy_pad_value", ir.TensorLoad(first, [0])),
        *_flags(release_pipe),
        _load(global_tensor, second),
    ]
    after = _module([first, second], [_scope("VECTOR", body)], global_tensor)
    assert _may_alias(after, first, second) == (release_pipe == "S_MTE2")


@pytest.mark.parametrize("release_pipe", ["PIPE_MTE1", "PIPE_FIX"])
def test_cross_core_handshake_only_releases_its_local_pipe(release_pipe):
    global_tensor = _tensor("global", "global")
    first, second = [_tensor(name, "shared.l1") for name in ("first", "second")]
    left = _tensor("left", "shared.l0a")
    cube_read = _scope(
        "CUBE",
        [
            _call("ascend_copy_gm_to_cbuf", _pointer(first, 2), _pointer(global_tensor, 1), 0, 16, 0, 4, 16, 0, 0, 0, 4, "float32"),
            *_flags("MTE2_MTE1"),
            _call("ascend_load_cbuf_to_ca", _pointer(left, 2), _pointer(first, 1)),
            _call("ascend_cross_core_set_flag", 4, release_pipe, 0),
        ],
    )
    vector = _scope(
        "VECTOR",
        [
            _call("ascend_cross_core_wait_flag", 4, "PIPE_MTE3", 0),
            _call("ascend_cross_core_set_flag", 4, "PIPE_MTE3", 1),
        ],
    )
    cube_write = _scope(
        "CUBE",
        [
            _call("ascend_cross_core_wait_flag", 4, "PIPE_MTE2", 1),
            _call("ascend_copy_gm_to_cbuf", _pointer(second, 2), _pointer(global_tensor, 1), 0, 16, 0, 4, 16, 0, 0, 0, 4, "float32"),
        ],
    )
    after = _module([first, second, left], [cube_read, vector, cube_write], global_tensor)
    assert _may_alias(after, first, second) == (release_pipe == "PIPE_MTE1")


def test_native_alias_declarations_do_not_extend_storage_lifetimes():
    global_tensor, first, second = _tensor("global", "global"), _tensor("first"), _tensor("second")
    view = _tensor("view")
    pointer = ir.Var("pointer", ir.PtrType(first.ty.dtype, first.ty.scope))
    body = [ir.Bind(pointer, _pointer(first, 1)), ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty]))]
    body += [
        _load(global_tensor, view),
        *_flags("MTE2_MTE3"),
        _store(view, global_tensor),
        *_flags("MTE3_MTE2"),
        _load(global_tensor, second),
    ]
    after = _module([first, second], body, global_tensor)
    assert _may_alias(after, first, second)


def test_late_alias_value_read_prevents_reuse_of_backing_storage():
    global_tensor, first, second = _tensor("global", "global"), _tensor("first"), _tensor("second")
    view = _tensor("view")
    body = [ir.Bind(view, ir.Call("tirx.decl_tensor", [_pointer(first, 1)], ty_args=[view.ty]))]
    body += [
        _load(global_tensor, first),
        *_flags("MTE2_MTE3"),
        _store(first, global_tensor),
        *_flags("MTE3_MTE2"),
        _load(global_tensor, second),
    ]
    body.append(_call("ascend_set_copy_pad_value", ir.TensorLoad(view, [0])))
    after = _module([first, second], body, global_tensor)
    assert not _may_alias(after, first, second)


@pytest.mark.parametrize("name", ["parameter", "sid", "index"])
def test_unrelated_symbolic_event_ids_are_not_substituted_with_loop_values(name):
    global_tensor, first, second = _tensor("global", "global"), _tensor("first"), _tensor("second")
    event_id = ir.Var(name, "int32")
    body = [_load(global_tensor, first), *_flags("MTE2_MTE3"), _store(first, global_tensor)]
    body += [_call("ascend_set_flag", "MTE3_MTE2", event_id), _call("ascend_wait_flag", "MTE3_MTE2", 0), _load(global_tensor, second)]
    after = _module([first, second], body, global_tensor, [event_id])
    assert not _may_alias(after, first, second)


@pytest.mark.parametrize("conditional", ["if", "empty_loop", "dynamic_loop", "while"])
def test_optional_completion_edges_do_not_fabricate_reuse(conditional):
    global_tensor, first, second = _tensor("global", "global"), _tensor("first"), _tensor("second")
    flag, index = ir.Var("flag", "int32"), ir.Var("index", "int32")
    barrier = _call("ascend_pipe_barrier", "PIPE_ALL")
    if conditional == "if":
        guarded = ir.If(flag > 0, barrier, None)
    elif conditional == "while":
        guarded = ir.While(flag > 0, [barrier, ir.Break()])
    else:
        guarded = ir.For(index, 0, flag if conditional == "dynamic_loop" else 0, 1, ir.ForKind.DEFAULT, barrier)
    body = [_load(global_tensor, first), *_flags("MTE2_MTE3"), _store(first, global_tensor), guarded, _load(global_tensor, second)]
    after = _module([first, second], body, global_tensor, [flag])
    assert not _may_alias(after, first, second)


def test_contracts_stay_with_nested_loop_definitions_and_merge_cleanly():
    global_tensor, first, second = _tensor("global", "global"), _tensor("first"), _tensor("second")
    index = ir.Var("index", "int32")
    body = [
        _allocate(first),
        _allocate(second),
        _load(global_tensor, first),
        *_flags("MTE2_MTE3"),
        _store(first, global_tensor),
        *_flags("MTE3_MTE2"),
        _load(global_tensor, second),
    ]
    after = _infer(ir.For(index, 0, 1, 1, ir.ForKind.DEFAULT, body), [global_tensor])
    assert "tl.buffer_alias_map" in after.body.seq[0].annotations
    assert _may_alias(after, first, second)
    merged = tvm_ffi.get_global_func("tl.transform.MergeUBAllocations")(16, False)(ir.IRModule({"main": after}))["main"]
    assert tirx.analysis.check_well_formed(merged)
    allocations = [
        binding
        for binding in _nodes(merged, ir.Bind)
        if isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.alloc_tensor"
    ]
    assert len(allocations) == 1
    assert int(allocations[0].var.ty.shape[0]) == 256


def test_disabling_reuse_keeps_an_explicit_empty_compatibility_contract():
    first, second = _tensor("first"), _tensor("second")
    result = _infer([_allocate(first), _allocate(second)], disable_reuse=True)
    assert not _may_alias(result, first, second)
    assert len(result.attrs["tl.buffer_alias_map"]) == 2
