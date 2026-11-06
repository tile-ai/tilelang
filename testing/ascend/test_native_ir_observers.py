"""Ascend observers use native identities, typed operands, and lexical Binds."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from testing.ascend._ir import (
    access_tensor,
    allocate,
    allocated_tensors,
    allocations,
    annotation,
    call_annotations,
    copy,
    kernel,
    kernel_body,
    kernel_bodies,
    launch_thread,
    disjoint_argument_fixture,
    loop_annotations,
    nodes,
    region,
    sblock,
    scope_annotations,
    scheduled_bind,
    scheduled_fields,
    substitute,
    tensor,
    walk_ir,
)


def test_access_tensor_keeps_distinct_view_identities():
    owner = tensor((4,), "int32", name="same_name")
    view = tensor((4,), "int32", name="same_name")
    for access in (owner, owner[0], ir.TensorStore(owner, [0], 3), region(owner, 1), allocate(owner)):
        assert access_tensor(access).same_as(owner)
        assert not access_tensor(access).same_as(view)
    for invalid in (ir.Var("scalar", "int32"), ir.const(1), ir.Evaluate(ir.const(0))):
        with pytest.raises(TypeError, match="native tensor"):
            access_tensor(invalid)


def test_allocations_keep_actual_and_scheduled_binding_contracts_separate():
    a, b, alias = [tensor((4,), "int32", name=name) for name in ("a", "b", "alias")]
    direct = allocate(a)
    pending = scheduled_bind(allocate(b), 2, {"core_mask": 7})
    # A tensor alias Bind does not own an allocation even though its result is a tensor.
    body = ir.SeqStmt([direct, pending, ir.Bind(alias, a), ir.TensorStore(alias, [0], 1)])
    assert len(allocations(body)) == 1
    assert allocations(body)[0].same_as(direct)
    selected = allocations(body, include_scheduled=True)
    assert [node.same_as(expected) for node, expected in zip(selected, [direct, pending])] == [True, True]
    assert int(scheduled_fields(selected[1])["stage"]) == 2
    assert allocated_tensors(body)[0].same_as(a)
    assert allocated_tensors(body, include_scheduled=True)[1].same_as(b)


def test_allocations_preserve_lexical_order_in_nested_regions():
    a, b = [tensor((4,), "int32", name=name) for name in ("a", "b")]
    body = ir.SeqStmt([allocate(a), ir.If(ir.const(True), [allocate(b)], None)])
    assert [var.name for var in allocated_tensors(body)] == ["a", "b"]


def test_walk_ir_visits_repeated_accesses_before_their_enclosing_statement():
    a = tensor((4,), "int32", name="a")
    load = a[0]
    store = ir.TensorStore(a, [0], load + load)
    visited = []
    walk_ir(store, visited.append)
    assert sum(isinstance(node, ir.TensorLoad) for node in visited) == 2
    assert visited[-1].same_as(store)
    assert all(isinstance(node, (ir.Expr, ir.Stmt)) for node in visited)


def test_substitute_uses_identity_and_is_simultaneous():
    a, b = ir.Var("n", "int32"), ir.Var("n", "int32")
    replacement = substitute(a + b, {a: b, b: 7})
    assert replacement.a.same_as(b)
    assert int(replacement.b) == 7
    assert int(substitute(b, {a: 4, b: 7})) == 7
    assert substitute(b, {a: 4}).same_as(b)


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "int64"])
def test_substitute_preserves_dtype_and_explicit_cast(dtype):
    n = ir.Var("n", dtype)
    value = ir.prim.Cast("int64", n)
    result = substitute(value, {n: 3})
    assert isinstance(result, ir.prim.Cast)
    assert str(result.ty.dtype) == "int64"
    assert str(result.value.ty.dtype) == dtype
    assert int(result.value) == 3


def test_substitute_updates_tensor_types_and_all_uses_without_mutating_input():
    n = ir.Var("n", "int32")
    a = tensor((n, 4), "int32", name="a")
    body = ir.SeqStmt([allocate(a), ir.TensorStore(a, [0, 0], a[0, 0])])
    after = substitute(body, {n: 8})
    (new_a,) = allocated_tensors(after)
    assert not new_a.same_as(a)
    assert tuple(int(dim) for dim in new_a.ty.shape) == (8, 4)
    assert new_a.ty.same_as(allocations(after)[0].value.ty_args[0]) or tvm_ffi.structural_equal(
        new_a.ty, allocations(after)[0].value.ty_args[0]
    )
    assert nodes(after, ir.TensorLoad)[0].source.same_as(new_a)
    assert nodes(after, ir.TensorStore)[0].dest.same_as(new_a)
    layout_uses = []
    tvm_ffi.structural_walk(new_a.ty.layout, (ir.Var, layout_uses.append))
    assert not layout_uses
    assert a.ty.shape[0].same_as(n)
    assert nodes(body, ir.TensorLoad)[0].source.same_as(a)
    # A free tensor use has no definition node at which to trigger type remapping.
    free = substitute(a[0, 0], {n: 9})
    assert int(free.source.ty.shape[0]) == 9


def test_substitute_rejects_dtype_changes_and_constant_binders():
    n = ir.Var("n", "int32")
    with pytest.raises(TypeError, match="dtype"):
        substitute(n, {n: ir.const(2, "int64")})
    with pytest.raises(ValueError, match="bound variable"):
        substitute(ir.For(n, 0, 4, 1, ir.ForKind.DEFAULT, [ir.Evaluate(n)]), {n: 3})


def test_call_annotations_decode_operand_payload_with_identity():
    a, b = tensor((4,), "int32", name="a"), tensor((4,), "int32", name="b")
    call = copy(a, b, annotations={"transpose": 1, "alias": {a: b}}).value
    fields = call_annotations(call)
    assert int(fields["transpose"]) == 1
    assert fields["alias"][a].same_as(b)
    assert call.args[0].source.same_as(a)
    assert call.args[1].source.same_as(b)
    assert int(call.attrs["operand_count"]) == 2
    assert call_annotations(allocate(a).value) == {}


def test_call_annotations_reject_malformed_metadata_shape():
    call = ir.Call("tirx.alloc_tensor", [], ty_args=[tirx.TensorType((4,), "int32")])
    with pytest.raises(TypeError, match="annotated Call"):
        call_annotations(ir.Evaluate(call))


def test_scope_annotations_keep_nested_block_and_loop_metadata_private():
    a = tensor((4,), "int32", name="a")
    i = ir.Var("i", "int32")
    child = sblock("child", [annotation({"child_only": a, "key": 99})], annotations={"constant_child_only": 7})
    loop = ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, [annotation({"loop_only": a, "key": 88})])
    parent = sblock("parent", [annotation({"key": 2, "direct_operand": a}), child, loop], annotations={"key": 1, "constant": 3})
    fields = scope_annotations(parent)
    assert set(fields) == {"key", "constant", "direct_operand"}
    assert int(fields["key"]) == 2
    assert fields["direct_operand"].same_as(a)
    assert int(scope_annotations(child)["key"]) == 99
    assert scope_annotations(ir.SeqStmt([parent])) == {}
    assert scope_annotations(ir.SeqStmt([parent, child])) == {}
    direct = ir.SeqStmt([annotation({"direct_operand": a}), parent, child])
    assert scope_annotations(direct)["direct_operand"].same_as(a)


@pytest.mark.parametrize("core,cost", [(7, None), (0, None), (3, (1, 2))])
def test_kernel_allocation_metadata_remains_explicit(core, cost):
    a = tensor((4,), "int32", name="a")
    mod = kernel(ir.TensorStore(a, [0], 1), buffers=[a], allocation_stage=0, allocation_core=core, allocation_cost=cost)
    assert isinstance(mod["main"].body, ir.SeqStmt)
    assert kernel_body(mod).same_as(mod["main"].body)
    assert not nodes(mod, ir.RegionStmt)
    assert allocations(mod) == []
    (binding,) = allocations(mod, include_scheduled=True)
    fields = scheduled_fields(binding)
    assert int(fields["stage"]) == 0
    assert ("core_mask" in fields["task"]) == bool(core)
    if core:
        assert int(fields["task"]["core_mask"]) == core
    if cost is None:
        assert "latency" not in fields["task"] and "ii" not in fields["task"]
    else:
        assert (int(fields["task"]["latency"]), int(fields["task"]["ii"])) == cost


def test_kernel_bodies_keep_sibling_launches_and_alias_preconditions_independent():
    a, b = [tensor((4,), "int32", name=name) for name in ("a", "b")]
    bodies = []
    for index in range(2):
        i = ir.Var(f"i{index}", "int32")
        body = ir.SeqStmt([ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, [ir.TensorStore(b, [i], a[i])])])
        bodies.append(launch_thread(ir.Var(f"block{index}", "int32"), "blockIdx.x", 1, body))
    before = kernel(bodies, params=[a, b])
    actual = kernel_bodies(before)
    assert len(actual) == 2 and all(body.same_as(scope.body) for body, scope in zip(actual, bodies, strict=True))
    with pytest.raises(ValueError):
        kernel_body(before)
    marked = disjoint_argument_fixture(before, ("a", "b"))
    assert all(not scope_annotations(body) for body in kernel_bodies(before))
    for body in kernel_bodies(marked):
        (pair,) = scope_annotations(body)["conflict_hint"]
        assert pair[0].same_as(a) and pair[1].same_as(b)
        (loop,) = nodes(body, ir.For)
        (loop_pair,) = loop_annotations(loop)["conflict_hint"]
        assert loop_pair[0].same_as(a) and loop_pair[1].same_as(b)
