import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx

from tilelang._compat import ir, stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def lower(body, parameters=()):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([])).with_attr("global_symbol", "allocation_test")
    module = tvm_ffi.get_global_func("tl.transform.PlanAndUpdateBufferAllocationLocation")()(ir.IRModule({"main": function}))
    assert ir.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def test_private_scratch_sinks_to_innermost_use_and_executes(compile_tilelang):
    scratch = TensorView.create("scratch", [1], "int32", scope="local")
    output = TensorView.create("output", [3, 4], "int32")
    outer, inner = ir.Var("outer", "int32"), ir.Var("inner", "int32")
    body = ir.For(
        outer,
        0,
        3,
        1,
        ir.ForKind.DEFAULT,
        ir.For(inner, 0, 4, 1, ir.ForKind.DEFAULT, [scratch.store(0, outer * 10 + inner), output.store([outer, inner], scratch.load(0))]),
    )
    module = lower([scratch.allocate(), body], [output.tensor])
    inner_loop = nodes(module, ir.For)[0]
    assert isinstance(inner_loop.body.seq[0], ir.Bind)
    assert inner_loop.body.seq[0].var.same_as(scratch.tensor)
    executable = compile_tilelang(module, target="c")
    actual = runtime.empty((3, 4), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(3)[:, None] * 10 + np.arange(4))


def test_loop_header_and_carried_state_keep_allocation_outside(compile_tilelang):
    bound = TensorView.create("bound", [1], "int32", scope="local")
    output = TensorView.create("output", [4], "int32")
    index = ir.Var("index", "int32")
    body = [bound.allocate(), bound.store(0, 4), ir.For(index, 0, bound.load(0), 1, ir.ForKind.DEFAULT, output.store(index, bound.load(0)))]
    module = lower(body, [output.tensor])
    loop = nodes(module, ir.For)[0]
    assert not nodes(loop.body, ir.Bind)
    executable = compile_tilelang(module, target="c")
    actual = runtime.empty((4,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [4, 4, 4, 4])


@pytest.mark.parametrize("scope", ["shared", "shared.barrier", "shared.cluster_barrier"])
def test_shared_lifetime_is_not_narrowed_into_conditional_or_loop(scope):
    scratch = TensorView.create("scratch", [1], "int32", scope=scope)
    condition = ir.Var("condition", "bool")
    index = ir.Var("index", "int32")
    branch = ir.If(condition, ir.For(index, 0, 2, 1, ir.ForKind.DEFAULT, scratch.store(0, index)), None)
    module = lower([scratch.allocate(), branch], [condition])
    assert not nodes(nodes(module, ir.If)[0], ir.Bind)


def test_pointer_alias_keeps_backing_allocation_lifetime():
    scratch = TensorView.create("scratch", [1], "int32", scope="local")
    pointer = ir.Var("pointer", ir.PtrType("int32", scope="local"))
    index = ir.Var("index", "int32")
    body = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        [ir.Bind(pointer, scratch.pointer_at(0)), ir.Evaluate(tirx.call_extern("consume", pointer, ty="void"))],
    )
    module = lower([scratch.allocate(), body])
    allocation = nodes(module, ir.Bind)[0]
    assert allocation.var.same_as(scratch.tensor)
    assert len(nodes(nodes(module, ir.For)[0], ir.Bind)) == 1


def test_allocation_initializers_are_not_moved_across_writes():
    scratch = TensorView.create("scratch", [1], "int32", scope="local")
    index = ir.Var("index", "int32")
    allocation = ir.Bind(
        scratch.tensor, ir.Call("tirx.alloc_tensor", [], attrs={"tl.local_var_init": ir.const(5)}, ty_args=[scratch.tensor.ty])
    )
    module = lower([allocation, ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, scratch.store(0, scratch.load(0) + index))])
    assert not nodes(nodes(module, ir.For)[0], ir.Bind)


def test_private_scratch_does_not_sink_into_vectorized_loop(compile_tilelang):
    scratch = TensorView.create("scratch", [4], "int32", scope="local")
    output = TensorView.create("output", [4], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.VECTORIZED,
        [scratch.store(index, index + 1), output.store(index, scratch.load(index))],
    )
    module = lower([scratch.allocate(), loop], [output.tensor])
    assert module["main"].body.seq[0].var.same_as(scratch.tensor)
    assert not nodes(nodes(module, ir.For)[0], ir.Bind)
    executable = compile_tilelang(module, target="c")
    actual = runtime.empty((4,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(4) + 1)


@pytest.mark.parametrize("reset", ["first", "every"])
def test_loop_carried_storage_is_only_sunk_after_a_complete_reset(reset, compile_tilelang):
    scratch = TensorView.create("scratch", [2], "int32", scope="local")
    output = TensorView.create("output", [4], "int32")
    index = ir.Var("index", "int32")
    element = ir.Var("element", "int32")
    initialize = ir.For(element, 0, 2, 1, ir.ForKind.DEFAULT, scratch.store(element, 1))
    if reset == "first":
        initialize = ir.If(index == 0, initialize, None)
    body = ir.For(
        index, 0, 4, 1, ir.ForKind.DEFAULT, [initialize, output.store(index, scratch.load(0)), scratch.store(0, scratch.load(0) + 1)]
    )
    module = lower([scratch.allocate(), body], [output.tensor])
    outer = nodes(module, ir.For)[-1]
    assert bool(nodes(outer.body, ir.Bind)) == (reset == "every")
    executable = compile_tilelang(module, target="c")
    actual = runtime.empty((4,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [1, 1, 1, 1] if reset == "every" else [1, 2, 3, 4])


@pytest.mark.parametrize("exit_kind", [ir.Break, ir.Continue])
def test_incomplete_reset_with_early_exit_keeps_loop_carried_storage(exit_kind, compile_tilelang):
    scratch = TensorView.create("scratch", [2], "int32", scope="local")
    output = TensorView.create("output", [4], "int32")
    index = ir.Var("index", "int32")
    element = ir.Var("element", "int32")
    initialize = ir.For(
        element,
        0,
        2,
        1,
        ir.ForKind.DEFAULT,
        [ir.If(index != 0, exit_kind(), None), scratch.store(element, 1)],
    )
    body = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        [initialize, output.store(index, scratch.load(0)), scratch.store(0, scratch.load(0) + 1)],
    )
    module = lower([scratch.allocate(), body], [output.tensor])
    outer = nodes(module, ir.For)[-1]
    assert not nodes(outer.body, ir.Bind)
    executable = compile_tilelang(module, target="c")
    actual = runtime.empty((4,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [1, 2, 3, 4])
