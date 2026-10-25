import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def lower(body, parameters=(), *, static=False, reuse=True, aggressive=False, alignments=None):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([])).with_attr("global_symbol", "arena_kernel")
    if alignments is not None:
        function = function.with_attr("tl.smem_alignment_map", alignments)
    with PassContext(config={"tirx.merge_static_smem": static}):
        module = tvm_ffi.get_global_func("tl.transform.MergeSharedMemoryAllocations")(aggressive, 16, not reuse)(
            ir.IRModule({"main": function})
        )
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def allocations(module):
    result = []
    stmt_functor.post_order_visit(
        module,
        lambda node: result.append(node)
        if isinstance(node, ir.Bind) and isinstance(node.value, ir.Call) and node.value.op.name == "tirx.alloc_tensor"
        else None,
    )
    return result


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
def test_disjoint_lifetimes_share_an_aligned_arena(scope, reuse):
    first = TensorView.create("first", [128], "int32", scope=scope)
    second = TensorView.create("second", [128], "int32", scope=scope)
    output = TensorView.create("output", [2], "int32")
    body = [
        first.allocate(),
        second.allocate(),
        first.store(0, 17),
        output.store(0, first.load(0)),
        second.store(0, 29),
        output.store(1, second.load(0)),
    ]
    module = lower(body, [output.tensor], static=True, reuse=reuse)
    arena = allocations(module)
    assert len(arena) == 1
    assert int(arena[0].var.ty.shape[0]) == (512 if reuse else 1024)
    assert arena[0].var.ty.data_alignment == first.tensor.ty.data_alignment


def test_alias_uses_extend_backing_storage_lifetime():
    first = TensorView.create("first", [128], "int32", scope="shared.dyn")
    second = TensorView.create("second", [128], "int32", scope="shared.dyn")
    alias = TensorView.create("alias", [128], "int32", scope="shared.dyn")
    output = TensorView.create("output", [1], "int32")
    body = [
        first.allocate(),
        second.allocate(),
        alias.declare(first.data),
        first.store(0, 17),
        second.store(0, 29),
        output.store(0, alias.load(0) + second.load(0)),
    ]
    module = lower(body, [output.tensor])
    assert int(allocations(module)[0].var.ty.shape[0]) == 1024


def test_offsets_respect_physical_span_origin_and_explicit_alignment():
    first = TensorView.create("first", [128], "int32", scope="shared.dyn", strides=[2], elem_offset=3)
    second = TensorView.create("second", [8], "int32", scope="shared.dyn")
    output = TensorView.create("output", [1], "int32")
    body = [
        first.allocate(),
        second.allocate(),
        first.store(127, 17),
        second.store(7, 29),
        output.store(0, first.load(127) + second.load(7)),
    ]
    module = lower(body, [output.tensor], alignments={"second": ir.const(256)})
    arena = allocations(module)[0].var.ty
    assert int(arena.shape[0]) == 1312
    assert arena.data_alignment == 256


def test_static_storage_is_merged_only_when_enabled():
    first = TensorView.create("first", [16], "int32", scope="shared")
    second = TensorView.create("second", [16], "int32", scope="shared")
    body = [first.allocate(), second.allocate(), first.store(0, 1), second.store(0, 2)]
    assert len(allocations(lower(body))) == 2
    assert len(allocations(lower(body, static=True))) == 1


def test_async_or_opaque_pointer_use_disables_lifetime_reuse():
    first = TensorView.create("first", [128], "int32", scope="shared.dyn")
    second = TensorView.create("second", [128], "int32", scope="shared.dyn")
    body = [
        first.allocate(),
        second.allocate(),
        ir.Evaluate(tirx.call_extern("asynchronous_use", first.data, ty="void")),
        second.store(0, 29),
    ]
    module = lower(body, aggressive=True)
    assert int(allocations(module)[0].var.ty.shape[0]) == 1024


def test_async_use_can_take_over_completed_synchronous_storage():
    first = TensorView.create("first", [128], "int32", scope="shared.dyn")
    second = TensorView.create("second", [128], "int32", scope="shared.dyn")
    third = TensorView.create("third", [128], "int32", scope="shared.dyn")
    output = TensorView.create("output", [1], "int32")
    body = [
        first.allocate(),
        second.allocate(),
        third.allocate(),
        first.store(0, 17),
        output.store(0, first.load(0)),
        ir.Evaluate(tirx.call_extern("asynchronous_use", second.data, ty="void")),
        third.store(0, 29),
    ]
    module = lower(body, [output.tensor], aggressive=True)
    assert int(allocations(module)[0].var.ty.shape[0]) == 1024


@pytest.mark.parametrize("remaining_groups", [0, 1])
@pytest.mark.parametrize("guarded", [False, True])
def test_completed_tma_store_can_release_its_shared_slot(remaining_groups, guarded):
    first = TensorView.create("first", [128], "int32", scope="shared.dyn")
    second = TensorView.create("second", [128], "int32", scope="shared.dyn")
    output = TensorView.create("output", [128], "int32")
    condition = ir.Var("condition", "bool")
    operations = [
        ir.Evaluate(ir.Call("tl.tma_store", [output.data, first.data, ir.const(512)], ty="void")),
        ir.Evaluate(ir.Call("tl.tma_store_arrive", [], ty="void")),
        ir.Evaluate(ir.Call("tl.tma_store_wait", [ir.const(remaining_groups), ir.const(True)], ty="void")),
    ]
    if guarded:
        operations = [ir.If(condition, operation, None) for operation in operations]
    module = lower(
        [first.allocate(), second.allocate(), first.store(0, 17), *operations, second.store(0, 29), output.store(0, second.load(0))],
        [output.tensor, condition],
    )
    assert int(allocations(module)[0].var.ty.shape[0]) == (512 if remaining_groups == 0 else 1024)


@pytest.mark.parametrize("concurrent", [False, True])
def test_tma_reuse_requires_matching_guards_and_sequential_roles(concurrent):
    first = TensorView.create("first", [128], "int32", scope="shared.dyn")
    second = TensorView.create("second", [128], "int32", scope="shared.dyn")
    output = TensorView.create("output", [128], "int32")
    condition = ir.Var("condition", "bool")
    store = ir.Evaluate(ir.Call("tl.tma_store", [output.data, first.data, ir.const(512)], ty="void"))
    arrive = ir.Evaluate(ir.Call("tl.tma_store_arrive", [], ty="void"))
    wait = ir.Evaluate(ir.Call("tl.tma_store_wait", [ir.const(0), ir.const(True)], ty="void"))
    operations = (
        [store, arrive, wait]
        if concurrent
        else [ir.If(condition, store, None), ir.If(condition, arrive, None), ir.If(~condition, wait, None)]
    )
    body = [first.allocate(), second.allocate(), first.store(0, 17), *operations, second.store(0, 29)]
    if concurrent:
        body = ir.RegionStmt("tirx.scope", [], [], {"tl.ws_concurrent": True}, body)
    module = lower(body, [output.tensor, condition])
    assert int(allocations(module)[0].var.ty.shape[0]) == 1024


def test_tma_completion_does_not_replay_memory_dependent_guards():
    first = TensorView.create("first", [128], "int32", scope="shared.dyn")
    second = TensorView.create("second", [128], "int32", scope="shared.dyn")
    output = TensorView.create("output", [128], "int32")
    flag = TensorView.create("flag", [1], "int32")
    condition = flag.load(0) > 0
    operations = [
        ir.Evaluate(ir.Call("tl.tma_store", [output.data, first.data, ir.const(512)], ty="void")),
        ir.Evaluate(ir.Call("tl.tma_store_arrive", [], ty="void")),
        ir.Evaluate(ir.Call("tl.tma_store_wait", [ir.const(0), ir.const(True)], ty="void")),
    ]
    body = [
        first.allocate(),
        second.allocate(),
        first.store(0, 17),
        *[ir.If(condition, operation, None) for operation in operations],
        second.store(0, 29),
    ]
    module = lower(body, [output.tensor, flag.tensor])
    assert int(allocations(module)[0].var.ty.shape[0]) == 1024


def test_dynamic_sizes_keep_scalar_dependencies_in_scope():
    extent = ir.Var("extent", "int32")
    first = TensorView.create("first", [extent], "int32", scope="shared.dyn")
    second = TensorView.create("second", [extent], "int32", scope="shared.dyn")
    body = [first.allocate(), second.allocate(), first.store(0, 1), second.store(0, 2)]
    module = lower(body, [extent], reuse=False)
    assert len(allocations(module)) == 1


@pytest.mark.parametrize("reverse_declarations", [False, True])
@pytest.mark.parametrize("lanes", [1, 4])
def test_lifetime_order_and_vector_width_do_not_prevent_reuse(reverse_declarations, lanes):
    dtype = "float32" if lanes == 1 else f"float32x{lanes}"
    first = TensorView.create("first", [128 // lanes], dtype, scope="shared.dyn")
    second = TensorView.create("second", [128], "float32", scope="shared.dyn")
    output = TensorView.create("output", [1], dtype)
    declarations = [first.allocate(), second.allocate()]
    if reverse_declarations:
        declarations.reverse()
    value = ir.const(17.0, "float32")
    if lanes != 1:
        value = ir.prim.Broadcast(value, lanes)
    body = declarations + [
        first.store(0, value),
        output.store(0, first.load(0)),
        ir.Evaluate(tirx.call_extern("asynchronous_use", second.data, ty="void")),
    ]
    module = lower(body, [output.tensor])
    assert int(allocations(module)[0].var.ty.shape[0]) == 512


def test_arena_cannot_cross_a_late_size_binding():
    extent = ir.Var("extent", "int32")
    first = TensorView.create("first", [128], "int32", scope="shared.dyn")
    second = TensorView.create("second", [extent], "int32", scope="shared.dyn")
    body = [first.allocate(), first.store(0, 1), ir.Bind(extent, ir.const(16)), second.allocate(), second.store(0, 2)]
    with pytest.raises(ValueError, match="definition of extent"):
        lower(body)


@pytest.mark.parametrize("reuse", [False, True])
def test_merged_aliases_and_thread_sync_execute_on_cuda(reuse):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    first = TensorView.create("first", [128], "int32", scope="shared.dyn")
    second = TensorView.create("second", [128], "int32", scope="shared.dyn")
    output = TensorView.create("output", [2, 128], "int32")
    thread = ir.Var("thread", "int32")
    body = [
        first.allocate(),
        second.allocate(),
        first.store(thread, thread),
        output.store([0, thread], first.load(thread ^ 1)),
        second.store(thread, 128 + thread),
        output.store([1, thread], second.load(thread ^ 1)),
    ]
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(128)], [thread], None, body)
    module = lower(launch, [output.tensor], reuse=reuse)
    assert int(allocations(module)[0].var.ty.shape[0]) == (512 if reuse else 1024)
    module = tvm_ffi.get_global_func("tl.transform.ThreadSync")("shared.dyn")(module)
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    executable = tirx.compile(module, target=target)
    actual = runtime.empty((2, 128), "int32", device=device)
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(2)[:, None] * 128 + (np.arange(128) ^ 1))
