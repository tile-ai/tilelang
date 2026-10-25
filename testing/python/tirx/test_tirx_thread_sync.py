import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def launch(thread, body, extent=128, dimension="x"):
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(f"threadIdx.{dimension}"), ir.const(extent)], [thread], None, body)


def lower(body, parameters=(), attributes=None):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([])).with_attr("global_symbol", "sync_kernel")
    for name, value in (attributes or {}).items():
        function = function.with_attr(name, value)
    module = tvm_ffi.get_global_func("tl.transform.ThreadSync")("shared")(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def barriers(module):
    result = []
    stmt_functor.post_order_visit(
        module,
        lambda node: result.append(node) if isinstance(node, ir.Call) and node.op.name == "tirx.gpu_storage_sync" else None,
    )
    return result


@pytest.mark.parametrize("bound_result", [False, True])
def test_handle_returning_intrinsic_keeps_pointer_memory_effects(bound_result):
    shared = TensorView.create("shared", [128], "float16", scope="shared")
    registers = TensorView.create("registers", [8], "float16", scope="local")
    thread = ir.Var("thread", "int32")
    source = ir.Call("tl.access_ptr", [shared.load(thread ^ 1), ir.const(1), ir.const(1)])
    destination = ir.Call("tl.access_ptr", [registers.load(0), ir.const(8), ir.const(2)])
    instruction = ir.Call("tl.ptx_ldmatrix", [ir.const(False), ir.const(4), source, destination])
    statement = ir.Bind(ir.Var("unused_handle", instruction.ty), instruction) if bound_result else ir.Evaluate(instruction)
    body = [shared.allocate(), registers.allocate(), shared.store(thread, prim.Cast("float16", thread)), statement]
    module = lower(launch(thread, body))
    assert barriers(module)


def test_concurrent_ws_roles_do_not_get_a_barrier_between_producer_and_consumer():
    shared = TensorView.create("shared", [32], "int32", scope="shared")
    completion = TensorView.create("completion", [1], "uint64", scope="shared")
    output = TensorView.create("output", [32], "int32")
    thread = ir.Var("thread", "int32")
    wait = ir.Evaluate(ir.Call("tl.mbarrier_wait_parity", [completion.load(0), 0], ty="void"))
    arrive = ir.Evaluate(ir.Call("tl.mbarrier_arrive", [completion.load(0)], ty="void"))
    sync = ir.Evaluate(ir.Call("tirx.gpu_storage_sync", ["shared"], ty="void"))
    consumer = ir.If(thread < 32, [wait, output.store(thread, shared.load(thread ^ 1))], None)
    producer = ir.If(thread >= 32, [shared.store(thread - 32, thread), sync, arrive], None)
    concurrent = ir.RegionStmt("tirx.scope", [], [], ir.DictAttrs({"tl.ws_concurrent": True}), [consumer, producer])
    module = lower(launch(thread, [shared.allocate(), completion.allocate(), concurrent], extent=64), [output.tensor])
    groups = []
    stmt_functor.post_order_visit(
        module, lambda node: groups.append(node) if isinstance(node, ir.RegionStmt) and node.attrs.get("tl.ws_concurrent", False) else None
    )
    assert len(groups) == 1
    assert len(groups[0].body.seq) == 2
    assert all(isinstance(role, ir.If) for role in groups[0].body.seq)
    assert barriers(module)
    assert all(len(barrier.args) == 3 and int(barrier.args[2]) == 32 for barrier in barriers(module))


def test_tma_barrier_updates_do_not_insert_cta_sync_inside_leader_election():
    from tilelang._compat.intrinsics import ptx_arrive_barrier

    shared = TensorView.create("shared", [128], "float16", scope="shared")
    barrier = TensorView.create("barrier", [1], "uint64", scope="shared")
    thread = ir.Var("thread", "int32")
    descriptor = ir.Var("descriptor", ir.PtrType("uint8"))
    destination = ir.Call("tl.access_ptr", [shared.load(0), ir.const(128), ir.const(2)])
    load = ir.Evaluate(ir.Call("tl.tma_load_gather4", [descriptor, barrier.load(0), destination, 0, 0, 1, 2, 3], ty="void"))
    arrive = ir.Evaluate(ptx_arrive_barrier(barrier.load(0)))
    elected = ir.Call("tl.tl_shuffle_elect", [128], ty="bool")
    body = [shared.allocate(), barrier.allocate(), ir.If(elected, [load, arrive], None)]
    module = lower(launch(thread, body), [descriptor])
    assert not barriers(module)


@pytest.mark.parametrize("cross_thread", [False, True])
@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_barrier_tracks_physical_thread_and_vector_ownership(cross_thread, lanes):
    shared = TensorView.create("shared", [128 * lanes], "int32", scope="shared")
    output = TensorView.create("output", [128 * lanes], "int32")
    thread = ir.Var("thread", "int32")
    other = thread ^ 1 if cross_thread else thread
    write_index = thread if lanes == 1 else prim.Ramp(thread * lanes, 1, lanes)
    read_index = other if lanes == 1 else prim.Ramp(other * lanes, 1, lanes)
    value = thread if lanes == 1 else prim.Broadcast(thread, lanes)
    module = lower(
        launch(thread, [shared.allocate(), shared.store(write_index, value), output.store(write_index, shared.load(read_index))]),
        [output.tensor],
    )
    assert len(barriers(module)) == int(cross_thread)


@pytest.mark.parametrize("byte_offset", [0, 4, 512])
def test_declared_aliases_use_byte_offsets(byte_offset):
    shared = TensorView.create("shared", [256], "int32", scope="shared")
    alias = TensorView.create("alias", [128], "int32", scope="shared")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    pointer = ir.Call("tirx.ptr_byte_offset", [shared.data, ir.const(byte_offset, "int64")], ty=shared.data.ty)
    body = [shared.allocate(), alias.declare(pointer), shared.store(thread, thread), output.store(thread, alias.load(thread))]
    module = lower(launch(thread, body), [output.tensor])
    assert len(barriers(module)) == int(byte_offset == 4)


def test_distinct_shared_allocations_do_not_alias():
    first = TensorView.create("first", [128], "int32", scope="shared")
    second = TensorView.create("second", [128], "int32", scope="shared")
    thread = ir.Var("thread", "int32")
    module = lower(launch(thread, [first.allocate(), second.allocate(), first.store(thread, thread), second.store(thread ^ 1, thread)]))
    assert not barriers(module)


@pytest.mark.parametrize("participants", [24, 40, 48])
def test_ragged_swizzled_stores_have_no_cross_thread_loop_dependency(participants):
    shared = TensorView.create("shared", [4096], "float16", scope="shared")
    thread = ir.Var("thread", "int32")
    partition = ir.Var("partition", "int32")
    index = thread // 16 * 1024 + partition * 16 + (thread % 16 // 8 + partition % 8 // 4) % 2 * 8 + thread % 8
    store = shared.store(index, prim.Cast("float16", thread))
    loop = ir.For(partition, 0, 64, 1, ir.ForKind.UNROLLED, [store])
    body = [shared.allocate(), ir.If(prim.And(thread // participants == 0, thread < 96), [loop], None)]
    module = lower(launch(thread, body))
    assert not barriers(module)


def test_named_barriers_use_distinct_rectangular_cohorts():
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    first = [shared.store(thread, thread), output.store(thread, shared.load(thread ^ 1))]
    second = [shared.store(thread, thread + 1), output.store(thread, shared.load(thread ^ 1))]
    module = lower(launch(thread, [shared.allocate(), ir.If(thread < 64, first, second)]), [output.tensor])
    calls = barriers(module)
    assert len(calls) == 2
    assert {int(call.args[1]) for call in calls} == {3, 4}
    assert {int(call.args[2]) for call in calls} == {64}


@pytest.mark.parametrize("predicate", [lambda thread: thread < 16, lambda thread: prim.Or(thread < 32, thread >= 64)])
def test_non_warp_complete_or_nonrectangular_cohort_is_rejected(predicate):
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    body = [
        shared.allocate(),
        ir.If(predicate(thread), [shared.store(thread, thread), output.store(thread, shared.load(thread ^ 1))], None),
    ]
    with pytest.raises(ValueError, match="warp-complete|split a hardware warp"):
        lower(launch(thread, body), [output.tensor])


@pytest.mark.parametrize("source_scope", ["global", "local"])
@pytest.mark.parametrize("varying", [False, True])
def test_loop_bounds_from_readonly_snapshots_require_uniform_values(source_scope, varying):
    source = TensorView.create("source", [128], "int32")
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    iteration = ir.Var("iteration", "int32")
    statements = [shared.allocate()]
    bound = source.load(thread if varying else 0)
    if source_scope == "local":
        snapshot = TensorView.create("snapshot", [1], "int32", scope="local")
        statements.extend([snapshot.allocate(), snapshot.store(0, bound)])
        bound = snapshot.load(0)
    statements.append(
        ir.For(
            iteration,
            0,
            bound,
            1,
            ir.ForKind.DEFAULT,
            [
                shared.store(thread, thread + iteration),
                output.store(thread, shared.load(thread ^ 1)),
            ],
        )
    )
    parameters = [source.tensor, output.tensor]
    attributes = {"tl.readonly_param_indices": [ir.const(0)]}
    if varying:
        with pytest.raises(ValueError, match="loop bounds"):
            lower(launch(thread, statements), parameters, attributes)
    else:
        module = lower(launch(thread, statements), parameters, attributes)
        assert barriers(module)
        assert all(len(call.args) == 1 for call in barriers(module))


def test_uniform_runtime_condition_uses_full_block_barrier():
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    enabled = ir.Var("enabled", "bool")
    body = [shared.allocate(), ir.If(enabled, [shared.store(thread, thread), output.store(thread, shared.load(thread ^ 1))], None)]
    calls = barriers(lower(launch(thread, body), [output.tensor, enabled]))
    assert len(calls) == 1 and len(calls[0].args) == 1


@pytest.mark.parametrize("source_scope", ["global", "shared", "local"])
def test_readonly_predicate_snapshots_are_uniform(source_scope):
    source = TensorView.create("source", [2], "bool")
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    predicate = TensorView.create("predicate", [1], "bool", scope="local")
    thread = ir.Var("thread", "int32")
    output = TensorView.create("output", [128], "int32")
    statements = [shared.allocate(), predicate.allocate()]
    mask = source
    if source_scope != "global":
        mask = TensorView.create("mask", [2], "bool", scope=source_scope)
        statements.append(mask.allocate())
        element = ir.Var("element", "int32")
        statements.append(ir.For(element, 0, 2, 1, ir.ForKind.DEFAULT, mask.store(element, source.load(element))))
    pointer = ir.Call("tirx.address_of", [mask.load(0)])
    statements.append(predicate.store(0, ir.Call("tl.all_of", [pointer, ir.const(2)], ty="bool")))
    statements.append(ir.If(predicate.load(0), [shared.store(thread, thread), output.store(thread, shared.load(thread ^ 1))], None))
    module = lower(launch(thread, statements), [source.tensor, output.tensor], {"tl.readonly_param_indices": [ir.const(0)]})
    assert barriers(module)
    assert all(len(call.args) == 1 for call in barriers(module))


@pytest.mark.parametrize("pattern", ["varying_load", "conditional_update", "may_alias", "uninitialized"])
def test_mutable_or_uninitialized_predicate_is_not_assumed_uniform(pattern):
    source = TensorView.create("source", [128], "bool")
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    predicate = TensorView.create("predicate", [1], "bool", scope="local")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    statements = [shared.allocate(), predicate.allocate()]
    if pattern != "uninitialized":
        statements.append(predicate.store(0, source.load(thread if pattern == "varying_load" else 0)))
    if pattern == "conditional_update":
        statements.append(ir.If(thread == 0, predicate.store(0, ir.const(False)), None))
    statements.append(ir.If(predicate.load(0), [shared.store(thread, thread), output.store(thread, shared.load(thread ^ 1))], None))
    attributes = {"tl.readonly_param_indices": [ir.const(0)]}
    if pattern == "may_alias":
        attributes["tirx.noalias"] = False
    with pytest.raises(ValueError, match="mutable or effectful predicates"):
        lower(launch(thread, statements), [source.tensor, output.tensor], attributes)


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("uninitialized_slot", [False, True])
def test_uniform_pipeline_predicate_tracks_initialized_ring_slots(depth, uninitialized_slot):
    source = TensorView.create("source", [8], "bool")
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    predicate = TensorView.create("predicate", [depth * 64], "bool", scope="local")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    time = ir.Var("pipeline_time", "int32")
    read_slot = (time - depth + 1) % depth * 64 + int(uninitialized_slot)
    loop = ir.For(
        time,
        0,
        8 + depth - 1,
        1,
        ir.ForKind.DEFAULT,
        [
            ir.If(time < 8, predicate.store(time % depth * 64, source.load(time)), None),
            ir.If(
                time >= depth - 1,
                ir.If(predicate.load(read_slot), [shared.store(thread, thread), output.store(thread, shared.load(thread ^ 1))], None),
                None,
            ),
        ],
    )
    body = launch(thread, [shared.allocate(), predicate.allocate(), loop])
    attributes = {"tl.readonly_param_indices": [ir.const(0)]}
    if uninitialized_slot:
        with pytest.raises(ValueError, match="mutable or effectful predicates"):
            lower(body, [source.tensor, output.tensor], attributes)
    else:
        calls = barriers(lower(body, [source.tensor, output.tensor], attributes))
        assert calls
        assert all(len(call.args) == 1 for call in calls)


@pytest.mark.parametrize("participants", [24, 40, 48])
def test_partial_warp_async_wait_joins_before_visibility_barrier(participants):
    thread = ir.Var("thread", "int32")
    wait = ir.Evaluate(ir.Call("tirx.ptx.cp_async_wait_group", [0, ir.StringImm("async"), ir.StringImm("wait_group"), ir.StringImm("")]))
    module = lower(launch(thread, ir.If(thread < participants, wait, None)))
    conditionals = []
    stmt_functor.post_order_visit(module, lambda node: conditionals.append(node) if isinstance(node, ir.If) else None)
    assert len(conditionals) == 1
    assert not barriers(conditionals[0])
    assert len(barriers(module)) == 1 and len(barriers(module)[0].args) == 1


def test_partial_async_wait_stays_inside_its_warp_specialized_role():
    thread = ir.Var("thread", "int32")
    wait = ir.Evaluate(ir.Call("tirx.ptx.cp_async_wait_group", [0, ir.StringImm("async"), ir.StringImm("wait_group"), ir.StringImm("")]))
    worker = ir.If(thread < 128, ir.Evaluate(0), None)
    loader = ir.If(prim.And(thread >= 128, thread < 256), ir.If(thread < 168, wait, None), None)
    concurrent = ir.RegionStmt("tirx.scope", [], [], {"tl.ws_concurrent": True}, [worker, loader])
    module = lower(launch(thread, concurrent, extent=256))
    groups = []
    stmt_functor.post_order_visit(
        module, lambda node: groups.append(node) if isinstance(node, ir.RegionStmt) and node.attrs.get("tl.ws_concurrent", False) else None
    )
    assert len(groups) == 1 and len(groups[0].body.seq) == 2
    assert all(isinstance(role, ir.If) for role in groups[0].body.seq)
    assert not barriers(groups[0].body.seq[0])
    calls = barriers(groups[0].body.seq[1])
    assert len(calls) == 1 and len(calls[0].args) == 3 and int(calls[0].args[2]) == 128


def test_partial_async_visibility_cannot_move_after_a_shared_read():
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    wait = ir.Evaluate(ir.Call("tirx.ptx.cp_async_wait_group", [0, ir.StringImm("async"), ir.StringImm("wait_group"), ir.StringImm("")]))
    body = [shared.allocate(), ir.If(thread < 24, [wait, output.store(thread, shared.load(thread ^ 1))], None)]
    with pytest.raises(ValueError, match="warp-complete"):
        lower(launch(thread, body), [output.tensor])


def test_async_wait_is_immediately_followed_by_visibility_barrier():
    thread = ir.Var("thread", "int32")
    wait = ir.Evaluate(ir.Call("tirx.ptx.cp_async_wait_group", [0, ir.StringImm("async"), ir.StringImm("wait_group"), ir.StringImm("")]))
    module = lower(launch(thread, wait))
    assert len(barriers(module)) == 1
    body = module["main"].body.seq[0].body.seq
    assert body[0].value.op.name == "tirx.ptx.cp_async_wait_group"
    assert body[1].value.op.name == "tirx.gpu_storage_sync"


def test_thread_divergent_loop_with_cross_thread_access_is_rejected():
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    iteration = ir.Var("iteration", "int32")
    body = [
        shared.allocate(),
        ir.For(
            iteration, 0, thread + 1, 1, ir.ForKind.DEFAULT, [shared.store(thread, thread), output.store(thread, shared.load(thread ^ 1))]
        ),
    ]
    with pytest.raises(ValueError, match="thread-divergent loop"):
        lower(launch(thread, body), [output.tensor])


@pytest.mark.parametrize("partial", [False, True])
def test_cross_thread_loop_execution(partial):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    output = TensorView.create("output", [7, 128], "int32")
    thread = ir.Var("thread", "int32")
    iteration = ir.Var("iteration", "int32")
    body = ir.For(
        iteration,
        0,
        7,
        1,
        ir.ForKind.DEFAULT,
        [shared.store(thread, thread + iteration * 128), output.store([iteration, thread], shared.load(thread ^ 1))],
    )
    if partial:
        body = ir.If(thread < 64, body, None)
    module = lower(launch(thread, [shared.allocate(), body]), [output.tensor])
    assert len(barriers(module)) == 2
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    executable = tirx.compile(module, target=target)
    actual = runtime.tensor(np.full((7, 128), -1, dtype="int32"), device=device)
    executable(actual)
    expected = np.arange(7, dtype="int32")[:, None] * 128 + (np.arange(128, dtype="int32") ^ 1)
    if partial:
        expected[:, 64:] = -1
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_in_place_cross_thread_exchange_snapshots_loads_before_storing():
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    module = lower(
        launch(
            thread,
            [
                shared.allocate(),
                shared.store(thread, thread),
                shared.store(thread, shared.load(thread ^ 1)),
                output.store(thread, shared.load(thread)),
            ],
        ),
        [output.tensor],
    )
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    executable = tirx.compile(module, target=target)
    actual = runtime.empty((128,), "int32", device=device)
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(128) ^ 1)


def test_explicit_block_sync_in_a_role_becomes_named_and_is_idempotent():
    thread = ir.Var("thread", "int32")
    enabled = ir.Var("enabled", "bool")
    synchronization = ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared")]))
    module = lower(launch(thread, ir.If(prim.And(thread < 64, enabled), synchronization, None)), [enabled])
    calls = barriers(module)
    assert len(calls) == 1 and int(calls[0].args[2]) == 64
    repeated = tvm_ffi.get_global_func("tl.transform.ThreadSync")("shared.dyn")(module)
    ir.assert_structural_equal(module, repeated)


def test_partial_explicit_barrier_does_not_order_the_entire_block():
    shared = TensorView.create("shared", [128], "int32", scope="shared")
    output = TensorView.create("output", [128], "int32")
    thread = ir.Var("thread", "int32")
    synchronization = ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared"), 3, 64]))
    module = lower(
        launch(thread, [shared.allocate(), shared.store(thread, thread), synchronization, output.store(thread, shared.load(thread ^ 1))]),
        [output.tensor],
    )
    assert len(barriers(module)) == 2
    assert len(barriers(module)[1].args) == 1


@pytest.mark.parametrize("extent", [8, 32, 64, 128])
def test_warp_barrier_orders_subwarp_memory_without_ordering_other_warps(extent):
    shared = TensorView.create("shared", [extent], "int32", scope="shared")
    output = TensorView.create("output", [extent], "int32")
    thread = ir.Var("thread", "int32")
    synchronization = ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("warp")]))
    body = [shared.store(thread, thread), synchronization, output.store(thread, shared.load(thread ^ 1))]
    module = lower(launch(thread, [shared.allocate(), ir.If(thread % 32 < 4, body, None)], extent), [output.tensor])
    calls = barriers(module)
    assert len(calls) == 1
    assert str(calls[0].args[0].value) == "warp"
    assert int(calls[0].args[1]) == 15


def test_warp_barrier_does_not_hide_cross_warp_dependencies():
    shared = TensorView.create("shared", [64], "int32", scope="shared")
    output = TensorView.create("output", [64], "int32")
    thread = ir.Var("thread", "int32")
    synchronization = ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("warp")]))
    body = [shared.allocate(), shared.store(thread, thread), synchronization, output.store(thread, shared.load(thread ^ 32))]
    calls = barriers(lower(launch(thread, body, 64), [output.tensor]))
    assert [str(call.args[0].value) for call in calls] == ["warp", "shared"]


def test_incomplete_warp_mask_does_not_hide_memory_dependencies():
    shared = TensorView.create("shared", [8], "int32", scope="shared")
    output = TensorView.create("output", [8], "int32")
    thread = ir.Var("thread", "int32")
    synchronization = ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("warp"), ir.const(3, "uint32")]))
    body = [shared.allocate(), shared.store(thread, thread), synchronization, output.store(thread, shared.load(thread ^ 1))]
    calls = barriers(lower(launch(thread, body, 8), [output.tensor]))
    assert [str(call.args[0].value) for call in calls] == ["warp", "shared"]


def test_boolean_aliases_use_byte_storage_not_bit_offsets():
    shared = TensorView.create("shared", [256], "uint8", scope="shared")
    alias = TensorView.create("alias", [128], "bool", scope="shared")
    output = TensorView.create("output", [128], "bool")
    thread = ir.Var("thread", "int32")
    pointer = ir.Call("tirx.ptr_byte_offset", [shared.data, ir.const(64, "int64")], ty=ir.PtrType("bool", "shared"))
    body = [
        shared.allocate(),
        alias.declare(pointer),
        shared.store(thread, prim.Cast("uint8", thread)),
        output.store(thread, alias.load(thread)),
    ]
    assert len(barriers(lower(launch(thread, body), [output.tensor]))) == 1
