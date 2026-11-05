import pytest

import tilelang.testing
import tvm_tirx
from tilelang.ascend import transform as ascend_transform
from tvm_tirx import ir, tirx
from testing.ascend._ir import is_sblock, launch_thread, nodes, sblock, sblock_metadata, single_statement, tensor


def _thread_body(make_body, thread_count=32):
    extents = (thread_count, 1, 1)
    thread_axes = [ir.Var(f"thread_{dimension}", "int32") for dimension in "xyz"]
    body = make_body(thread_axes[0])
    for dimension, thread_axis, extent in reversed(list(zip("xyz", thread_axes, extents))):
        body = launch_thread(thread_axis, f"threadIdx.{dimension}", extent, body)
    return body


def _vf(body, name="SIMT_VF"):
    return sblock(name, body)


def _module(body, buffers=()):
    func = tirx.Function(list(buffers), body)
    func = func.with_attr("global_symbol", "main").with_attr("target", tvm_tirx.target.Target("ascend"))
    assert tirx.analysis.check_well_formed(func)
    return tvm_tirx.IRModule.from_expr(func)


def _syncs(body):
    # Whole-domain barriers use the native operation; partial CUDA barriers
    # retain their ID and participating-thread count in tl.storage_sync.
    return [
        node for node in nodes(body, ir.Call) if isinstance(node.op, ir.Op) and node.op.name in {"tirx.gpu_storage_sync", "tl.storage_sync"}
    ]


def _loop_body(index, start, stop, step, make_body, canonicalize):
    step = 1 if step is None else step
    if canonicalize and (start != 0 or step != 1):
        # The retired s_tir.CanonicalizeLoop used ceildiv(stop-start, step)
        # iterations and substituted index*step+start in the complete body.
        body = make_body(index * step + start)
        start, stop, step = 0, (stop - start + step - 1) // step, 1
    else:
        body = make_body(index)
    return ir.For(index, ir.const(start), ir.const(stop), ir.const(step), ir.ForKind.DEFAULT, body)


def _loop_in_vf(body):
    block = single_statement(body)
    assert is_sblock(block)
    assert sblock_metadata(block)["name"] == "SIMT_VF"
    body = block.body
    for dimension in "xyz":
        launch = single_statement(body)
        assert isinstance(launch, ir.RegionStmt) and launch.op.name == "tirx.launch_thread"
        assert launch.args[0].value == f"threadIdx.{dimension}"
        body = launch.body
    loop = single_statement(body)
    assert isinstance(loop, ir.For)
    return loop


def _buffers(storage_scope, thread_count=32):
    shared = tensor((thread_count,), "float32", name="shared", scope=storage_scope)
    output = tensor((thread_count,), "float32", name="output")
    return shared, output


def _read_after_write(shared, output, thread_count=32, cross_thread=True):
    def make_body(thread_id):
        read_index = (thread_id + 1) % thread_count if cross_thread else thread_id
        return ir.SeqStmt(
            [
                ir.TensorStore(shared, [thread_id], ir.const(1, "float32")),
                ir.TensorStore(output, [thread_id], ir.TensorLoad(shared, [read_index])),
            ]
        )

    return _thread_body(make_body, thread_count)


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("thread_count", [32, 64])
def test_ascend_thread_sync_inserts_barrier_inside_vf(storage_scope, thread_count):
    shared, output = _buffers(storage_scope, thread_count)
    before = _module(_vf(_read_after_write(shared, output, thread_count)), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)["main"]

    block = single_statement(after.body)
    assert is_sblock(block)
    assert sblock_metadata(block)["name"] == "SIMT_VF"
    syncs = _syncs(block.body)
    assert len(syncs) == 1
    assert len(syncs[0].args) == 1
    assert syncs[0].args[0].value == storage_scope
    tvm_tirx.ir.assert_structural_equal(after.params, before["main"].params)
    assert all(actual.same_as(original) for actual, original in zip(after.params, before["main"].params))


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_keeps_thread_private_accesses_unsynchronized(storage_scope):
    shared, output = _buffers(storage_scope)
    before = _module(_vf(_read_after_write(shared, output, cross_thread=False)), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)

    tvm_tirx.ir.assert_structural_equal(after, before)


@pytest.mark.parametrize(
    "start, stop, step, expected_syncs",
    [
        pytest.param(0, 2, None, 0, id="default_step"),
        pytest.param(0, 2, 1, 0, id="unit_step"),
        pytest.param(0, 6, 2, 1, id="non_unit_step"),
        pytest.param(3, 9, 2, 1, id="nonzero_start"),
        pytest.param(3, 8, 2, 1, id="nondivisible_extent"),
        pytest.param(3, 5, 2, 0, id="single_iteration"),
        pytest.param(3, 4, 2, 0, id="extent_smaller_than_step"),
    ],
)
@pytest.mark.parametrize("canonicalize", [False, True])
def test_ascend_thread_sync_loop_carry_respects_loop_step(start, stop, step, expected_syncs, canonicalize):
    shared = tensor((4, 32), "float32", name="shared", scope="shared")
    output = tensor((32,), "float32", name="output")
    index = ir.Var("i", "int32")

    def make_body(thread_id):
        return _loop_body(
            index,
            start,
            stop,
            step,
            lambda value: ir.SeqStmt(
                [
                    ir.TensorStore(shared, [value % 4, thread_id], ir.prim.Cast("float32", value)),
                    ir.TensorStore(output, [thread_id], ir.TensorLoad(shared, [(value + 2) % 4, (thread_id + 1) % 32])),
                ]
            ),
            canonicalize,
        )

    before = _module(_vf(_thread_body(make_body)), [shared, output])

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]

    loop = _loop_in_vf(after.body)
    assert isinstance(loop, ir.For)
    assert len(_syncs(loop.body)) == expected_syncs
    if expected_syncs:
        assert isinstance(loop.body, ir.SeqStmt)
        assert len(_syncs(loop.body.seq[0])) == 1


@pytest.mark.parametrize("canonicalize", [False, True])
def test_ascend_thread_sync_loop_carry_non_unit_step_last_pair(canonicalize):
    shared = tensor((24, 32), "float32", name="shared", scope="shared")
    output = tensor((32,), "float32", name="output")
    index = ir.Var("i", "int32")

    def make_body(thread_id):
        # Only i=9 -> i=12 aliases across neighboring threads.
        return _loop_body(
            index,
            3,
            13,
            3,
            lambda value: ir.SeqStmt(
                [
                    ir.TensorStore(shared, [value, thread_id], ir.prim.Cast("float32", value)),
                    ir.TensorStore(output, [thread_id], ir.TensorLoad(shared, [21 - value, (thread_id + 1) % 32])),
                ]
            ),
            canonicalize,
        )

    before = _module(_vf(_thread_body(make_body)), [shared, output])

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]

    loop = _loop_in_vf(after.body)
    assert isinstance(loop, ir.For)
    assert len(_syncs(loop.body)) == 1
    assert isinstance(loop.body, ir.SeqStmt)
    assert len(_syncs(loop.body.seq[0])) == 1


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_does_not_synchronize_between_vfs(storage_scope):
    shared, output = _buffers(storage_scope)
    writer = _vf(_thread_body(lambda thread_id: ir.TensorStore(shared, [thread_id], ir.const(1, "float32"))))
    reader = _vf(_thread_body(lambda thread_id: ir.TensorStore(output, [thread_id], ir.TensorLoad(shared, [(thread_id + 1) % 32]))))
    before = _module(ir.SeqStmt([writer, reader]), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)

    tvm_tirx.ir.assert_structural_equal(after, before)


def test_ascend_thread_sync_plans_each_vf_independently():
    shared, output = _buffers("shared", 64)
    first_vf = _vf(_read_after_write(shared, output, thread_count=32))
    second_vf = _vf(_read_after_write(shared, output, thread_count=64))
    before = _module(ir.SeqStmt([first_vf, second_vf]), [shared, output])

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]

    assert isinstance(after.body, ir.SeqStmt)
    assert len(after.body.seq) == 2
    for block in after.body.seq:
        assert is_sblock(block)
        assert sblock_metadata(block)["name"] == "SIMT_VF"
        assert len(_syncs(block.body)) == 1


@pytest.mark.parametrize("block_name", [None, "ordinary", "SIMD_VF"])
def test_ascend_thread_sync_leaves_non_simt_code_unchanged(block_name):
    shared, output = _buffers("shared")
    body = _read_after_write(shared, output)
    if block_name is not None:
        body = _vf(body, block_name)
    before = _module(body, [shared, output])

    after = ascend_transform.AscendThreadSync("shared")(before)

    tvm_tirx.ir.assert_structural_equal(after, before)


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_handles_captured_access_ptrs(storage_scope):
    shared, _output = _buffers(storage_scope)

    def make_body(thread_id):
        write_ptr = ir.Call("tl.access_ptr", [ir.TensorLoad(shared, [thread_id]), 1, 2])
        read_ptr = ir.Call("tl.access_ptr", [ir.TensorLoad(shared, [(thread_id + 1) % 32]), 1, 1])
        return ir.SeqStmt(
            [
                ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("write_shared"), write_ptr], ty="int32")),
                ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("read_shared"), read_ptr], ty="int32")),
            ]
        )

    before = _module(_vf(_thread_body(make_body)), [shared])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)["main"]

    assert len(_syncs(after.body)) == 1


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_respects_disable_config(storage_scope):
    shared, output = _buffers(storage_scope)
    before = _module(_vf(_read_after_write(shared, output)), [shared, output])

    with tvm_tirx.transform.PassContext(config={"tl.disable_thread_storage_sync": True}):
        after = ascend_transform.AscendThreadSync(storage_scope)(before)

    tvm_tirx.ir.assert_structural_equal(after, before)


@pytest.mark.parametrize("storage_scope", ["global", "shared.dyn"])
def test_ascend_thread_sync_leaves_other_scopes_unchanged(storage_scope):
    shared, output = _buffers("shared")
    before = _module(_vf(_read_after_write(shared, output)), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)

    tvm_tirx.ir.assert_structural_equal(after, before)


def test_ascend_thread_sync_skips_wait_group_insertion():
    wait_group = ir.Evaluate(ir.Call("tl.ptx_wait_group", [0], ty="void"))
    before = _module(_vf(_thread_body(lambda _thread_id: wait_group)))

    after = ascend_transform.AscendThreadSync("shared")(before)

    tvm_tirx.ir.assert_structural_equal(after, before)


def test_ascend_thread_sync_skips_partial_barrier_rewriting():
    sync = ir.Evaluate(tirx.op.gpu_storage_sync("shared"))
    body = _thread_body(lambda thread_id: ir.If(thread_id < 32, sync, None), thread_count=64)
    before = _module(_vf(body))

    after = ascend_transform.AscendThreadSync("shared")(before)

    tvm_tirx.ir.assert_structural_equal(after, before)


def test_ascend_thread_sync_cuda_fallback_distinguishes_rectangles():
    sync = ir.Evaluate(tirx.op.gpu_storage_sync("shared"))

    def make_body(tx):
        return ir.SeqStmt(
            [
                ir.If(tx // 64 == 0, sync, None),
                ir.If(tx // 64 == 1, sync, None),
                ir.If(tx < 64, sync, None),
            ]
        )

    func = tirx.Function([], _thread_body(make_body, 256)).with_attr("target", tvm_tirx.target.Target("cuda"))
    before = tvm_tirx.IRModule.from_expr(func)

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]

    syncs = _syncs(after.body)
    assert len(syncs) == 3
    assert [sync.args[2].value for sync in syncs] == [64, 64, 64]
    assert syncs[0].args[1].value != syncs[1].args[1].value
    assert syncs[0].args[1].value == syncs[2].args[1].value


@pytest.mark.parametrize("domain_extent", [None, 32])
def test_ascend_thread_sync_cuda_fallback_multidimensional_rectangle(domain_extent):
    # Both historical IterVar.dom cases map to the same native launch domain:
    # native Var has no separate declared range that can override this binder.
    tx = ir.Var("tx", "int32")
    ty = ir.Var("ty", "int32")
    sync = ir.Evaluate(tirx.op.gpu_storage_sync("shared"))
    body = ir.If(tx // 32 == 0, sync, None)
    body = launch_thread(tx, "threadIdx.x", 64, launch_thread(ty, "threadIdx.y", 2, body))
    func = tirx.Function([], body).with_attr("target", tvm_tirx.target.Target("cuda"))
    before = tvm_tirx.IRModule.from_expr(func)

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]
    syncs = _syncs(after.body)
    assert len(syncs) == 1
    assert len(syncs[0].args) == 3
    assert syncs[0].args[2].value == 64


if __name__ == "__main__":
    tilelang.testing.main()
