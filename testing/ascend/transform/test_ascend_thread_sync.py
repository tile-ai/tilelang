import pytest

import tilelang.testing
from tilelang import tvm
from tilelang.ascend import transform as ascend_transform
from tvm import tirx


def _thread_body(make_body, thread_count=32):
    extents = (thread_count, 1, 1)
    thread_axes = [
        tirx.IterVar(
            tvm.ir.Range(0, extent),
            tirx.Var(f"thread_{dimension}", "int32"),
            tirx.IterVar.ThreadIndex,
            f"threadIdx.{dimension}",
        )
        for dimension, extent in zip("xyz", extents)
    ]
    body = make_body(thread_axes[0].var)
    for thread_axis, extent in reversed(list(zip(thread_axes, extents))):
        body = tirx.AttrStmt(thread_axis, "thread_extent", extent, body)
    return body


def _vf(body, name="SIMT_VF"):
    return tirx.SBlock([], [], [], name, body)


def _module(body, buffers=()):
    func = tirx.PrimFunc(
        [buffer.data for buffer in buffers],
        body,
        buffer_map={buffer.data: buffer for buffer in buffers},
    )
    func = func.with_attr("global_symbol", "main").with_attr("target", tvm.target.Target("ascend"))
    return tvm.IRModule.from_expr(func)


def _syncs(body):
    syncs = []
    sync_op = tvm.ir.Op.get("tirx.tvm_storage_sync")

    def collect(node):
        if isinstance(node, tirx.Call) and node.op.same_as(sync_op):
            syncs.append(node)

    tirx.stmt_functor.post_order_visit(body, collect)
    return syncs


def _buffers(storage_scope, thread_count=32):
    shared = tirx.decl_buffer((thread_count,), "float32", name="shared", scope=storage_scope)
    output = tirx.decl_buffer((thread_count,), "float32", name="output")
    return shared, output


def _read_after_write(shared, output, thread_count=32, cross_thread=True):
    def make_body(thread_id):
        read_index = (thread_id + 1) % thread_count if cross_thread else thread_id
        return tirx.SeqStmt(
            [
                tirx.BufferStore(shared, tirx.const(1, "float32"), [thread_id]),
                tirx.BufferStore(output, tirx.BufferLoad(shared, [read_index]), [thread_id]),
            ]
        )

    return _thread_body(make_body, thread_count)


def test_ascend_thread_sync_has_separate_registration():
    assert tilelang.transform.ThreadSync("shared").info.name == "tl.ThreadSync"
    assert ascend_transform.AscendThreadSync("shared").info.name == "tl.AscendThreadSync"
    assert "AscendThreadSync" in ascend_transform.__all__


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("thread_count", [32, 64])
def test_ascend_thread_sync_inserts_barrier_inside_vf(storage_scope, thread_count):
    shared, output = _buffers(storage_scope, thread_count)
    before = _module(_vf(_read_after_write(shared, output, thread_count)), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)["main"]

    assert isinstance(after.body, tirx.SBlock)
    assert after.body.name_hint == "SIMT_VF"
    syncs = _syncs(after.body.body)
    assert len(syncs) == 1
    assert len(syncs[0].args) == 1
    assert syncs[0].args[0].value == storage_scope
    tvm.ir.assert_structural_equal(after.buffer_map, before["main"].buffer_map)


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_keeps_thread_private_accesses_unsynchronized(storage_scope):
    shared, output = _buffers(storage_scope)
    before = _module(_vf(_read_after_write(shared, output, cross_thread=False)), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)

    tvm.ir.assert_structural_equal(after, before)


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_does_not_synchronize_between_vfs(storage_scope):
    shared, output = _buffers(storage_scope)
    writer = _vf(_thread_body(lambda thread_id: tirx.BufferStore(shared, tirx.const(1, "float32"), [thread_id])))
    reader = _vf(_thread_body(lambda thread_id: tirx.BufferStore(output, tirx.BufferLoad(shared, [(thread_id + 1) % 32]), [thread_id])))
    before = _module(tirx.SeqStmt([writer, reader]), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)

    tvm.ir.assert_structural_equal(after, before)


def test_ascend_thread_sync_plans_each_vf_independently():
    shared, output = _buffers("shared", 64)
    first_vf = _vf(_read_after_write(shared, output, thread_count=32))
    second_vf = _vf(_read_after_write(shared, output, thread_count=64))
    before = _module(tirx.SeqStmt([first_vf, second_vf]), [shared, output])

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]

    assert isinstance(after.body, tirx.SeqStmt)
    assert len(after.body.seq) == 2
    for block in after.body.seq:
        assert isinstance(block, tirx.SBlock)
        assert block.name_hint == "SIMT_VF"
        assert len(_syncs(block.body)) == 1


@pytest.mark.parametrize("block_name", [None, "ordinary", "SIMD_VF"])
def test_ascend_thread_sync_leaves_non_simt_code_unchanged(block_name):
    shared, output = _buffers("shared")
    body = _read_after_write(shared, output)
    if block_name is not None:
        body = _vf(body, block_name)
    before = _module(body, [shared, output])

    after = ascend_transform.AscendThreadSync("shared")(before)

    tvm.ir.assert_structural_equal(after, before)


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_handles_captured_access_ptrs(storage_scope):
    shared, _output = _buffers(storage_scope)

    def make_body(thread_id):
        write_ptr = shared.access_ptr("w", offset=thread_id, extent=1)
        read_ptr = shared.access_ptr("r", offset=(thread_id + 1) % 32, extent=1)
        return tirx.SeqStmt(
            [
                tirx.Evaluate(tirx.call_extern("int32", "write_shared", write_ptr)),
                tirx.Evaluate(tirx.call_extern("int32", "read_shared", read_ptr)),
            ]
        )

    before = _module(_vf(_thread_body(make_body)), [shared])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)["main"]

    assert len(_syncs(after.body)) == 1


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_respects_disable_config(storage_scope):
    shared, output = _buffers(storage_scope)
    before = _module(_vf(_read_after_write(shared, output)), [shared, output])

    with tvm.transform.PassContext(config={"tl.disable_thread_storage_sync": True}):
        after = ascend_transform.AscendThreadSync(storage_scope)(before)

    tvm.ir.assert_structural_equal(after, before)


@pytest.mark.parametrize("storage_scope", ["global", "shared.dyn"])
def test_ascend_thread_sync_leaves_other_scopes_unchanged(storage_scope):
    shared, output = _buffers("shared")
    before = _module(_vf(_read_after_write(shared, output)), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)

    tvm.ir.assert_structural_equal(after, before)


def test_ascend_thread_sync_skips_wait_group_insertion():
    wait_group = tirx.Evaluate(tirx.call_intrin("handle", "tirx.ptx_wait_group", 0))
    before = _module(_vf(_thread_body(lambda _thread_id: wait_group)))

    after = ascend_transform.AscendThreadSync("shared")(before)

    tvm.ir.assert_structural_equal(after, before)


def test_ascend_thread_sync_skips_partial_barrier_rewriting():
    sync = tirx.Evaluate(tirx.call_intrin("int32", "tirx.tvm_storage_sync", "shared"))
    body = _thread_body(lambda thread_id: tirx.IfThenElse(thread_id < 32, sync, None), thread_count=64)
    before = _module(_vf(body))

    after = ascend_transform.AscendThreadSync("shared")(before)

    tvm.ir.assert_structural_equal(after, before)


if __name__ == "__main__":
    tilelang.testing.main()
