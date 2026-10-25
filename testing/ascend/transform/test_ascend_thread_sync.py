import pytest
import tvm_tirx as tvm
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

import tilelang.testing
import tilelang.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


def _thread_body(make_body, thread_count=32):
    axes = [ir.Var(f"thread_{dimension}", "int32") for dimension in "xyz"]
    body = make_body(axes[0])
    for dimension, variable, extent in reversed(list(zip("xyz", axes, (thread_count, 1, 1)))):
        body = ir.RegionStmt("tirx.launch_thread", [f"threadIdx.{dimension}", extent], [variable], {}, body)
    return body


def _vf(body, name="SIMT_VF"):
    return ir.RegionStmt("tl.scope", [], [], {"name_hint": name}, body)


def _module(body, buffers=()):
    function = tirx.Function([buffer.tensor for buffer in buffers], body, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "main").with_attr("target", tvm.target.Target("ascend"))
    assert tirx.analysis.check_well_formed(function)
    return ir.IRModule({"main": function})


def _nodes(body, kind):
    result = []
    stmt_functor.post_order_visit(body, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _syncs(body):
    return [call for call in _nodes(body, ir.Call) if call.op.name == "tirx.gpu_storage_sync"]


def _tensor(shape, dtype, name, scope="global"):
    return TensorView.create(name, shape, dtype, scope=scope)


def _store(tensor, value, indices):
    return tensor.store(indices, value)


def _load(tensor, indices):
    return tensor.load(indices)


def _buffers(storage_scope, thread_count=32):
    return _tensor((thread_count,), "float32", "shared", storage_scope), _tensor((thread_count,), "float32", "output")


def _read_after_write(shared, output, thread_count=32, cross_thread=True):
    def make_body(thread_id):
        read_index = (thread_id + 1) % thread_count if cross_thread else thread_id
        return ir.SeqStmt([shared.store(thread_id, 1.0), output.store(thread_id, shared.load(read_index))])

    return _thread_body(make_body, thread_count)


def _canonicalize(module):
    def rewrite(node):
        if not isinstance(node, ir.For):
            return None
        begin, end, step = int(node.begin), int(node.end), int(node.step)
        ordinal = ir.Var(node.var.name + "_ordinal", node.var.ty)
        body = stmt_functor.substitute(node.body, {node.var: ordinal * step + begin})
        return ir.For(ordinal, 0, len(range(begin, end, step)), 1, node.kind, body)

    return ir.IRModule(
        {name: function.with_body(stmt_functor.ir_transform(function.body, None, rewrite)) for name, function in module.functions.items()}
    )


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("thread_count", [32, 64])
def test_ascend_thread_sync_inserts_barrier_inside_vf(storage_scope, thread_count):
    shared, output = _buffers(storage_scope, thread_count)
    before = _module(_vf(_read_after_write(shared, output, thread_count)), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)["main"]

    block = after.body.seq[0]
    assert isinstance(block, ir.RegionStmt)
    assert block.attrs.dict["name_hint"] == "SIMT_VF"
    syncs = _syncs(block.body)
    assert len(syncs) == 1
    assert len(syncs[0].args) == 1
    assert syncs[0].args[0].value == storage_scope
    tvm.ir.assert_structural_equal(after.params, before["main"].params)


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_keeps_thread_private_accesses_unsynchronized(storage_scope):
    shared, output = _buffers(storage_scope)
    before = _module(_vf(_read_after_write(shared, output, cross_thread=False)), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)

    tvm.ir.assert_structural_equal(after, before)


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
    shared = _tensor((4, 32), "float32", name="shared", scope="shared")
    output = _tensor((32,), "float32", name="output")
    index = ir.Var("i", "int32")

    def make_body(thread_id):
        body = ir.SeqStmt(
            [
                _store(shared, prim.Cast("float32", index), [index % 4, thread_id]),
                _store(output, _load(shared, [(index + 2) % 4, (thread_id + 1) % 32]), [thread_id]),
            ]
        )
        return ir.For(index, start, stop, step or 1, ir.ForKind.DEFAULT, body)

    before = _module(_vf(_thread_body(make_body)), [shared, output])
    if canonicalize:
        before = _canonicalize(before)

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]

    loop = _nodes(after.body, ir.For)[0]
    assert isinstance(loop, ir.For)
    assert len(_syncs(loop.body)) == expected_syncs
    if expected_syncs:
        assert isinstance(loop.body, ir.SeqStmt)
        assert len(_syncs(loop.body.seq[-1])) == 1


@pytest.mark.parametrize("canonicalize", [False, True])
def test_ascend_thread_sync_loop_carry_non_unit_step_last_pair(canonicalize):
    shared = _tensor((24, 32), "float32", name="shared", scope="shared")
    output = _tensor((32,), "float32", name="output")
    index = ir.Var("i", "int32")

    def make_body(thread_id):
        # Only i=9 -> i=12 aliases across neighboring threads.
        body = ir.SeqStmt(
            [
                _store(shared, prim.Cast("float32", index), [index, thread_id]),
                _store(output, _load(shared, [21 - index, (thread_id + 1) % 32]), [thread_id]),
            ]
        )
        return ir.For(index, 3, 13, 3, ir.ForKind.DEFAULT, body)

    before = _module(_vf(_thread_body(make_body)), [shared, output])
    if canonicalize:
        before = _canonicalize(before)

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]

    loop = _nodes(after.body, ir.For)[0]
    assert isinstance(loop, ir.For)
    assert len(_syncs(loop.body)) == 1
    assert isinstance(loop.body, ir.SeqStmt)
    assert len(_syncs(loop.body.seq[-1])) == 1


@pytest.mark.parametrize("storage_scope", ["shared", "shared.dyn"])
def test_ascend_thread_sync_does_not_synchronize_between_vfs(storage_scope):
    shared, output = _buffers(storage_scope)
    writer = _vf(_thread_body(lambda thread_id: _store(shared, ir.const(1, "float32"), [thread_id])))
    reader = _vf(_thread_body(lambda thread_id: _store(output, _load(shared, [(thread_id + 1) % 32]), [thread_id])))
    before = _module(ir.SeqStmt([writer, reader]), [shared, output])

    after = ascend_transform.AscendThreadSync(storage_scope)(before)

    tvm.ir.assert_structural_equal(after, before)


def test_ascend_thread_sync_plans_each_vf_independently():
    shared, output = _buffers("shared", 64)
    first_vf = _vf(_read_after_write(shared, output, thread_count=32))
    second_vf = _vf(_read_after_write(shared, output, thread_count=64))
    before = _module(ir.SeqStmt([first_vf, second_vf]), [shared, output])

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]

    assert isinstance(after.body, ir.SeqStmt)
    assert len(after.body.seq) == 2
    for block in after.body.seq:
        assert isinstance(block, ir.RegionStmt)
        assert block.attrs.dict["name_hint"] == "SIMT_VF"
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
        write_ptr = T.access_ptr(shared.tensor, "w", offset=thread_id, extent=1)
        read_ptr = T.access_ptr(shared.tensor, "r", offset=(thread_id + 1) % 32, extent=1)
        return ir.SeqStmt(
            [
                ir.Evaluate(T.call_extern("int32", "write_shared", write_ptr)),
                ir.Evaluate(T.call_extern("int32", "read_shared", read_ptr)),
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
    wait_group = ir.Evaluate(ir.Call("tirx.ptx.cp_async_wait_group", [0]))
    before = _module(_vf(_thread_body(lambda _thread_id: wait_group)))

    after = ascend_transform.AscendThreadSync("shared")(before)

    tvm.ir.assert_structural_equal(after, before)


def test_ascend_thread_sync_skips_partial_barrier_rewriting():
    sync = ir.Evaluate(ir.Call("tirx.gpu_storage_sync", ["shared"]))
    body = _thread_body(lambda thread_id: ir.If(thread_id < 32, sync, None), thread_count=64)
    before = _module(_vf(body))

    after = ascend_transform.AscendThreadSync("shared")(before)

    tvm.ir.assert_structural_equal(after, before)


def test_ascend_thread_sync_cuda_fallback_distinguishes_rectangles():
    sync = ir.Evaluate(ir.Call("tirx.gpu_storage_sync", ["shared"]))

    def make_body(tx):
        return ir.SeqStmt(
            [
                ir.If(tx // 64 == 0, sync, None),
                ir.If(tx // 64 == 1, sync, None),
                ir.If(tx < 64, sync, None),
            ]
        )

    func = tirx.Function([], _thread_body(make_body, 256)).with_attr("target", tvm.target.Target("cuda"))
    before = ir.IRModule({"main": func})

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]

    syncs = _syncs(after.body)
    assert len(syncs) == 3
    assert [sync.args[2].value for sync in syncs] == [64, 64, 64]
    assert syncs[0].args[1].value != syncs[1].args[1].value
    assert syncs[0].args[1].value == syncs[2].args[1].value


@pytest.mark.parametrize("axis_order", ["xy", "yx"])
def test_ascend_thread_sync_cuda_fallback_multidimensional_rectangle(axis_order):
    tx = ir.Var("tx", "int32")
    ty = ir.Var("ty", "int32")
    sync = ir.Evaluate(ir.Call("tirx.gpu_storage_sync", ["shared"]))
    body = ir.If(tx // 32 == 0, sync, None)
    for axis in axis_order:
        variable, extent = (tx, 64) if axis == "x" else (ty, 2)
        body = ir.RegionStmt("tirx.launch_thread", [f"threadIdx.{axis}", extent], [variable], {}, body)
    func = tirx.Function([], body).with_attr("target", tvm.target.Target("cuda"))
    before = ir.IRModule({"main": func})

    after = ascend_transform.AscendThreadSync("shared")(before)["main"]
    syncs = _syncs(after.body)
    assert len(syncs) == 1
    assert len(syncs[0].args) == 3
    assert syncs[0].args[2].value == 64


if __name__ == "__main__":
    tilelang.testing.main()
