"""Native storage, participation, and loop-carried synchronization contracts."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
from tilelang.language.tir.entry import prim_func
from tilelang.testing.ir import collect_calls, collect_nodes


def _tensor(name, shape, dtype, scope="shared", layout="default"):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout=layout))


def _integer(value, dtype="int32"):
    return ir.const(value, dtype)


def _alloc(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _view(tensor, backing, byte_offset=0, pointer_op="tirx.ptr_byte_offset"):
    pointer = tirx.tensor_data_ptr(backing)
    offset = ir.Call(pointer_op, [pointer, _integer(byte_offset, "int64")], ty=pointer.ty)
    pointer = tirx.reinterpret(offset, ty=ir.PtrType(tensor.ty.dtype, tensor.ty.scope))
    return ir.Bind(tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty]))


def _thread(var, count, body, tag="threadIdx.x"):
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(tag), _integer(count)], [var], {}, ir.SeqStmt(body), [])


def _syncs(root, scope="shared"):
    return [
        call
        for name in ("tirx.gpu_storage_sync", "tl.storage_sync")
        for call in collect_calls(root, op=name)
        if call.args[0].value == scope
    ]


def _lower(function, scope="shared"):
    function = function.with_attr("target", tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90"}))
    return tilelang.transform.ThreadSync(scope)(tvm_tirx.IRModule({"main": function}))["main"]


def _sequence_contains(root, first, last):
    for seq in collect_nodes(root, ir.SeqStmt):
        starts = [i for i, stmt in enumerate(seq.seq) if stmt.same_as(first)]
        ends = [i for i, stmt in enumerate(seq.seq) if stmt.same_as(last)]
        if starts and ends and starts[0] < ends[0]:
            return True
    return False


def test_write_only_loop_carry_orders_real_waw_schedule():
    # This is an allowed inter-warp schedule even if a barrier precedes the
    # loop: warp 0 completes both iterations before warp 1 starts its first.
    # A barrier on the carried edge instead completes iteration 0 in both
    # warps before either starts iteration 1.
    def simulate(schedule):
        memory = [-1] * 64
        for warp, iteration in schedule:
            for lane in range(32):
                thread = warp * 32 + lane
                memory[(thread + iteration * 32) % 64] = iteration
        return memory

    before_loop_only = simulate([(0, 0), (0, 1), (1, 0), (1, 1)])
    between_iterations = simulate([(0, 0), (1, 0), (0, 1), (1, 1)])
    assert before_loop_only == [1] * 32 + [0] * 32
    assert between_iterations == [1] * 64

    @prim_func(private=True)
    def kernel():
        shared = T.alloc_shared((64,), "int32", scope="shared")
        tx = T.launch_thread("threadIdx.x", 64)
        for i in range(2):
            shared[(tx + i * 32) % 64] = T.int32(i)

    lowered = _lower(kernel)
    loops = collect_nodes(lowered, ir.For)
    assert len(loops) == 1
    barriers = _syncs(lowered)
    assert len(barriers) == 1 and len(_syncs(loops[0])) == 1
    evaluation = next(stmt for stmt in collect_nodes(loops[0], ir.Evaluate) if stmt.value.same_as(barriers[0]))
    write = collect_nodes(loops[0], ir.TensorStore)[0]
    assert _sequence_contains(loops[0].body, evaluation, write)


@pytest.mark.parametrize("pointer_op", ["tirx.ptr_byte_offset", "tirx.handle_add_byte_offset"])
@pytest.mark.parametrize("second_index, expected", [(lambda tx: tx, 0), (lambda tx: tx + 1, 1)])
def test_pointer_offset_views_use_one_backing_storage(pointer_op, second_index, expected):
    backing = _tensor("backing", [512], "uint8")
    first = _tensor("first", [128], "float32")
    second = _tensor("second", [127], "float32")
    tx = ir.Var("tx", "int32")
    function = tirx.Function(
        [],
        _thread(
            tx,
            64,
            [
                _alloc(backing),
                _view(first, backing, pointer_op=pointer_op),
                _view(second, backing, byte_offset=4, pointer_op=pointer_op),
                ir.TensorStore(first, [tx + 1], ir.const(1, "float32")),
                ir.TensorStore(second, [second_index(tx)], ir.const(2, "float32")),
            ],
        ),
        ir.PrimType("void"),
    )
    assert len(_syncs(_lower(function))) == expected


def test_cross_dtype_views_order_conflicts_by_bits():
    backing = _tensor("backing", [512], "uint8")
    floats = _tensor("floats", [128], "float32")
    tx = ir.Var("tx", "int32")
    function = tirx.Function(
        [],
        _thread(
            tx,
            64,
            [
                _alloc(backing),
                _view(floats, backing),
                ir.TensorStore(floats, [tx], ir.const(1, "float32")),
                ir.TensorStore(backing, [tx], _integer(7, "uint8")),
            ],
        ),
        ir.PrimType("void"),
    )
    assert len(_syncs(_lower(function))) == 1


def test_joint_thread_domain_cannot_masquerade_as_rectangle():
    @prim_func(private=True)
    def kernel():
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 2)
        if tx // 32 == ty:
            T.tvm_storage_sync("shared")

    # Every axis projection is its whole range, but only half of the 2-D
    # Cartesian product participates. A projection-only proof would accept it.
    with pytest.raises(RuntimeError, match="joint thread rectangle"):
        _lower(kernel)


@pytest.mark.parametrize("existing_fence", [False, True])
def test_wait_group_needs_barrier_after_completion(existing_fence):
    @prim_func(private=True)
    def kernel():
        _tx = T.launch_thread("threadIdx.x", 64)
        T.tvm_storage_sync("shared")
        T.evaluate(T.call_intrin("void", "tl.ptx_wait_group", 0))
        if existing_fence:
            T.tvm_storage_sync("shared")

    lowered = _lower(kernel)
    calls = collect_nodes(lowered, ir.Call)
    effects = [call.op.name for call in calls if call.op.name in ("tirx.gpu_storage_sync", "tl.ptx_wait_group")]
    assert effects == ["tirx.gpu_storage_sync", "tl.ptx_wait_group", "tirx.gpu_storage_sync"]


def test_named_barrier_keeps_native_void_type_and_operands():
    call = T.tvm_storage_sync("shared", 3, 32)
    assert call.op.name == "tl.storage_sync"
    assert tvm_ffi.structural_equal(call.ty, ir.PrimType("void"))
    assert tuple(int(value) for value in call.args[1:]) == (3, 32)
    with pytest.raises(ValueError, match="ID must be"):
        T.tvm_storage_sync("shared", 16, 32)
