"""Supported cross-core handshakes and user-reserved flag slots."""

import pytest
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import calls, copy, kernel, nodes, schedule, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


def _program(reservation):
    @T.prim_func
    def main(C: T.Tensor((16, 16), "float32"), flag: T.int32):
        with T.Kernel(1):
            accum = T.alloc_l0c((16, 16), "float32")
            temp = T.alloc_shared((8, 16), "float32")
            # This is an InsertSync input: Fixpipe produces UB and MTE3 reads
            # it. Layout inference prepares the scheduling input; no GEMM or
            # device execution is needed for this flag-allocation contract.
            T.dual_copy(accum, temp)
            if reservation == "sparse":
                T.ascend_sync_inter_arrive("PIPE_FIX", 15)
                T.ascend_sync_inter_wait("PIPE_FIX", 15)
            elif reservation == "bound":
                for outer in T.serial(2):
                    for inner in T.serial(2):
                        flag_id = T.bind(outer * 2 + inner)
                        T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
                        T.ascend_sync_inter_wait("PIPE_FIX", flag_id)
            elif reservation == "guard":
                for flag_id in T.serial(32):
                    if flag_id < 4:
                        T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
                        T.ascend_sync_inter_wait("PIPE_FIX", flag_id)
            else:
                if reservation == "assume":
                    T.assume(flag >= 0)
                    T.assume(flag < 4)
                T.ascend_sync_inter_arrive("PIPE_FIX", flag)
                T.ascend_sync_inter_wait("PIPE_FIX", flag)
            T.dual_copy(temp, C)

    return main


def _generated_slots(mod, operation):
    generated = [call for call in calls(mod, "tl.ascend_cross_core_" + operation + "_flag") if int(call.args[0]) == 4]
    assert generated, "Missing the Fixpipe/MTE3 cross-core handshake"
    # Both AIV subcores use the same 16-slot namespace, offset by subcore id.
    slots = set()
    for call in generated:
        if isinstance(call.args[2], tirx.IntImm):
            slots.add(int(call.args[2]) % 16)
    assert slots
    return slots


@pytest.mark.parametrize(
    "reservation, reserved",
    [
        pytest.param("sparse", {15}, id="sparse-explicit"),
        pytest.param("bound", set(range(4)), id="nested-bind"),
        pytest.param("guard", set(range(4)), id="guard-bounded"),
        pytest.param("assume", set(range(4)), id="assumption-bounded"),
    ],
)
def test_generated_flags_do_not_collide_with_explicit_flags(reservation, reserved):
    with tvm.transform.PassContext(config={"tl.Simplify": {"enable_simplify_let_inline": False}}):
        before = schedule(_program(reservation), through="ResolveCore", auto_schedule=False, infer_layouts=True)
        mod = transform.InsertSync()(before)
    sets = _generated_slots(mod, "set")
    waits = _generated_slots(mod, "wait")
    assert sets == waits
    assert not sets & reserved
    assert all(0 <= slot < 16 for slot in sets)


def test_unbounded_explicit_flag_is_rejected():
    with pytest.raises(tvm.error.InternalError, match="integer range cannot be bounded"):
        before = schedule(_program("unbounded"), through="ResolveCore", auto_schedule=False, infer_layouts=True)
        transform.InsertSync()(before)


@pytest.mark.parametrize("overlap", [False, True])
def test_dynamic_nz_prefix_keeps_subcore_write_regions(overlap):
    rows = tirx.Var("rows", "int32")
    l1 = tirx.decl_buffer((32, 64), "float8_e4m3fn", name="l1", scope="shared.l1")
    nz = [tirx.decl_buffer((17, 64), "float8_e4m3fn", name=f"nz{sid}", scope="shared.dyn") for sid in range(2)]
    copies = []
    for sid in range(2):
        offset = 0 if overlap else sid * 16
        call = T.ascend_nd2nz_post_copy(
            T.access_ptr(l1[offset, 0], "w", rows, 64),
            T.access_ptr(nz[sid][0, 0], "r", 17, 64),
            rows,
            64,
            32,
            "float8_e4m3_t",
        )
        copies.append(unit(tirx.Evaluate(call), core=1 if sid == 0 else 4))
    before = kernel(
        unit(tirx.For(rows, 1, 16, tirx.ForKind.SERIAL, seq(*copies)), core=None),
        buffers=[l1, *nz],
        annotations={"tl.split_subcores": True},
    )
    if overlap:
        with pytest.raises(tvm.error.InternalError, match="does not support synchronization between AIV0 and AIV1"):
            transform.InsertSync()(before)
    else:
        after = transform.InsertSync()(before)
        assert not calls(after, "tl.ascend_cross_core_set_flag")


@pytest.mark.parametrize("versions", [2, 15])
def test_alternating_epilogue_uses_each_aiv_flag_bank(versions):
    @T.prim_func
    def main(C: T.Tensor((7, 16, 16), "float32")):
        with T.MixedKernel(1) as (_, sid):
            accum = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            T.annotate_buffer_versions({ub: (versions, "counter")})
            for i in T.Pipelined(7, num_stages=versions):
                if sid == i % 2:
                    T.copy(accum, ub)
                    T.copy(ub, C[i, :, :])

    before = schedule(main, through="ResolveCore", auto_schedule=False, infer_layouts=True)
    (owner,) = nodes(before, tirx.For)
    counters = list(owner.annotations["tl.multi_buffer_counter_map"].values())
    assert len(counters) == 2 and not counters[0].same_as(counters[1])
    counter_masks = {
        int(task.node["core_mask"])
        for task in nodes(before, tirx.AttrStmt)
        if task.attr_key == "tl.ascend_task"
        and any(store.buffer.same_as(counter) for store in nodes(task.body, tirx.BufferStore) for counter in counters)
    }
    assert counter_masks == {3, 6}
    after = transform.InsertSync()(before)
    for task in nodes(after, tirx.AttrStmt):
        if task.attr_key != "tl.ascend_task":
            continue
        core = int(task.node["core_mask"])
        for call in calls(task.body, "tl.ascend_cross_core_set_flag") + calls(task.body, "tl.ascend_cross_core_wait_flag"):
            analyzer = tvm.arith.Analyzer()
            bound = analyzer.const_int_bound(analyzer.simplify(call.args[2]))
            assert bound.min_value >= 0
            assert bound.max_value < (32 if core == 2 else 16)
    cube_ids = {
        int(call.args[2])
        for task in nodes(after, tirx.AttrStmt)
        if task.attr_key == "tl.ascend_task" and int(task.node["core_mask"]) == 2
        for call in calls(task.body, "tl.ascend_cross_core_wait_flag")
        if isinstance(call.args[2], tirx.IntImm)
    }
    assert any(slot < 16 for slot in cube_ids) and any(slot >= 16 for slot in cube_ids)


@pytest.mark.parametrize("hazard", ["RAW", "WAR", "WAW", "disjoint"])
def test_cross_aiv_memory_dependencies_require_supported_ordering(hazard):
    shared = tirx.decl_buffer((2, 64), "float32", name="shared")
    ub0 = tirx.decl_buffer((64,), "float32", name="ub0", scope="shared.dyn")
    ub1 = tirx.decl_buffer((64,), "float32", name="ub1", scope="shared.dyn")
    write0 = copy(ub0, shared, dst_indices=[0, 0], dst_shape=[1, 64])
    read0 = copy(shared, ub0, src_indices=[0, 0], src_shape=[1, 64])
    write1 = copy(ub1, shared, dst_indices=[1 if hazard == "disjoint" else 0, 0], dst_shape=[1, 64])
    read1 = copy(shared, ub1, src_indices=[0, 0], src_shape=[1, 64])
    before = kernel(
        seq(unit(read0 if hazard == "WAR" else write0, core=1), unit(read1 if hazard == "RAW" else write1, core=4)),
        buffers=[ub0, ub1],
        params=[shared],
        annotations={"tl.split_subcores": True},
    )
    if hazard == "disjoint":
        after = transform.InsertSync()(before)
        assert len(calls(after, "tl.tileop.ascend_copy")) == 2
        assert not calls(after, "tl.ascend_cross_core_set_flag")
    else:
        with pytest.raises(tvm.error.InternalError, match="does not support synchronization between AIV0 and AIV1"):
            transform.InsertSync()(before)


@pytest.mark.parametrize("orders_accesses", [False, True], ids=["unrelated-cube-task", "cube-data-dependency"])
def test_only_actual_cube_dependencies_can_order_aiv_accesses(orders_accesses):
    shared = tirx.decl_buffer((64,), "float32", name="shared")
    unrelated = tirx.decl_buffer((64,), "float32", name="unrelated")
    ub0 = tirx.decl_buffer((64,), "float32", name="ub0", scope="shared.dyn")
    ub1 = tirx.decl_buffer((64,), "float32", name="ub1", scope="shared.dyn")
    l1 = tirx.decl_buffer((64,), "float32", name="l1", scope="shared.l1")
    before = kernel(
        seq(
            unit(copy(ub0, shared), core=1),
            unit(copy(shared if orders_accesses else unrelated, l1), core=2),
            unit(copy(ub1, shared), core=4),
        ),
        buffers=[ub0, ub1, l1],
        params=[shared, unrelated],
        annotations={"tl.split_subcores": True},
    )
    if not orders_accesses:
        with pytest.raises(tvm.error.InternalError, match="does not support synchronization between AIV0 and AIV1"):
            transform.InsertSync()(before)
        return
    after = transform.InsertSync()(before)
    endpoints = {"set": set(), "wait": set()}
    for task in nodes(after, tirx.AttrStmt):
        if task.attr_key != "tl.ascend_task":
            continue
        for operation, cores in endpoints.items():
            for call in calls(task.body, "tl.ascend_cross_core_" + operation + "_flag"):
                assert int(call.args[0]) == 4 and call.args[1].value != "PIPE_S"
                cores.add(int(task.node["core_mask"]))
    assert endpoints == {"set": {1, 2}, "wait": {2, 4}}


@pytest.mark.parametrize("scope", ["shared.l1", "global"])
def test_alternating_aiv_writers_without_cube_ordering_are_rejected(scope):
    @T.prim_func
    def main(A: T.Tensor((5, 16, 16), "float32"), C: T.Tensor((16, 16), "float32")):
        with T.MixedKernel(1) as (_, sid):
            ub = T.alloc_shared((16, 16), "float32")
            l1 = T.alloc_l1((16, 16), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(5, num_stages=2):
                if sid == i % 2:
                    T.copy(A[i, :, :], ub)
                    if scope == "shared.l1":
                        T.copy(ub, l1)
                    else:
                        T.copy(ub, C)

    before = schedule(main, through="ResolveCore", auto_schedule=False)
    with pytest.raises(tvm.error.InternalError, match="does not support synchronization between AIV0 and AIV1"):
        transform.InsertSync()(before)


if __name__ == "__main__":
    tilelang.testing.main()
