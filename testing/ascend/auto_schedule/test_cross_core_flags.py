"""Supported cross-core handshakes and user-reserved flag slots."""

from tvm_tirx import ir
import tvm_ffi
from testing.ascend._ir import access_tensor, attribute_fields, attribute_key, loop_annotations, tensor

import pytest
import tilelang.ascend.language as T
import tilelang.testing
import tvm_tirx
from tilelang.ascend import transform
from testing.ascend._ir import calls, copy, kernel, nodes, schedule, seq, substitute
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule._finite_protocol import assert_finite_protocol


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
        slot = tvm_tirx.sym.Analyzer().simplify(call.args[2])
        if isinstance(slot, ir.IntImm):
            slots.add(int(slot) % 16)
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
    with tvm_tirx.transform.PassContext(config={"tl.Simplify": {"enable_simplify_let_inline": False}}):
        before = schedule(_program(reservation), through="ResolveCore", auto_schedule=False, infer_layouts=True)
        mod = transform.InsertSync()(before)
    sets = _generated_slots(mod, "set")
    waits = _generated_slots(mod, "wait")
    assert sets == waits
    assert not sets & reserved
    assert all(0 <= slot < 16 for slot in sets)


def test_unbounded_explicit_flag_is_rejected():
    with pytest.raises((ValueError, RuntimeError), match="integer range cannot be bounded"):
        before = schedule(_program("unbounded"), through="ResolveCore", auto_schedule=False, infer_layouts=True)
        transform.InsertSync()(before)


@pytest.mark.parametrize("overlap", [False, True])
def test_dynamic_nz_prefix_keeps_subcore_write_regions(overlap):
    rows = ir.Var("rows", "int32")
    l1 = tensor((32, 64), "bfloat16", name="l1", scope="shared.l1")
    nz = [tensor((17, 64), "bfloat16", name=f"nz{sid}", scope="shared.dyn") for sid in range(2)]
    copies = []
    for sid in range(2):
        offset = 0 if overlap else sid * 16
        call = T.ascend_nd2nz_post_copy(
            T.access_ptr(l1[offset, 0], "w", rows, 64),
            T.access_ptr(nz[sid][0, 0], "r", 17, 64),
            rows,
            64,
            src_pitch=17,
            dst_pitch=32,
            dtype="bfloat16",
        )
        # This is a prepared InsertSync input with a proven NZ destination.
        call = tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
            ir.PrimType("void"), call.op, call.args, {"nd2nz_logical_regions": ir.IntImm("int32", 1)}, ir.UnknownLoc()
        )
        copies.append(unit(ir.Evaluate(call), core=1 if sid == 0 else 4))
    before = kernel(
        unit(ir.For(rows, ir.const(1), ir.const(1) + ir.const(16), ir.const(1), ir.ForKind.DEFAULT, seq(*copies)), core=None),
        buffers=[l1, *nz],
        annotations={"tl.split_subcores": True},
        allocation_stage=0,
    )
    if overlap:
        with pytest.raises((ValueError, RuntimeError), match="does not support synchronization between AIV0 and AIV1"):
            transform.InsertSync()(before)
    else:
        after = transform.InsertSync()(before)
        assert not calls(after, "tl.ascend_cross_core_set_flag")


@pytest.mark.parametrize("annotated_layout,dst_pitch,overlap", [(False, 64, True), (True, 16, True), (True, 64, False)])
def test_explicit_nz_pitch_cannot_hide_overlapping_subcore_writes(annotated_layout, dst_pitch, overlap):
    from tilelang.layout import make_ascend_nz_layout

    l1 = tensor((64, 64), "bfloat16", name="l1", scope="shared.l1")
    nz = [tensor((1024,), "bfloat16", name=f"nz{sid}", scope="shared.dyn") for sid in range(2)]
    copies = []
    for sid in range(2):
        # Without a layout the second pointer is at element 1024, exactly the
        # first DMA's second burst. With NZ layout, a 16-block pitch similarly
        # makes that burst overlap row 16 despite disjoint logical row ranges.
        call = T.ascend_nd2nz_post_copy(l1[sid * 16, 0], nz[sid][0], 8, 32, src_pitch=8, dst_pitch=dst_pitch, dtype="bfloat16")
        copies.append(unit(ir.Evaluate(call), core=1 if sid == 0 else 4))
    annotations = {"tl.split_subcores": True}
    if annotated_layout:
        annotations["layout_map"] = {l1: make_ascend_nz_layout(l1)}
    before = transform.InsertNd2Nz()(kernel(seq(*copies), buffers=[l1, *nz], annotations=annotations, allocation_stage=0))
    if overlap:
        with pytest.raises((ValueError, RuntimeError), match="does not support synchronization between AIV0 and AIV1"):
            transform.InsertSync()(before)
    else:
        transform.InsertSync()(before)


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
    (owner,) = nodes(before, ir.For)
    counters = list(loop_annotations(owner)["tl.multi_buffer_counter_map"].values())
    assert len(counters) == 2 and not counters[0].same_as(counters[1])
    counter_masks = {
        int(attribute_fields(task)["node"]["core_mask"])
        for task in nodes(before, ir.RegionStmt)
        if attribute_key(task) == "tl.ascend_task"
        and any(access_tensor(store).same_as(counter) for store in nodes(task.body, ir.TensorStore) for counter in counters)
    }
    assert counter_masks == {3, 6}
    after = transform.InsertSync()(before)
    for task in nodes(after, ir.RegionStmt):
        if attribute_key(task) != "tl.ascend_task":
            continue
        core = int(attribute_fields(task)["node"]["core_mask"])
        for call in calls(task.body, "tl.ascend_cross_core_set_flag") + calls(task.body, "tl.ascend_cross_core_wait_flag"):
            analyzer = tvm_tirx.sym.Analyzer()
            bound = analyzer.const_int_bound(analyzer.simplify(call.args[2]))
            assert bound.min_value >= 0
            assert bound.max_value < (32 if core == 2 else 16)
    cube_ids = {
        int(tvm_tirx.sym.Analyzer().simplify(call.args[2]))
        for task in nodes(after, ir.RegionStmt)
        if attribute_key(task) == "tl.ascend_task" and int(attribute_fields(task)["node"]["core_mask"]) == 2
        for call in calls(task.body, "tl.ascend_cross_core_wait_flag")
        if isinstance(tvm_tirx.sym.Analyzer().simplify(call.args[2]), ir.IntImm)
    }
    assert any(slot < 16 for slot in cube_ids) and any(slot >= 16 for slot in cube_ids)


@pytest.mark.parametrize("hazard", ["RAW", "WAR", "WAW", "disjoint"])
def test_cross_aiv_memory_dependencies_require_supported_ordering(hazard):
    shared = tensor((2, 64), "float32", name="shared")
    ub0 = tensor((64,), "float32", name="ub0", scope="shared.dyn")
    ub1 = tensor((64,), "float32", name="ub1", scope="shared.dyn")
    write0 = copy(ub0, shared, dst_indices=[0, 0], dst_shape=[1, 64])
    read0 = copy(shared, ub0, src_indices=[0, 0], src_shape=[1, 64])
    write1 = copy(ub1, shared, dst_indices=[1 if hazard == "disjoint" else 0, 0], dst_shape=[1, 64])
    read1 = copy(shared, ub1, src_indices=[0, 0], src_shape=[1, 64])
    before = kernel(
        seq(unit(read0 if hazard == "WAR" else write0, core=1), unit(read1 if hazard == "RAW" else write1, core=4)),
        buffers=[ub0, ub1],
        params=[shared],
        annotations={"tl.split_subcores": True},
        allocation_stage=0,
    )
    if hazard == "disjoint":
        after = transform.InsertSync()(before)
        assert len(calls(after, "tl.tileop.ascend_copy")) == 2
        assert not calls(after, "tl.ascend_cross_core_set_flag")
    else:
        with pytest.raises((ValueError, RuntimeError), match="does not support synchronization between AIV0 and AIV1"):
            transform.InsertSync()(before)


@pytest.mark.parametrize("orders_accesses", [False, True], ids=["unrelated-cube-task", "cube-data-dependency"])
def test_only_actual_cube_dependencies_can_order_aiv_accesses(orders_accesses):
    shared = tensor((64,), "float32", name="shared")
    unrelated = tensor((64,), "float32", name="unrelated")
    ub0 = tensor((64,), "float32", name="ub0", scope="shared.dyn")
    ub1 = tensor((64,), "float32", name="ub1", scope="shared.dyn")
    l1 = tensor((64,), "float32", name="l1", scope="shared.l1")
    before = kernel(
        seq(
            unit(copy(ub0, shared), core=1),
            unit(copy(shared if orders_accesses else unrelated, l1), core=2),
            unit(copy(ub1, shared), core=4),
        ),
        buffers=[ub0, ub1, l1],
        params=[shared, unrelated],
        annotations={"tl.split_subcores": True},
        allocation_stage=0,
    )
    after = transform.InsertSync()(before)
    if not orders_accesses:
        # Distinct global formals may alias. Only an actual mode4 completion
        # path through the Cube can cover the resulting AIV dependency.
        assert_finite_protocol(before, after, "cube_bridge")
        independent = ir.IRModule({"main": before["main"].with_attr("tirx.noalias", True)})
        with pytest.raises((ValueError, RuntimeError), match="does not support synchronization between AIV0 and AIV1"):
            transform.InsertSync()(independent)
        assert "tirx.noalias" not in before["main"].attrs
    endpoints = {"set": set(), "wait": set()}
    for task in nodes(after, ir.RegionStmt):
        if attribute_key(task) != "tl.ascend_task":
            continue
        for operation, cores in endpoints.items():
            for call in calls(task.body, "tl.ascend_cross_core_" + operation + "_flag"):
                assert int(call.args[0]) == 4 and call.args[1].value != "PIPE_S"
                cores.add(int(attribute_fields(task)["node"]["core_mask"]))
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
    with pytest.raises((ValueError, RuntimeError), match="does not support synchronization between AIV0 and AIV1"):
        transform.InsertSync()(before)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("to_cube", [False, True], ids=["cube-to-aivs", "aivs-to-cube"])
@pytest.mark.parametrize("versions", [1, 3], ids=["publication", "ring-reuse"])
def test_ss_uses_standard_cross_core_sync(packed, to_cube, versions):
    ss = tensor((2,), "uint32", name="ss", scope="shared.ss")
    view = tensor((1,), "uint64", name="view", scope="shared.ss") if packed else ss
    out = tensor((2,), "uint32", name="out")
    if to_cube:
        tasks = [unit(ir.TensorStore(ss, [sid], ir.const(7, "uint32")), core=core) for sid, core in enumerate((1, 4))]
        value = ir.prim.Cast("uint32", view[0]) if packed else ss[0] + ss[1]
        tasks.append(unit(ir.TensorStore(out, [0], value), core=2))
    else:
        writes = [ir.TensorStore(view, [i], ir.const(7, view.ty.dtype)) for i in range(1 if packed else 2)]
        tasks = [unit(seq(*writes), core=2)]
        tasks += [unit(ir.TensorStore(out, [sid], ss[sid]), core=core) for sid, core in enumerate((1, 4))]
    body = seq(*tasks)
    annotations = {"tl.split_subcores": True}
    i = ir.Var("i", "int32")
    if versions > 1:
        body = unit(
            ir.For(
                i,
                ir.const(0),
                ir.const(0) + ir.const(8),
                ir.const(1),
                ir.ForKind.DEFAULT,
                body,
                annotations={"multi_buffer_eligible": [ss]},
            ),
            core=None,
        )
        annotations["tl.buffer_versions_map"] = {ss: versions}
    if packed:
        pointer = ir.Call("tirx.reinterpret", [ss.data], ty=ir.PtrType("uint64", ss.ty.scope))
        declaration = unit(ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])), core=7)
        body = seq(declaration, body)
    before = kernel(body, buffers=[ss], params=[out], annotations=annotations, allocation_stage=0)
    after = transform.InsertSync()(before)
    endpoints = {"set": [], "wait": []}
    for task in nodes(after, ir.RegionStmt):
        if attribute_key(task) != "tl.ascend_task":
            continue
        for operation, events in endpoints.items():
            for call in calls(task.body, "tl.ascend_cross_core_" + operation + "_flag"):
                assert int(call.args[0]) == 4 and call.args[1].value == "PIPE_S"
                events.append((int(attribute_fields(task)["node"]["core_mask"]), tvm_tirx.sym.Analyzer().simplify(call.args[2])))
    if versions > 1:
        initial, drained = (
            {int(slot) + (16 if core == 4 else 0) for core, slot in endpoints[operation] if isinstance(slot, ir.IntImm)}
            for operation in ("set", "wait")
        )
        assert initial == drained and len(initial) == 2 * versions
        (ring_loop,) = nodes(after, ir.For)
        assert (int(ring_loop.begin), int(ring_loop.end), int(ring_loop.step)) == (0, 8, 1)
        analyzer = tvm_tirx.sym.Analyzer()
        for operation, events in endpoints.items():
            dynamic = [(core, slot) for core, slot in events if not isinstance(slot, ir.IntImm)]
            assert len(dynamic) == 4  # Publication and reuse, separately for both AIVs.
            bases = []
            for core, slot in dynamic:
                # Check every iteration of this finite original loop. Keep
                # native casts and substitute the actual output loop Var.
                differences = set()
                for index in range(int(ring_loop.begin), int(ring_loop.end), int(ring_loop.step)):
                    actual = analyzer.simplify(substitute(slot, {ring_loop.var: ir.IntImm(ring_loop.var.ty, index)}))
                    assert isinstance(actual, ir.IntImm)
                    differences.add(int(actual) - index % versions)
                assert len(differences) == 1
                bases.append((core, differences.pop()))
            endpoints[operation] = bases
        for sid, core in enumerate((1, 4)):
            for operation, peer in (("set", "wait"), ("wait", "set")):
                vector = {slot + 16 * sid for endpoint, slot in endpoints[operation] if endpoint == core}
                cube = {slot for endpoint, slot in endpoints[peer] if endpoint == 2 and 16 * sid <= slot < 16 * (sid + 1)}
                assert vector == cube and len(vector) == 1
        return
    endpoints = {operation: [(core, int(slot)) for core, slot in events] for operation, events in endpoints.items()}
    vectors = endpoints["set" if to_cube else "wait"]
    cube = endpoints["wait" if to_cube else "set"]
    assert {core for core, _ in vectors} == {1, 4}
    assert len(vectors) == len(cube) == 2
    assert set(cube) == {(2, slot + (16 if core == 4 else 0)) for core, slot in vectors}


def test_ss_disjoint_aiv_read_modify_write_needs_no_sync():
    ss = tensor((2,), "uint32", name="ss", scope="shared.ss")
    before = kernel(
        seq(*(unit(ir.TensorStore(ss, [sid], ss[sid] + 1), core=core) for sid, core in enumerate((1, 4)))),
        buffers=[ss],
        annotations={"tl.split_subcores": True},
        allocation_stage=0,
    )
    after = transform.InsertSync()(before)
    assert not calls(after, "tl.ascend_cross_core_set_flag")
    assert not calls(after, "tl.ascend_cross_core_wait_flag")


if __name__ == "__main__":
    tilelang.testing.main()
