"""InsertSync protects RAW and storage reuse independently of scheduling."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_major_k_layout, make_ascend_major_mn_layout
from tvm import tirx
from testing.ascend._ir import allocated_buffer, calls, copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit


def _events(mod, operation, pipe):
    return [call for call in calls(mod, "tl.ascend_" + operation + "_flag") if call.args[0].value == pipe]


@pytest.mark.parametrize("mode, owners, guarded", [("iteration", 1, False), ("counter", 1, True), ("counter", 2, True)])
def test_ring_flags_use_the_data_clock(mode, owners, guarded):
    before = copy_ring(mode=mode, owners=owners, guarded=guarded, prepared=True)
    after = transform.InsertSync()(before)
    clock = allocated_buffer(before, "epoch")[0] if mode == "counter" else nodes(before, tirx.For)[0].loop_var
    analyzer = tvm.arith.Analyzer()
    for pipe in ("MTE2_MTE3", "MTE3_MTE2"):
        for operation in ("set", "wait"):
            events = _events(after, operation, pipe)
            dynamic = [event.args[1] for event in events if not isinstance(event.args[1], tirx.IntImm)]
            assert len(dynamic) == owners
            for flag in dynamic:
                base = analyzer.simplify(flag - clock % 2)
                assert isinstance(base, tirx.IntImm), "The ring must advance with the storage clock"
            if operation == "set":
                released = dynamic
            else:
                assert all(analyzer.can_prove_equal(left, right) for left, right in zip(released, dynamic))
    # Ring reuse needs an initial token and a final drain for every slot. IDs
    # need only match each other, not a particular allocator numbering.
    initial = {int(event.args[1]) for event in _events(after, "set", "MTE3_MTE2") if isinstance(event.args[1], tirx.IntImm)}
    drained = {int(event.args[1]) for event in _events(after, "wait", "MTE3_MTE2") if isinstance(event.args[1], tirx.IntImm)}
    assert initial == drained and len(initial) == 2
    if guarded:
        guarded_events = []
        for stmt, parents in statements(after):
            if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
                continue
            if stmt.value.op.name not in ("tl.ascend_set_flag", "tl.ascend_wait_flag") or isinstance(stmt.value.args[1], tirx.IntImm):
                continue
            owner = next(parent for parent in reversed(parents) if isinstance(parent, tirx.For))
            assert any(
                isinstance(parent, tirx.IfThenElse) and analyzer.can_prove_equal(parent.condition, owner.loop_var % 2 == 0)
                for parent in parents
            )
            guarded_events.append(stmt)
        assert guarded_events


@pytest.mark.parametrize("guard_inside_task", [False, True], ids=["guarded-task", "atomic-internal-guard"])
def test_sync_respects_atomic_task_boundary(guard_inside_task):
    a = tirx.decl_buffer((64,), "float32", name="A")
    c = tirx.decl_buffer((64,), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    enabled = tirx.Var("enabled", "bool")
    tasks = [copy(a, ub), copy(ub, c)]
    if guard_inside_task:
        tasks = [unit(tirx.IfThenElse(enabled, task, None)) for task in tasks]
    else:
        tasks = [unit(task, guard=enabled) for task in tasks]
    before = kernel(seq(*tasks), buffers=[ub], params=[a, c, enabled])
    after = transform.InsertSync()(before)
    events = _events(after, "set", "MTE2_MTE3") + _events(after, "wait", "MTE2_MTE3")
    assert len(events) == 2
    for stmt, parents in statements(after):
        if isinstance(stmt, tirx.Evaluate) and any(stmt.value.same_as(event) for event in events):
            guards = [parent.condition for parent in parents if isinstance(parent, tirx.IfThenElse)]
            if guard_inside_task:
                assert not guards
            if guards:
                tvm.ir.assert_structural_equal(guards[-1], enabled)


def test_dma_issue_order_is_not_completion_order():
    a = tirx.decl_buffer((64,), "float32", name="A")
    c = tirx.decl_buffer((64,), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    # A DMA store must finish before MTE2 may overwrite its UB source.
    before = kernel(seq(unit(copy(ub, c)), unit(copy(a, ub))), buffers=[ub], params=[a, c])
    after = transform.InsertSync()(before)
    assert len(_events(after, "set", "MTE3_MTE2")) == 1
    assert len(_events(after, "wait", "MTE3_MTE2")) == 1


def test_manually_indexed_storage_keeps_the_lexical_clock_under_a_guard():
    a = tirx.decl_buffer((4, 64), "float32", name="A")
    c = tirx.decl_buffer((4, 64), "float32", name="C")
    ub = tirx.decl_buffer((2, 64), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    body = seq(
        unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64], dst_indices=[i % 2, 0], dst_shape=[1, 64]), guard=i % 2 == 0),
        unit(copy(ub, c, src_indices=[i % 2, 0], src_shape=[1, 64], dst_indices=[i, 0], dst_shape=[1, 64]), guard=i % 2 == 0),
    )
    before = kernel(
        unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body), core=None),
        buffers=[ub],
        params=[a, c],
        annotations={"tl.manual_multi_buffer": {ub.data: 2}},
    )
    after = transform.InsertSync()(before)
    dynamic = []
    for stmt, parents in statements(after):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if call.op.name not in ("tl.ascend_set_flag", "tl.ascend_wait_flag") or call.args[0].value != "MTE2_MTE3":
            continue
        assert not any(isinstance(parent, tirx.IfThenElse) for parent in parents)
        assert isinstance(tvm.arith.Analyzer().simplify(call.args[1] - i % 2), tirx.IntImm)
        dynamic.append(call)
    assert len(dynamic) == 2
    tvm.ir.assert_structural_equal(dynamic[0].args[1], dynamic[1].args[1])


@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint-rows", "same-row"])
def test_dma_row_writes_only_serialize_overlapping_regions(overlap):
    a = tirx.decl_buffer((2, 64), "float32", name="A")
    out = tirx.decl_buffer((2, 64), "float32", name="out")
    ub = tirx.decl_buffer((2, 64), "float32", name="ub", scope="shared.dyn")
    row = tirx.Var("row", "int32")
    load = copy(a, ub, src_indices=[row, 0], dst_indices=[0 if overlap else row, 0], src_shape=[1, 64], dst_shape=[1, 64])
    rows = tirx.For(row, 0, 2, tirx.ForKind.SERIAL, unit(load))
    before = kernel(seq(unit(rows, core=None), unit(copy(ub, out))), buffers=[ub], params=[a, out])
    after = transform.InsertSync()(before)
    barriers = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE2"]
    assert bool(barriers) == overlap
    # The whole-buffer consumer must still wait for the row loads to finish.
    releases = _events(after, "set", "MTE2_MTE3")
    waits = _events(after, "wait", "MTE2_MTE3")
    assert len(releases) == len(waits) == 1
    tvm.ir.assert_structural_equal(releases[0].args[1], waits[0].args[1])


@pytest.mark.parametrize(
    "overlap, declared_rows",
    [(False, 32), (False, 48), (True, 48)],
    ids=["disjoint-halves", "declared-overlap", "overwrite-middle"],
)
def test_partial_l0_loads_complete_before_gemm_and_order_overlapping_writes(overlap, declared_rows):
    src = tirx.decl_buffer((64, 64), "bfloat16", name="src", scope="shared.l1")
    a = tirx.decl_buffer((64, 64), "bfloat16", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((64, 64), "bfloat16", name="b", scope="shared.l0b")
    c = tirx.decl_buffer((64, 64), "float32", name="c", scope="shared.l0c")
    # B and the L1 source are already prepared at this pass boundary.
    # Dependencies conservatively cover the declared destination region.
    body = seq(
        unit(
            copy(
                src,
                a,
                src_shape=[48 if overlap else 32, 64],
                dst_shape=[declared_rows, 64],
                annotations={"l0_pitch": tirx.IntImm("int32", 4)},
            ),
            core=2,
        ),
        unit(
            copy(
                src,
                a,
                src_indices=[32, 0],
                dst_indices=[32, 0],
                src_shape=[32, 64],
                dst_shape=[32, 64],
                annotations={"l0_pitch": tirx.IntImm("int32", 4)},
            ),
            core=2,
        ),
        unit(tirx.Evaluate(T.gemm(a, b, c, transpose_B=True, clear_accum=True)), core=2),
    )
    layouts = {src: make_ascend_major_k_layout(src), a: make_ascend_major_k_layout(a)}
    after = transform.InsertSync()(kernel(body, buffers=[src, a, b, c], annotations={"layout_map": layouts}))
    barriers = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE1"]
    assert bool(barriers) == (declared_rows > 32)
    releases, waits = _events(after, "set", "MTE1_M"), _events(after, "wait", "MTE1_M")
    assert releases and waits
    for release in releases:
        assert any(tvm.ir.structural_equal(release.args[1], wait.args[1]) for wait in waits)
    operations = [stmt.value for stmt in nodes(after, tirx.Evaluate) if isinstance(stmt.value, tirx.Call)]
    loads = [i for i, call in enumerate(operations) if call.op.name == "tl.tileop.ascend_copy"]
    release_sites = [i for i, call in enumerate(operations) if any(call.same_as(release) for release in releases)]
    wait_sites = [i for i, call in enumerate(operations) if any(call.same_as(wait) for wait in waits)]
    gemm_site = next(i for i, call in enumerate(operations) if call.op.name == "tl.tileop.gemm")
    assert len(loads) == 2 and max(loads) < max(release_sites) <= max(wait_sites) < gemm_site


@pytest.mark.parametrize("transposed", [False, True], ids=["major-k", "major-mn"])
@pytest.mark.parametrize(
    "plane, mn_start, k_start, mn_size, k_size, overlap",
    [(1, 32, 0, 32, 64, False), (0, 32, 0, 32, 64, True), (0, 48, 48, 16, 16, True)],
    ids=["separate-planes", "physical-overlap", "different-pitch-disjoint-regions"],
)
def test_different_l0_pitches_conflict_within_same_plane(transposed, plane, mn_start, k_start, mn_size, k_size, overlap):
    src = tirx.decl_buffer((64, 64), "bfloat16", name="src", scope="shared.l1")
    a = tirx.decl_buffer((2, 64, 64), "bfloat16", name="a", scope="shared.l0b" if transposed else "shared.l0a")
    b = tirx.decl_buffer((64, 64), "bfloat16", name="b", scope="shared.l0a" if transposed else "shared.l0b")
    c = tirx.decl_buffer((32, 64), "float32", name="c", scope="shared.l0c")
    # GEMM reads rows [0,32) at pitch=2. A subsequent load of [32,64)
    # at pitch=4 overlaps that compact storage, despite logical separation.
    operand_shape = (1, 64, 32) if transposed else (1, 32, 64)
    lhs = tirx.BufferRegion(a, [tvm.ir.Range.from_min_extent(0, e) for e in operand_shape])
    if transposed:
        c = tirx.decl_buffer((64, 32), "float32", name="c", scope="shared.l0c")
        gemm = T.gemm(b, lhs, c, clear_accum=True)
        start, shape = [k_start, mn_start], [k_size, mn_size]
    else:
        gemm = T.gemm(lhs, b, c, transpose_B=True, clear_accum=True)
        start, shape = [mn_start, k_start], [mn_size, k_size]
    body = seq(
        unit(tirx.Evaluate(gemm), core=2),
        unit(
            copy(
                src,
                a,
                src_shape=shape,
                dst_shape=[1, *shape],
                dst_indices=[plane, *start],
                annotations={"l0_pitch": tirx.IntImm("int32", 4)},
            ),
            core=2,
        ),
    )
    layouts = {
        src: make_ascend_major_k_layout(src),
        a: (make_ascend_major_mn_layout if transposed else make_ascend_major_k_layout)(a),
    }
    after = transform.InsertSync()(kernel(body, buffers=[src, a, b, c], annotations={"layout_map": layouts}))
    assert bool(_events(after, "set", "M_MTE1")) == overlap
    assert bool(_events(after, "wait", "M_MTE1")) == overlap


@pytest.mark.parametrize("pitch_relation", ["same", "equivalent", "unknown"])
def test_partial_l0_copies_require_provably_equal_pitches(pitch_relation):
    src = tirx.decl_buffer((64, 64), "bfloat16", name="src", scope="shared.l1")
    a = tirx.decl_buffer((80, 64), "bfloat16", name="a", scope="shared.l0a")
    p, q = tirx.Var("p", "int32"), tirx.Var("q", "int32")
    first_pitch = 4 + p % 2
    second_pitch = {"same": first_pitch, "equivalent": 5 - (1 - p % 2), "unknown": 4 + q % 2}[pitch_relation]
    body = seq(
        *(
            unit(
                copy(
                    src,
                    a,
                    src_indices=[start, 0],
                    dst_indices=[start, 0],
                    src_shape=[32, 64],
                    dst_shape=[32, 64],
                    annotations={"l0_pitch": pitch},
                ),
                core=2,
            )
            for start, pitch in [(0, first_pitch), (32, second_pitch)]
        )
    )
    layouts = {src: make_ascend_major_k_layout(src), a: make_ascend_major_k_layout(a)}
    after = transform.InsertSync()(kernel(body, buffers=[src, a], params=[p, q], annotations={"layout_map": layouts}))
    barriers = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE1"]
    assert bool(barriers) == (pitch_relation == "unknown")


@pytest.mark.parametrize("varying_pitch", [False, True], ids=["disjoint", "overlap-after-pitch-change"])
def test_l0_copy_pitch_is_instantiated_per_iteration(varying_pitch):
    src = tirx.decl_buffer((16, 16), "bfloat16", name="src", scope="shared.l1")
    a = tirx.decl_buffer((32, 32), "bfloat16", name="a", scope="shared.l0a")
    i = tirx.Var("i", "int32")
    # Disjoint logical blocks (m,k)=(0,1),(1,0). With pitches 1,2 they
    # both address physical block 1; at a fixed pitch 2 they address 2,1.
    load = copy(
        src,
        a,
        dst_indices=[16 * i, 16 * (1 - i)],
        dst_shape=[16, 16],
        annotations={"l0_pitch": 1 + i if varying_pitch else tirx.IntImm("int32", 2)},
    )
    body = unit(tirx.For(i, 0, 2, tirx.ForKind.SERIAL, unit(load, core=2)), core=None)
    layouts = {src: make_ascend_major_k_layout(src), a: make_ascend_major_k_layout(a)}
    after = transform.InsertSync()(kernel(body, buffers=[src, a], annotations={"layout_map": layouts}))
    barriers = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE1"]
    assert bool(barriers) == varying_pitch


@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint", "overlap"])
@pytest.mark.parametrize("unroll", [False, True], ids=["internal-loop", "multiple-calls"])
@pytest.mark.parametrize("varying_pitch", [False, True], ids=["same-pitch", "mixed-pitches"])
def test_atomic_l0_copies_require_a_uniform_pitch(overlap, unroll, varying_pitch):
    src = tirx.decl_buffer((16, 16), "bfloat16", name="src", scope="shared.l1")
    a = tirx.decl_buffer((32, 48), "bfloat16", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((16, 32 if overlap else 16), "bfloat16", name="b", scope="shared.l0b")
    c = tirx.decl_buffer((16, 16), "float32", name="c", scope="shared.l0c")
    i = tirx.Var("i", "int32")
    load = copy(
        src,
        a,
        dst_indices=[0, 16 * (i + 1)],
        dst_shape=[16, 16],
        annotations={"l0_pitch": 1 + i if varying_pitch else tirx.IntImm("int32", 1)},
    )
    lhs = tirx.BufferRegion(a, [tvm.ir.Range.from_min_extent(0, e) for e in (16, 32 if overlap else 16)])
    # Mixed pitches conservatively conflict even when the K regions are disjoint.
    loads = (
        seq(*(tirx.stmt_functor.substitute(load, {i: value}) for value in (0, 1)))
        if unroll
        else tirx.For(i, 0, 2, tirx.ForKind.SERIAL, load)
    )
    body = seq(
        unit(loads, core=2),
        unit(tirx.Evaluate(T.gemm(lhs, b, c, transpose_B=True, clear_accum=True)), core=2),
    )
    layouts = {src: make_ascend_major_k_layout(src), a: make_ascend_major_k_layout(a)}
    after = transform.InsertSync()(kernel(body, buffers=[src, a, b, c], annotations={"layout_map": layouts}))
    assert bool(_events(after, "set", "MTE1_M")) == (overlap or varying_pitch)
    assert bool(_events(after, "wait", "MTE1_M")) == (overlap or varying_pitch)


def test_equivalent_lexical_epoch_serializes_counter_ring():
    a = tirx.decl_buffer((64,), "float32", name="A")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    epoch = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var")
    i = tirx.Var("i", "int32")
    guard = i % 2 == 0
    compute = tirx.SBlock([], [], [], "SIMD_VF", tirx.BufferStore(ub, ub[0], [0]))
    body = seq(
        unit(copy(a, ub), guard=guard),
        unit(compute, guard=guard),
        unit(copy(ub, a), guard=guard),
        unit(tirx.BufferStore(epoch, epoch[0] + 1, [0]), guard=guard),
    )
    owner = tirx.For(
        i,
        0,
        4,
        tirx.ForKind.SERIAL,
        body,
        annotations={
            "multi_buffer_eligible": [ub.data],
            "tl.multi_buffer_counter_map": {ub.data: epoch},
            "tl.storage_epoch_guard_map": {ub.data: guard, a.data: guard},
        },
    )
    before = kernel(
        seq(unit(tirx.BufferStore(epoch, 0, [0])), unit(owner, core=None)),
        buffers=[ub, epoch],
        params=[a],
        annotations={"tl.buffer_versions_map": {ub.data: 2}},
    )
    after = transform.InsertSync()(before)
    # GM is reused every active epoch, so its lexical MTE3->MTE2 closure
    # also serializes the equivalent counter domain despite two UB versions.
    for pipe in ("MTE2_V", "V_MTE3", "MTE3_MTE2"):
        released = _events(after, "set", pipe)
        acquired = _events(after, "wait", pipe)
        assert released and acquired
        assert all(isinstance(event.args[1], tirx.IntImm) for event in released + acquired)
        assert sorted(int(event.args[1]) for event in released) == sorted(int(event.args[1]) for event in acquired)
    root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
    assert int(root.annotations["tl.buffer_versions_map"][ub.data]) == 2
