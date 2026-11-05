"""InsertSync protects RAW and storage reuse independently of scheduling."""

from tvm_tirx import ir
import tvm_ffi
from testing.ascend._ir import sblock, sblock_annotations, sblock_metadata, sblocks, substitute, tensor

import pytest
import tvm_tirx
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_major_k_layout, make_ascend_major_mn_layout
from testing.ascend._ir import allocated_buffer, calls, copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit
from testing.ascend.auto_schedule._finite_protocol import assert_finite_protocol


def _events(mod, operation, pipe):
    return [call for call in calls(mod, "tl.ascend_" + operation + "_flag") if call.args[0].value == pipe]


@pytest.mark.parametrize("mode, owners, guarded", [("iteration", 1, False), ("counter", 1, True), ("counter", 2, True)])
def test_ring_flags_use_the_data_clock(mode, owners, guarded):
    before = copy_ring(mode=mode, owners=owners, guarded=guarded, prepared=True)
    after = transform.InsertSync()(before)
    case = "iteration_ring" if mode == "iteration" else {1: "counter_ring_one", 2: "counter_ring_two"}[owners]
    assert_finite_protocol(before, after, case)
    if mode == "iteration":
        # Static tokens may serialize the ring when A/C can alias. The
        # finite proof still checks actual addresses, epochs and completion.
        return
    clock = allocated_buffer(before, "epoch")[0] if mode == "counter" else nodes(before, ir.For)[0].var
    analyzer = tvm_tirx.sym.Analyzer()
    ring_slots = set()
    for pipe in ("MTE2_MTE3", "MTE3_MTE2"):
        for operation in ("set", "wait"):
            events = _events(after, operation, pipe)
            identifiers = [analyzer.simplify(event.args[1]) for event in events]
            dynamic = [identifier for identifier in identifiers if not isinstance(identifier, ir.IntImm)]
            assert len(dynamic) == owners
            for flag in dynamic:
                base = analyzer.simplify(flag - clock % 2)
                assert isinstance(base, ir.IntImm), "The ring must advance with the storage clock"
                if pipe == "MTE3_MTE2":
                    ring_slots.update((int(base), int(base) + 1))
            if operation == "set":
                released = dynamic
            else:
                assert all(analyzer.can_prove_equal(left, right) for left, right in zip(released, dynamic))
    # Counter slots retain their dynamic data clock. Additional static alias
    # channels are checked by the full protocol proof, separately from them.
    initial, drained = set(), set()
    for stmt, parents in statements(after):
        if not isinstance(stmt, ir.Evaluate) or not isinstance(stmt.value, ir.Call):
            continue
        call = stmt.value
        if call.op.name not in ("tl.ascend_set_flag", "tl.ascend_wait_flag") or call.args[0].value != "MTE3_MTE2":
            continue
        if any(isinstance(parent, ir.For) for parent in parents):
            continue
        identifier = analyzer.simplify(call.args[1])
        assert isinstance(identifier, ir.IntImm)
        (initial if call.op.name == "tl.ascend_set_flag" else drained).add(int(identifier))
    assert len(ring_slots) == 2 and ring_slots <= initial
    assert initial == drained
    if guarded:
        guarded_events = []
        for stmt, parents in statements(after):
            if not isinstance(stmt, ir.Evaluate) or not isinstance(stmt.value, ir.Call):
                continue
            if stmt.value.op.name not in ("tl.ascend_set_flag", "tl.ascend_wait_flag") or isinstance(
                analyzer.simplify(stmt.value.args[1]), ir.IntImm
            ):
                continue
            owner = next(parent for parent in reversed(parents) if isinstance(parent, ir.For))
            assert any(isinstance(parent, ir.If) and analyzer.can_prove_equal(parent.condition, owner.var % 2 == 0) for parent in parents)
            guarded_events.append(stmt)
        assert guarded_events


@pytest.mark.parametrize("guard_inside_task", [False, True], ids=["guarded-task", "atomic-internal-guard"])
def test_sync_respects_atomic_task_boundary(guard_inside_task):
    a = tensor((64,), "float32", name="A")
    c = tensor((64,), "float32", name="C")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    enabled = ir.Var("enabled", "bool")
    tasks = [copy(a, ub), copy(ub, c)]
    if guard_inside_task:
        tasks = [unit(ir.If(enabled, task, None)) for task in tasks]
    else:
        tasks = [unit(task, guard=enabled) for task in tasks]
    before = kernel(seq(*tasks), buffers=[ub], params=[a, c, enabled], allocation_stage=0)
    after = transform.InsertSync()(before)
    events = _events(after, "set", "MTE2_MTE3") + _events(after, "wait", "MTE2_MTE3")
    assert len(events) == 2
    for stmt, parents in statements(after):
        if isinstance(stmt, ir.Evaluate) and any(stmt.value.same_as(event) for event in events):
            guards = [parent.condition for parent in parents if isinstance(parent, ir.If)]
            if guard_inside_task:
                assert not guards
            if guards:
                tvm_tirx.ir.assert_structural_equal(guards[-1], enabled)


def test_dma_issue_order_is_not_completion_order():
    a = tensor((64,), "float32", name="A")
    c = tensor((64,), "float32", name="C")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    # A DMA store must finish before MTE2 may overwrite its UB source.
    before = kernel(seq(unit(copy(ub, c)), unit(copy(a, ub))), buffers=[ub], params=[a, c], allocation_stage=0)
    after = transform.InsertSync()(before)
    assert len(_events(after, "set", "MTE3_MTE2")) == 1
    assert len(_events(after, "wait", "MTE3_MTE2")) == 1


def test_manually_indexed_storage_keeps_the_lexical_clock_under_a_guard():
    a = tensor((4, 64), "float32", name="A")
    c = tensor((4, 64), "float32", name="C")
    ub = tensor((2, 64), "float32", name="ub", scope="shared.dyn")
    i = ir.Var("i", "int32")
    body = seq(
        unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64], dst_indices=[i % 2, 0], dst_shape=[1, 64]), guard=i % 2 == 0),
        unit(copy(ub, c, src_indices=[i % 2, 0], src_shape=[1, 64], dst_indices=[i, 0], dst_shape=[1, 64]), guard=i % 2 == 0),
    )
    before = kernel(
        unit(ir.For(i, ir.const(0), ir.const(0) + ir.const(4), ir.const(1), ir.ForKind.DEFAULT, body), core=None),
        buffers=[ub],
        params=[a, c],
        annotations={"tl.manual_multi_buffer": {ub: 2}},
        allocation_stage=0,
    )
    after = transform.InsertSync()(before)
    assert_finite_protocol(before, after, "manual_ring")
    dynamic = []
    for stmt, parents in statements(after):
        if not isinstance(stmt, ir.Evaluate) or not isinstance(stmt.value, ir.Call):
            continue
        call = stmt.value
        if call.op.name not in ("tl.ascend_set_flag", "tl.ascend_wait_flag") or call.args[0].value != "MTE2_MTE3":
            continue
        assert not any(isinstance(parent, ir.If) for parent in parents)
        identifier = tvm_tirx.sym.Analyzer().simplify(call.args[1])
        assert isinstance(identifier, ir.IntImm) or isinstance(tvm_tirx.sym.Analyzer().simplify(identifier - i % 2), ir.IntImm)
        dynamic.append(call)
    assert len(dynamic) == 2
    tvm_tirx.ir.assert_structural_equal(dynamic[0].args[1], dynamic[1].args[1])


@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint-rows", "same-row"])
def test_dma_row_writes_only_serialize_overlapping_regions(overlap):
    a = tensor((2, 64), "float32", name="A")
    out = tensor((2, 64), "float32", name="out")
    ub = tensor((2, 64), "float32", name="ub", scope="shared.dyn")
    row = ir.Var("row", "int32")
    load = copy(a, ub, src_indices=[row, 0], dst_indices=[0 if overlap else row, 0], src_shape=[1, 64], dst_shape=[1, 64])
    rows = ir.For(row, ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, unit(load))
    before = kernel(seq(unit(rows, core=None), unit(copy(ub, out))), buffers=[ub], params=[a, out], allocation_stage=0)
    after = transform.InsertSync()(before)
    barriers = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE2"]
    assert bool(barriers) == overlap
    # The whole-buffer consumer must still wait for the row loads to finish.
    releases = _events(after, "set", "MTE2_MTE3")
    waits = _events(after, "wait", "MTE2_MTE3")
    assert len(releases) == len(waits) == 1
    tvm_tirx.ir.assert_structural_equal(releases[0].args[1], waits[0].args[1])


@pytest.mark.parametrize(
    "overlap, declared_rows",
    [(False, 32), (False, 48), (True, 48)],
    ids=["disjoint-halves", "declared-overlap", "overwrite-middle"],
)
def test_partial_l0_loads_complete_before_gemm_and_order_overlapping_writes(overlap, declared_rows):
    src = tensor((64, 64), "bfloat16", name="src", scope="shared.l1")
    a = tensor((64, 64), "bfloat16", name="a", scope="shared.l0a")
    b = tensor((64, 64), "bfloat16", name="b", scope="shared.l0b")
    c = tensor((64, 64), "float32", name="c", scope="shared.l0c")
    # B and the L1 source are already prepared at this pass boundary.
    # Dependencies conservatively cover the declared destination region.
    body = seq(
        unit(
            copy(
                src,
                a,
                src_shape=[48 if overlap else 32, 64],
                dst_shape=[declared_rows, 64],
                annotations={"l0_pitch": ir.IntImm("int32", 4)},
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
                annotations={"l0_pitch": ir.IntImm("int32", 4)},
            ),
            core=2,
        ),
        unit(ir.Evaluate(T.gemm(a, b, c, transpose_B=True, clear_accum=True)), core=2),
    )
    layouts = {src: make_ascend_major_k_layout(src), a: make_ascend_major_k_layout(a)}
    after = transform.InsertSync()(kernel(body, buffers=[src, a, b, c], annotations={"layout_map": layouts}, allocation_stage=0))
    barriers = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE1"]
    assert bool(barriers) == (declared_rows > 32)
    releases, waits = _events(after, "set", "MTE1_M"), _events(after, "wait", "MTE1_M")
    assert releases and waits
    for release in releases:
        assert any(tvm_ffi.structural_equal(release.args[1], wait.args[1]) for wait in waits)
    operations = [stmt.value for stmt in nodes(after, ir.Evaluate) if isinstance(stmt.value, ir.Call)]
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
    src = tensor((64, 64), "bfloat16", name="src", scope="shared.l1")
    a = tensor((2, 64, 64), "bfloat16", name="a", scope="shared.l0b" if transposed else "shared.l0a")
    b = tensor((64, 64), "bfloat16", name="b", scope="shared.l0a" if transposed else "shared.l0b")
    c = tensor((32, 64), "float32", name="c", scope="shared.l0c")
    # GEMM reads rows [0,32) at pitch=2. A subsequent load of [32,64)
    # at pitch=4 overlaps that compact storage, despite logical separation.
    operand_shape = (1, 64, 32) if transposed else (1, 32, 64)
    lhs = ir.TensorRegion(a, [tvm_tirx.ir.Range.from_min_extent(0, e) for e in operand_shape], ir.TensorRegionType())
    if transposed:
        c = tensor((64, 32), "float32", name="c", scope="shared.l0c")
        gemm = T.gemm(b, lhs, c, clear_accum=True)
        start, shape = [k_start, mn_start], [k_size, mn_size]
    else:
        gemm = T.gemm(lhs, b, c, transpose_B=True, clear_accum=True)
        start, shape = [mn_start, k_start], [mn_size, k_size]
    body = seq(
        unit(ir.Evaluate(gemm), core=2),
        unit(
            copy(
                src,
                a,
                src_shape=shape,
                dst_shape=[1, *shape],
                dst_indices=[plane, *start],
                annotations={"l0_pitch": ir.IntImm("int32", 4)},
            ),
            core=2,
        ),
    )
    layouts = {
        src: make_ascend_major_k_layout(src),
        a: (make_ascend_major_mn_layout if transposed else make_ascend_major_k_layout)(a),
    }
    after = transform.InsertSync()(kernel(body, buffers=[src, a, b, c], annotations={"layout_map": layouts}, allocation_stage=0))
    assert bool(_events(after, "set", "M_MTE1")) == overlap
    assert bool(_events(after, "wait", "M_MTE1")) == overlap


@pytest.mark.parametrize("pitch_relation", ["same", "equivalent", "unknown"])
def test_partial_l0_copies_require_provably_equal_pitches(pitch_relation):
    src = tensor((64, 64), "bfloat16", name="src", scope="shared.l1")
    a = tensor((80, 64), "bfloat16", name="a", scope="shared.l0a")
    p, q = ir.Var("p", "int32"), ir.Var("q", "int32")
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
    after = transform.InsertSync()(kernel(body, buffers=[src, a], params=[p, q], annotations={"layout_map": layouts}, allocation_stage=0))
    barriers = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE1"]
    assert bool(barriers) == (pitch_relation == "unknown")


@pytest.mark.parametrize("varying_pitch", [False, True], ids=["disjoint", "overlap-after-pitch-change"])
def test_l0_copy_pitch_is_instantiated_per_iteration(varying_pitch):
    src = tensor((16, 16), "bfloat16", name="src", scope="shared.l1")
    a = tensor((32, 32), "bfloat16", name="a", scope="shared.l0a")
    i = ir.Var("i", "int32")
    # Disjoint logical blocks (m,k)=(0,1),(1,0). With pitches 1,2 they
    # both address physical block 1; at a fixed pitch 2 they address 2,1.
    load = copy(
        src,
        a,
        dst_indices=[16 * i, 16 * (1 - i)],
        dst_shape=[16, 16],
        annotations={"l0_pitch": 1 + i if varying_pitch else ir.IntImm("int32", 2)},
    )
    body = unit(ir.For(i, ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, unit(load, core=2)), core=None)
    layouts = {src: make_ascend_major_k_layout(src), a: make_ascend_major_k_layout(a)}
    after = transform.InsertSync()(kernel(body, buffers=[src, a], annotations={"layout_map": layouts}, allocation_stage=0))
    barriers = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE1"]
    assert bool(barriers) == varying_pitch


@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint", "overlap"])
@pytest.mark.parametrize("unroll", [False, True], ids=["internal-loop", "multiple-calls"])
@pytest.mark.parametrize("varying_pitch", [False, True], ids=["same-pitch", "mixed-pitches"])
def test_atomic_l0_copies_require_a_uniform_pitch(overlap, unroll, varying_pitch):
    src = tensor((16, 16), "bfloat16", name="src", scope="shared.l1")
    a = tensor((32, 48), "bfloat16", name="a", scope="shared.l0a")
    b = tensor((16, 32 if overlap else 16), "bfloat16", name="b", scope="shared.l0b")
    c = tensor((16, 16), "float32", name="c", scope="shared.l0c")
    i = ir.Var("i", "int32")
    load = copy(
        src,
        a,
        dst_indices=[0, 16 * (i + 1)],
        dst_shape=[16, 16],
        annotations={"l0_pitch": 1 + i if varying_pitch else ir.IntImm("int32", 1)},
    )
    lhs = ir.TensorRegion(a, [tvm_tirx.ir.Range.from_min_extent(0, e) for e in (16, 32 if overlap else 16)], ir.TensorRegionType())
    # Mixed pitches conservatively conflict even when the K regions are disjoint.
    loads = (
        seq(*(substitute(load, {i: value}) for value in (0, 1)))
        if unroll
        else ir.For(i, ir.const(0), ir.const(0) + ir.const(2), ir.const(1), ir.ForKind.DEFAULT, load)
    )
    body = seq(
        unit(loads, core=2),
        unit(ir.Evaluate(T.gemm(lhs, b, c, transpose_B=True, clear_accum=True)), core=2),
    )
    layouts = {src: make_ascend_major_k_layout(src), a: make_ascend_major_k_layout(a)}
    after = transform.InsertSync()(kernel(body, buffers=[src, a, b, c], annotations={"layout_map": layouts}, allocation_stage=0))
    assert bool(_events(after, "set", "MTE1_M")) == (overlap or varying_pitch)
    assert bool(_events(after, "wait", "MTE1_M")) == (overlap or varying_pitch)


def test_equivalent_lexical_epoch_serializes_counter_ring():
    a = tensor((64,), "float32", name="A")
    ub = tensor((64,), "float32", name="ub", scope="shared.dyn")
    epoch = tensor((1,), "int32", name="epoch", scope="local.var")
    i = ir.Var("i", "int32")
    guard = i % 2 == 0
    compute = sblock("SIMD_VF", ir.TensorStore(ub, [0], ub[0]))
    body = seq(
        unit(copy(a, ub), guard=guard),
        unit(compute, guard=guard),
        unit(copy(ub, a), guard=guard),
        unit(ir.TensorStore(epoch, [0], epoch[0] + 1), guard=guard),
    )
    owner = ir.For(
        i,
        ir.const(0),
        ir.const(0) + ir.const(4),
        ir.const(1),
        ir.ForKind.DEFAULT,
        body,
        annotations={
            "multi_buffer_eligible": [ub],
            "tl.multi_buffer_counter_map": {ub: epoch},
            "tl.storage_epoch_guard_map": {ub: guard, a: guard},
        },
    )
    before = kernel(
        seq(unit(ir.TensorStore(epoch, [0], 0)), unit(owner, core=None)),
        buffers=[ub, epoch],
        params=[a],
        annotations={"tl.buffer_versions_map": {ub: 2}},
        allocation_stage=0,
    )
    after = transform.InsertSync()(before)
    # GM is reused every active epoch, so its lexical MTE3->MTE2 closure
    # also serializes the equivalent counter domain despite two UB versions.
    for pipe in ("MTE2_V", "V_MTE3", "MTE3_MTE2"):
        released = _events(after, "set", pipe)
        acquired = _events(after, "wait", pipe)
        assert released and acquired
        assert all(isinstance(tvm_tirx.sym.Analyzer().simplify(event.args[1]), ir.IntImm) for event in released + acquired)
        assert sorted(int(tvm_tirx.sym.Analyzer().simplify(event.args[1])) for event in released) == sorted(
            int(tvm_tirx.sym.Analyzer().simplify(event.args[1])) for event in acquired
        )
    root = next(block for block in sblocks(after) if sblock_metadata(block)["name"] == "tilelang_root")
    assert int(sblock_annotations(root)["tl.buffer_versions_map"][ub]) == 2
