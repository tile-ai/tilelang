"""Exact L1 placement survives physical layouts and automatic versions."""

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes


def _allocation_program(buffers, offsets=None, *, aliases=True):
    body = tirx.SeqStmt(
        [
            *[tirx.AllocBuffer(buffer) for buffer in buffers],
            *[tirx.BufferStore(buffer, tirx.const(index, buffer.dtype), [0] * len(buffer.shape)) for index, buffer in enumerate(buffers)],
        ]
    )
    if aliases:
        # Even an explicit positive alias permission cannot authorize overlap.
        body = tirx.AttrStmt(
            {buffer.data: [other.data for other in buffers if other != buffer] for buffer in buffers}, "tl.buffer_alias_map", 1, body
        )
    if offsets is not None:
        body = tirx.AttrStmt(offsets, "tl.ascend_l1_offsets", 1, body)
    thread = tirx.IterVar(None, tirx.Var("bx", "int32"), tirx.IterVar.ThreadIndex, "blockIdx.x")
    return tvm.IRModule({"main": tirx.PrimFunc([], tirx.AttrStmt(thread, "thread_extent", 1, body))})


def _l1(name, size=256, dtype="bfloat16"):
    return tirx.decl_buffer((size,), dtype, name=name, scope="shared.l1")


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_exact_plan_keeps_holes_and_physical_offsets(disable_reuse):
    a, b = _l1("a", 1024), _l1("b", 2048)
    before = _allocation_program([a, b], {a.data: 262144, b.data: 0})
    after = transform.MergeUBAllocations(disable_reuse=disable_reuse)(before)
    stores = nodes(after, tirx.BufferStore)
    assert [int(tvm.arith.Analyzer().simplify(store.indices[0])) * 2 for store in stores] == [262144, 0]
    (allocation,) = nodes(after, tirx.AllocBuffer)
    assert allocation.buffer.dtype == "uint8"
    assert int(allocation.buffer.shape[0]) == 262144 + 2048
    assert not any(node.attr_key == "tl.ascend_l1_offsets" for node in nodes(after, tirx.AttrStmt))


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_single_allocation_honors_nonzero_offset(disable_reuse):
    a = _l1("a")
    after = transform.MergeUBAllocations(disable_reuse=disable_reuse)(_allocation_program([a], {a.data: 262144}))
    (allocation,) = nodes(after, tirx.AllocBuffer)
    assert int(allocation.buffer.shape[0]) == 262144 + 512
    (store,) = nodes(after, tirx.BufferStore)
    assert int(tvm.arith.Analyzer().simplify(store.indices[0])) * 2 == 262144


@pytest.mark.parametrize(
    "case,message",
    [
        ("missing", "missing"),
        ("unknown", "unknown L1 allocation"),
        ("scope", "shared.l1"),
        ("unaligned", "512-byte aligned"),
        ("negative", "512-byte aligned"),
        ("boolean", "integer offsets"),
        ("overlap", "overlap"),
        ("capacity", "512 KiB capacity"),
        ("overflow", "512 KiB capacity"),
        ("dynamic", "positive constant"),
        ("empty", "must not be empty"),
        ("no-contract", "scheduled allocation"),
    ],
)
def test_invalid_physical_plans_fail(case, message):
    a, b = _l1("a"), _l1("b")
    if case == "dynamic":
        a = _l1("a", tirx.Var("n", "int32"))
    elif case == "overflow":
        a = _l1("a", tirx.IntImm("int64", 2**62))
    elif case == "scope":
        a = tirx.decl_buffer((256,), "bfloat16", name="a", scope="shared.dyn")
    offsets = {a.data: 0, b.data: 1024}
    if case == "missing":
        offsets.pop(b.data)
    elif case == "unknown":
        offsets[_l1("unknown").data] = 2048
    elif case in ("unaligned", "negative", "boolean", "capacity"):
        offsets[a.data] = {"unaligned": 1, "negative": -512, "boolean": tirx.const(True, "bool"), "capacity": 524288}[case]
    elif case == "overlap":
        offsets[b.data] = 0
    elif case == "empty":
        offsets = {}
    with pytest.raises(tvm.error.InternalError, match=message):
        transform.MergeUBAllocations()(_allocation_program([a, b], offsets, aliases=case != "no-contract"))


def test_existing_dynamic_l1_arena_is_rejected():
    a = _l1("a")
    arena = tirx.decl_buffer((512,), "uint8", name="arena", scope="shared.l1.dyn")
    with pytest.raises(tvm.error.InternalError, match="existing shared.l1.dyn arena"):
        transform.MergeUBAllocations()(_allocation_program([a, arena], {a.data: 0}))


@pytest.mark.parametrize("offset", [True, -512, 1, 1.5, 524288, 2**64])
def test_frontend_rejects_invalid_offsets(offset):
    with pytest.raises(ValueError, match="L1 byte offset"):
        T.annotate_l1_offsets({_l1("a"): offset})


def test_frontend_aliases_share_one_storage_offset():
    a = _l1("a")
    view = tirx.decl_buffer((512,), "uint8", data=a.data, name="view", scope="shared.l1")
    with pytest.raises(ValueError, match="same byte offset"):
        T.annotate_l1_offsets({a: 0, view: 512})


def _versioned_kernel(b_offset, unlimit=False, rows=16):
    @T.prim_func
    def main(A: T.Tensor((4, rows, 16), "bfloat16"), B: T.Tensor((4, 16, 16), "bfloat16"), C: T.Tensor((4, rows, 16), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((rows, 16), "bfloat16")
            b = T.alloc_l1((16, 16), "bfloat16")
            a_view = T.view(a, shape=(rows, 16))
            aq = T.alloc_l0a((rows, 16), "bfloat16")
            bk = T.alloc_l0b((16, 16), "bfloat16")
            c = T.alloc_l0c((rows, 16), "float32")
            T.annotate_buffer_versions({a: 2, b: 2, aq: 1, bk: 1, c: 1})
            T.annotate_l1_offsets({a: 0, a_view: 0, b: b_offset})
            if unlimit:
                T.annotate_unlimit_memory("shared.l1")
            for i in T.Pipelined(4, num_stages=2, annotations={"enable_offset": True}):
                with T.Stage(0), T.Task():
                    T.copy(A[i, :, :], a)
                    T.copy(B[i, :, :], b)
                with T.Stage(1):
                    T.copy(a_view, aq)
                    T.copy(b, bk)
                    T.gemm(aq, bk, c, transpose_B=True, clear_accum=True)
                    T.copy(c, C[i, :, :])

    return main


def test_pipeline_preserves_offsets_across_versions_layout_remap_and_flatten():
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.MaterializeMultiBuffer", "tl.LowerScheduledTIR", "tl.AscendLowerTileOp", "tl.MergeUBAllocations"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(_versioned_kernel(262144), target="ascend")
    physical = snapshots["tl.MaterializeMultiBuffer"]
    buffers = [buffer for block in nodes(physical, tirx.SBlock) for buffer in block.alloc_buffers if buffer.scope() == "shared.l1"]
    assert len(buffers) == 2 and all(int(buffer.shape[0]) == 2 for buffer in buffers)
    for stage in ("tl.LowerScheduledTIR", "tl.AscendLowerTileOp"):
        mod = snapshots[stage]
        (placement,) = [node for node in nodes(mod, tirx.AttrStmt) if node.attr_key == "tl.ascend_l1_offsets"]
        storages = [buffer.data for block in nodes(mod, tirx.SBlock) for buffer in block.alloc_buffers if buffer.scope() == "shared.l1"]
        assert len(placement.node) == 2
        assert all(any(key.same_as(storage) for storage in storages) for key in placement.node)
        assert sorted(int(offset) for offset in placement.node.values()) == [0, 262144]
    merged = snapshots["tl.MergeUBAllocations"]
    allocations = [node for func in merged.functions.values() for node in nodes(func, tirx.AllocBuffer)]
    (allocation,) = [node for node in allocations if node.buffer.scope() == "shared.l1.dyn"]
    assert int(allocation.buffer.shape[0]) == 262144 + 1024
    assert not any(node.attr_key == "tl.ascend_l1_offsets" for func in merged.functions.values() for node in nodes(func, tirx.AttrStmt))
    assert artifact.kernel_source and artifact.device_mod


def test_overlap_is_checked_after_automatic_versions_expand():
    # Each logical tile is only 512 bytes, but two physical versions occupy 1024.
    with pytest.raises(tvm.error.InternalError, match="L1 placements overlap"):
        tilelang.lower(_versioned_kernel(512), target="ascend")


def test_fractal_padding_is_included_in_the_physical_footprint():
    # Two 17x16 BF16 logical tiles take 1088 bytes, but padded 32x16 tiles
    # take 2048. An offset of 1536 therefore overlaps only after padding.
    with pytest.raises(tvm.error.InternalError, match="L1 placements overlap"):
        tilelang.lower(_versioned_kernel(1536, rows=17), target="ascend")


@pytest.mark.parametrize("unlimit", [False, True])
def test_holes_and_versions_cannot_exceed_physical_capacity(unlimit):
    # The original tile fits; the final second version crosses the L1 boundary.
    with pytest.raises(tvm.error.InternalError, match="512 KiB capacity"):
        tilelang.lower(_versioned_kernel(524288 - 512, unlimit), target="ascend")


def test_unscheduled_pipeline_rejects_placement_instead_of_dropping_it():
    with (
        pytest.raises(tvm.error.InternalError, match="requires the Ascend scheduled pipeline"),
        tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False}),
    ):
        tilelang.lower(_versioned_kernel(262144), target="ascend")


def test_absent_placement_preserves_existing_no_contract_behavior():
    before = _allocation_program([_l1("a")], aliases=False)
    tvm.ir.assert_structural_equal(before, transform.MergeUBAllocations()(before))
