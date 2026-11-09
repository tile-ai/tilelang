"""Buffer reuse must preserve the original synchronization access contracts."""

import pytest

import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.layout import make_ascend_nz_layout
from testing.ascend._ir import calls, copy, kernel, nodes, schedule, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from tvm import tirx


def _events(program, disable_reuse=False, auto_schedule=False):
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": disable_reuse}):
        before = schedule(program, through="ResolveCore", auto_schedule=auto_schedule)
        after = transform.InsertSync()(before)
        return {
            (operation, call.args[0].value) for operation in ("set", "wait") for call in calls(after, "tl.ascend_" + operation + "_flag")
        }


@pytest.mark.parametrize("auto_schedule", [False, True], ids=["sync", "schedule-and-sync"])
@pytest.mark.parametrize("layout", ["cross_row", "short_cross_row", "dense_cross_row", "transposed"])
def test_contiguous_pointer_load_keeps_dma_dependency(layout, auto_schedule):
    width = 256 if layout == "short_cross_row" else 128
    transposed = layout == "transposed"

    @T.prim_func
    def main(A: T.Tensor((64,), "float16"), Out: T.Tensor((128,), "float16")):
        with T.Kernel(1):
            work = T.alloc_shared((2, width), "float16")
            out = T.alloc_shared((128,), "float16")
            if transposed:
                view = T.view(work, shape=(width, 2), strides=(1, width))
                T.copy(A, view[:64, 1])
                with T.SimdVF():
                    T.simd.vsts(out[0], T.simd.vld(view[width - 64, 0]), dist="NORM_B16")
            elif layout == "dense_cross_row":
                view = T.view(work, shape=(2, width), strides=(width, 1))
                T.copy(A, view[1, :64])
                with T.SimdVF():
                    T.simd.vsts(out[0], T.simd.vld(view[0, width - 64]), dist="NORM_B16")
            else:
                T.copy(A, work[1, :64])
                with T.SimdVF():
                    T.simd.vsts(out[0], T.simd.vld(work[0, width - 64]), dist="NORM_B16")
            T.copy(out, Out)

    # IR-only: the DMA initializes just the overlapping half of the vector.
    after = schedule(main, through="InsertSync", auto_schedule=auto_schedule)
    trace = []
    events = []
    for call in nodes(after, tirx.Call):
        if call.op.name == "tl.tileop.ascend_copy":
            trace.append("copy")
        elif call.op.name == "tl.simd.vld":
            trace.append("load")
        elif call.op.name in ("tl.ascend_set_flag", "tl.ascend_wait_flag") and call.args[0].value == "MTE2_V":
            trace.append("set" if call.op.name == "tl.ascend_set_flag" else "wait")
            events.append(call)
    assert trace == ["copy", "set", "wait", "load", "copy"]
    tvm.ir.assert_structural_equal(events[0].args[1], events[1].args[1])


@pytest.mark.parametrize("row", [0, 1], ids=["overlap", "adjacent-slot"])
def test_dense_pointer_view_preserves_slot_dependency(row):
    @T.prim_func
    def main(A: T.Tensor((128,), "bfloat16"), Out: T.Tensor((128,), "bfloat16")):
        with T.Kernel(1):
            work = T.alloc_shared((2, 64), "float32")
            view = T.view(work, (2, 128), "bfloat16")
            out = T.alloc_shared((128,), "bfloat16")
            with T.Stage(0):
                T.copy(A, view[0, :])
                with T.SimdVF():
                    T.simd.vsts(out[0], T.simd.vld(view[0, 0]), dist="NORM_B16")
                T.copy(A, view[row, :])
                T.copy(out, Out)

    events = _events(main)
    assert (("set", "V_MTE2") in events) == (row == 0)
    assert (("wait", "V_MTE2") in events) == (row == 0)


@pytest.mark.parametrize("layout", ["dense-view", "implicit", "flat"])
@pytest.mark.parametrize("access", ["gather", "gatherb", "scatter", "vld2-offset", "vld2", "contiguous"])
def test_indirect_pointer_access_orders_dma_overwrite(layout, access):
    shape = (512,) if layout == "flat" else (2, 256)
    base = (0,) if layout == "flat" else (0, 0)

    @T.prim_func
    def main(A: T.Tensor(shape, "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            work = T.alloc_shared(shape, "float32")
            if layout == "dense-view":
                view = T.view(work, shape=(2, 256), strides=(256, 1))
            else:
                view = work
            with T.Stage(0):
                T.copy(A, view)
                with T.SimdVF():
                    if access == "gather":
                        T.evaluate(T.simd.vgather2(view[base], T.simd.vdup(T.uint32(256), "uint32")))
                    elif access == "gatherb":
                        T.evaluate(T.simd.vgatherb(view[base], T.simd.vdup(T.uint32(1024), "uint32")))
                    elif access == "scatter":
                        T.simd.vscatter(T.simd.vdup(T.float32(1), "float32"), view[base], T.simd.vdup(T.uint32(256), "uint32"))
                    elif access == "vld2-offset":
                        first, second = T.simd.vld2(view[base], dist="DINTLV_B32", off=T.int32(256))
                        T.evaluate(first)
                        T.evaluate(second)
                    elif access == "vld2":
                        first, second = T.simd.vld2(view[base], dist="DINTLV_B32")
                        T.evaluate(first)
                        T.evaluate(second)
                    else:
                        T.evaluate(T.simd.vld(view[base]))
                if layout == "flat":
                    T.copy(B, view[256:])
                else:
                    T.copy(B, view[1, :])

    after = schedule(main, through="InsertSync", auto_schedule=False)
    trace = []
    events = []
    for call in nodes(after, tirx.Call):
        if call.op.name == "tl.tileop.ascend_copy":
            trace.append("copy")
        elif call.op.name in ("tl.simd.vgather2", "tl.simd.vgatherb", "tl.simd.vscatter", "tl.simd.vld2", "tl.simd.vld"):
            trace.append("access")
        elif call.op.name in ("tl.ascend_set_flag", "tl.ascend_wait_flag") and call.args[0].value == "V_MTE2":
            trace.append("set" if call.op.name == "tl.ascend_set_flag" else "wait")
            events.append(call)
    if access in ("contiguous", "vld2"):
        assert trace == ["copy", "access", "copy"]
    else:
        assert trace == ["copy", "access", "set", "wait", "copy"]
        tvm.ir.assert_structural_equal(events[0].args[1], events[1].args[1])


@pytest.mark.parametrize("fill_first", [False, True], ids=["protect-source-reuse", "wait-for-source-padding"])
@pytest.mark.parametrize("row", [24, 32], ids=["same-fractal", "next-fractal"])
@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
def test_l0_tail_load_accounts_for_source_fractal_padding(fill_first, row, scope):
    src = tirx.decl_buffer((48, 16), "bfloat16", name="src", scope="shared.l1")
    dst = tirx.decl_buffer((32, 16), "bfloat16", name="dst", scope=scope)
    load = unit(copy(src, dst, src_shape=[24, 16], dst_shape=[24, 16]), core=2)
    padding = tirx.BufferRegion(src, [tvm.ir.Range.from_min_extent(row, 8), tvm.ir.Range(0, 16)])
    fill = unit(tirx.Evaluate(T.fill(padding, 0)), core=2)
    # The copy normalizer exposes its two transferred row16 blocks. Fill
    # keeps the explicitly requested eight rows.
    before = kernel(
        seq(fill, load) if fill_first else seq(load, fill),
        buffers=[src, dst],
        annotations={"layout_map": {src: make_ascend_nz_layout(src), dst: make_ascend_nz_layout(dst)}},
    )
    before = transform.NormalizeAscendFractalStorage()(before)
    after = transform.InsertSync()(before)
    pipe = "MTE2_MTE1" if fill_first else "MTE1_MTE2"
    events = [call for op in ("set", "wait") for call in calls(after, "tl.ascend_" + op + "_flag") if call.args[0].value == pipe]
    if row == 24:
        assert len(events) == 2
        tvm.ir.assert_structural_equal(events[0].args[1], events[1].args[1])
    else:
        assert not events


def test_l0_load_matches_explicit_transfer_region_conflicts():
    @T.prim_func
    def main():
        with T.Kernel(1):
            left = T.alloc_l1((32, 16), "bfloat16")
            right = T.alloc_l1((32, 16), "bfloat16")
            dst = T.alloc_l0a((32, 16), "bfloat16")
            T.assume_conflict(left[:24, :], right[:32, :], cross=False)
            with T.Stage(0):
                T.fill(left[:24, :], 0)
                # The source is already prepared at this sync-only boundary.
                T.copy(right[:32, :], dst[:32, :])

    events = _events(main)
    assert ("set", "MTE2_MTE1") in events
    assert ("wait", "MTE2_MTE1") in events


@pytest.mark.parametrize("fill_first", [False, True])
@pytest.mark.parametrize("row", [16, 24], ids=["overlap", "adjacent"])
def test_declared_l1_pointer_read_keeps_its_own_footprint(fill_first, row):
    src = tirx.decl_buffer((32, 16), "bfloat16", name="src", scope="shared.l1")
    # A pointer consumer declares 24 rows. Unlike the L1->L0 load above,
    # this access contract does not imply a whole-fractal read of 32 rows.
    read = unit(tirx.Evaluate(tirx.call_extern("int32", "read_l1", T.access_ptr(src[0, 0], "r", 24, 16))), core=2)
    region = tirx.BufferRegion(src, [tvm.ir.Range.from_min_extent(row, 8), tvm.ir.Range(0, 16)])
    fill = unit(tirx.Evaluate(T.fill(region, 0)), core=2)
    before = kernel(seq(fill, read) if fill_first else seq(read, fill), buffers=[src])
    after = transform.InsertSync()(before)
    pipe = "MTE2_S" if fill_first else "S_MTE2"
    events = [call for op in ("set", "wait") for call in calls(after, "tl.ascend_" + op + "_flag") if call.args[0].value == pipe]
    assert len(events) == (2 if row == 16 else 0)
    if events:
        tvm.ir.assert_structural_equal(events[0].args[1], events[1].args[1])


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_pointer_store_past_view_keeps_scalar_read_dependency(disable_reuse, strided):
    @T.prim_func
    def main(out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "float32")
            if strided:
                view = T.StridedTensor((80,), (2,), "float32", data=allocation.data, scope="shared.dyn")
            else:
                view = tvm.tirx.decl_buffer((96,), "float32", data=allocation.data, scope="shared.dyn")
            with T.Stage(0):
                with T.SimdVF():
                    T.simd.vsts(view[64], T.simd.vdup(T.float32(1), "float32"))
                out[0] = view[88 if strided else 96]

    events = _events(main, disable_reuse)
    assert ("set", "V_S") in events
    assert ("wait", "V_S") in events


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("kind", ["dense_view", "strided", "padded_copy"])
@pytest.mark.parametrize("hint", ["region", "storage", "region_overrides_storage"])
def test_expanded_footprint_preserves_explicit_conflict(kind, disable_reuse, hint):
    @T.prim_func
    def main(A: T.Tensor((30,), "float32"), C: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "float32")
            consumer = T.alloc_shared((64,), "float32")
            view = T.StridedTensor((128,), (2 if kind == "strided" else 1,), "float32", data=allocation.data, scope="shared.dyn")
            # Initialize independently of the producer's pipe so this does not
            # provide the ordering asserted below.
            consumer[0] = T.float32(2)
            if kind != "padded_copy":
                if hint == "region_overrides_storage":
                    T.assume_no_conflict(view, consumer, cross=False)
                if hint == "storage":
                    T.assume_conflict(view, consumer, cross=False)
                else:
                    T.assume_conflict(view[:64], consumer[:64], cross=False)
                with T.SimdVF():
                    T.simd.vsts(view[0], T.simd.vdup(T.float32(1), "float32"))
            else:
                if hint == "region_overrides_storage":
                    T.assume_no_conflict(allocation, consumer, cross=False)
                if hint == "storage":
                    T.assume_conflict(allocation, consumer, cross=False)
                else:
                    T.assume_conflict(allocation[:30], consumer[:64], cross=False)
                T.copy(A, allocation[:30], pad_value=0.0)
            T.copy(consumer, C)

    events = _events(main, disable_reuse)
    producer_pipe = "MTE2" if kind == "padded_copy" else "V"
    assert ("set", f"{producer_pipe}_MTE3") in events
    assert ("wait", f"{producer_pipe}_MTE3") in events


@pytest.mark.parametrize("hint", ["none", "logical_region", "storage"])
def test_no_conflict_keeps_original_logical_region_matching(hint):
    @T.prim_func
    def main(out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "float32")
            view = T.StridedTensor((80,), (2,), "float32", data=allocation.data, scope="shared.dyn")
            if hint == "logical_region":
                T.assume_no_conflict(view[64:128], view[88:89], cross=False)
            elif hint == "storage":
                # An explicit storage-wide assertion keeps its existing scope.
                T.assume_no_conflict(view, cross=False)
            with T.Stage(0):
                with T.SimdVF():
                    T.simd.vsts(view[64], T.simd.vdup(T.float32(1), "float32"))
                out[0] = view[88]

    events = _events(main)
    # Physical opacity only blocks reuse; it does not change hint matching.
    assert (("set", "V_S") in events) == (hint == "none")


def test_unknown_pointer_read_does_not_create_read_read_dependency():
    @T.prim_func
    def main(out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "float32")
            result = T.alloc_shared((64,), "float32")
            view = T.StridedTensor((80,), (2,), "float32", data=allocation.data, scope="shared.dyn")
            with T.Stage(0):
                view[88] = T.float32(1)
                with T.SimdVF():
                    T.simd.vsts(result[0], T.simd.vld(view[64]))
                out[0] = view[88]

    events = _events(main)
    assert ("set", "S_V") in events
    assert ("set", "V_S") not in events


@pytest.mark.parametrize("versions", [1, 2])
@pytest.mark.parametrize("cross", [False, True])
def test_padded_source_hint_survives_loop_scheduling(versions, cross):
    @T.prim_func
    def main(A: T.Tensor((4, 30), "float32"), C: T.Tensor((4, 64), "float32")):
        with T.Kernel(1):
            producer = T.alloc_shared((32,), "float32")
            consumer = T.alloc_shared((64,), "float32")
            consumer[0] = T.float32(1)
            for i in T.Pipelined(4, num_stages=versions):
                T.assume_conflict(producer[:30], consumer[:64], cross=cross)
                T.copy(A[i, :], producer[:30], pad_value=0.0)
                T.copy(consumer, C[i, :])

    events = _events(main, auto_schedule=True)
    assert ("set", "MTE2_MTE3") in events
    assert ("wait", "MTE2_MTE3") in events


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("control", ["condition", "loop_extent"])
def test_physical_pointer_write_orders_control_expression_reads(control, disable_reuse):
    @T.prim_func
    def main(out: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "int32")
            view = tvm.tirx.decl_buffer((96,), "int32", data=allocation.data, scope="shared.dyn")
            with T.Stage(0):
                with T.SimdVF():
                    T.simd.vsts(view[64], T.simd.vdup(T.int32(1), "int32"))
                if control == "condition":
                    if view[96] > 0:
                        out[0] = 1
                else:
                    for _i in T.serial(view[96]):
                        out[0] = 1

    events = _events(main, disable_reuse)
    assert ("set", "V_S") in events
    assert ("wait", "V_S") in events
