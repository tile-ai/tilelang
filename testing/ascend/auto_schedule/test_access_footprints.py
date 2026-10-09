"""Access footprints contracts."""

import pytest
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.language.utils import region
from testing.ascend.auto_schedule._reuse_utils import _insert_sync_alias_pairs, _make_cyclic_scalar_program


def _make_internal_conditional_write_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_a: T.Tensor((8,), "int32"),
        out_b: T.Tensor((8,), "int32"),
        cond: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                out_a[k] = a[0]
                b[0] = y[k]
                out_b[k] = b[0]
                with T.Task():
                    if cond > 0:
                        a[0] = x[k]

    return main


def _make_predicated_buffer_store_program():
    main = _make_internal_conditional_write_program()

    # Re-express the conditional store using TIRX's equivalent BufferStore
    # predicate field. Imported TIRX and custom transforms may use this form
    # even though ordinary TileLang assignments use IfThenElse.
    def replace_conditional_store(node):
        if not isinstance(node, tvm.tirx.IfThenElse) or node.else_case:
            return None
        store = node.then_case
        if not isinstance(store, tvm.tirx.BufferStore) or store.buffer.name != "a":
            return None
        return tvm.tirx.BufferStore(store.buffer, store.value, store.indices, node.condition)

    body = tvm.tirx.stmt_functor.ir_transform(
        main.body,
        None,
        replace_conditional_store,
    )
    return main.with_body(body)


def _make_disjoint_task_writes_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_a: T.Tensor((8,), "int32"),
        out_b: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((3,), "int32")
            b = T.alloc_shared((3,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                a[1] = x[k]
                b[1] = y[k]
                out_b[k] = b[1]
                with T.Task():
                    a[0] = x[k]
                    a[2] = x[k]
                out_a[k] = a[1]

    return main


def _make_strided_task_write_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_a: T.Tensor((8,), "int32"),
        out_b: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((3,), "int32")
            b = T.alloc_shared((3,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                a[1] = x[k]
                b[1] = y[k]
                out_b[k] = b[1]
                with T.Task():
                    for i in T.serial(2):
                        a[2 * i] = x[k]
                out_a[k] = a[1]

    return main


def _make_raw_stepped_task_write_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_a: T.Tensor((8,), "int32"),
        out_b: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((4,), "int32")
            b = T.alloc_shared((4,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                a[1] = x[k]
                b[1] = y[k]
                out_b[k] = b[1]
                with T.Task():
                    for i in T.serial(4):
                        a[i] = x[k]
                out_a[k] = a[1]

    # The eager frontend normalizes a stepped loop to a unit-step loop plus an
    # affine binding. Rebuild this loop directly to cover imported/raw TIRX,
    # where ForNode::step remains non-unit.
    def add_raw_step(node):
        if isinstance(node, tvm.tirx.For) and node.loop_var.name == "i":
            return tvm.tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                node.annotations,
                tvm.tirx.IntImm(node.loop_var.dtype, 2),
            )
        return None

    body = tvm.tirx.stmt_functor.ir_transform(main.body, None, add_raw_step)
    return main.with_body(body)


def _make_loop_break_task_write_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_a: T.Tensor((8,), "int32"),
        out_b: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((4,), "int32")
            b = T.alloc_shared((4,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                out_a[k] = a[3]
                b[3] = y[k]
                out_b[k] = b[3]
                with T.Task():
                    for i in T.serial(4):
                        a[i] = x[k]
                        if i == 1:
                            T.loop_break()

    return main


def _make_loop_break_before_writer_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_a: T.Tensor((4,), "int32"),
        out_b: T.Tensor((8,), "int32"),
        stop: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((1,), "int32")
            b = T.alloc_shared((1,), "int32")
            for k in T.Pipelined(4, num_stages=1):
                for i in T.serial(2):
                    offset = k * 2 + i
                    b[0] = y[offset]
                    out_b[offset] = b[0]
                    if i >= stop:
                        T.loop_break()
                    a[0] = x[offset]
                out_a[k] = a[0]

    return main


def _make_diagonal_task_write_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_a: T.Tensor((8,), "int32"),
        out_b: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((2, 2), "int32")
            b = T.alloc_shared((2, 2), "int32")
            for k in T.Pipelined(8, num_stages=1):
                a[0, 1] = x[k]
                b[0, 1] = y[k]
                out_b[k] = b[0, 1]
                with T.Task():
                    for i in T.serial(2):
                        a[i, i] = x[k]
                out_a[k] = a[0, 1]

    return main


def _make_predicated_access_ptr_write_program():
    tile = 64
    iterations = 8

    @T.prim_func
    def main(
        y: T.Tensor((iterations, tile), "int32"),
        out_a: T.Tensor((iterations,), "int32"),
        out_b: T.Tensor((iterations, tile), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((tile,), "int32")
            b = T.alloc_shared((tile,), "int32")
            for k in T.Pipelined(iterations, num_stages=1):
                if k == 0:
                    a[tile - 1] = 7
                out_a[k] = a[tile - 1]
                T.copy(y[k, :], b)
                T.copy(b, out_b[k, :])
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    value = T.simd.vdup(k, "int32", one)
                    # The access_ptr spans one vector, but the predicate writes
                    # only lane zero. It is a may-write, not a full definition.
                    T.simd.vsts(a[0], value, one)

    return main


def _make_maybe_empty_single_trip_program():
    @T.prim_func
    def main(
        x: T.Tensor((1,), "int32"),
        y: T.Tensor((1,), "int32"),
        z: T.Tensor((1,), "int32"),
        out_a: T.Tensor((1,), "int32"),
        out_b: T.Tensor((1,), "int32"),
        trip_count: T.int32,
    ):
        with T.Kernel(1):
            T.assume(trip_count >= 0)
            T.assume(trip_count <= 1)
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            a[0] = x[0]
            b[0] = y[0]
            out_b[0] = b[0]
            for _ in T.serial(trip_count):
                a[0] = z[0]
            out_a[0] = a[0]

    return main


def _make_mixed_opaque_access_program():
    @T.prim_func
    def main(
        x: T.Tensor((1,), "int32"),
        y: T.Tensor((1,), "int32"),
        z: T.Tensor((1,), "int32"),
        out_a: T.Tensor((1,), "int32"),
        out_b: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            a[0] = x[0]
            b[0] = y[0]
            out_b[0] = b[0]
            with T.Task():
                T.evaluate(tvm.tirx.call_extern("int32", "consume", a.access_ptr("r")))
                a[0] = z[0]
            out_a[0] = a[0]

    return main


def _make_address_of_escape_program():
    @T.prim_func
    def main(
        x: T.Tensor((2,), "int32"),
        y: T.Tensor((1,), "int32"),
        out_a: T.Tensor((1,), "int32"),
        out_b: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            a[1] = x[0]
            b[1] = y[0]
            out_b[0] = b[1]
            a[0] = x[1]
            out_a[0] = T.call_extern("int32", "read_second", T.address_of(a[0])) + a[0]

    return main


def _make_unknown_write_index_program(index_kind: str):
    @T.prim_func
    def main(
        x: T.Tensor((2,), "int32"),
        y: T.Tensor((1,), "int32"),
        indices: T.Tensor((1,), "int32"),
        out_a: T.Tensor((1,), "int32"),
        out_b: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            a[1] = x[0]
            b[1] = y[0]
            out_b[0] = b[1]
            if index_kind == "extern":
                a[T.call_extern("int32", "select_index", indices[0])] = x[1]
            elif index_kind == "wrapped":
                a[T.call_extern("int32", "select_index", indices[0]) % 8] = x[1]
            else:
                a[indices[0]] = x[1]
            out_a[0] = a[1]

    return main


def _make_cross_row_access_ptr_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_b: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((4, 8), "int32")
            b = T.alloc_shared((4, 8), "int32")
            for k in T.Pipelined(8, num_stages=1):
                b[0, 0] = y[k]
                out_b[k] = b[0, 0]
                with T.Task():
                    for j in T.serial(8):
                        a[1, j] = x[k]
                with T.Task():
                    T.evaluate(
                        tvm.tirx.call_extern(
                            "int32",
                            "consume",
                            T.access_ptr(a[1, 4], "r", extent=8),
                        )
                    )
                with T.Task():
                    for j in T.serial(8):
                        a[2, j] = x[k]

    return main


def test_address_of_escape_keeps_unknown_read_footprint_live():
    # read_second dereferences p[1], outside the base element named by
    # address_of. Rewriting a[0] cannot kill that still-live value of a[1].
    program = _make_address_of_escape_program()
    assert ("a", "b") not in _insert_sync_alias_pairs(program)


@pytest.mark.parametrize("index_kind", ["extern", "load", "wrapped"])
def test_unknown_write_index_does_not_kill_other_elements(index_kind):
    # With an index value of zero, the second store leaves a[1] live. Its
    # may-write hull must not become a definition of every element in a.
    program = _make_unknown_write_index_program(index_kind)
    assert ("a", "b") not in _insert_sync_alias_pairs(program)


@pytest.mark.parametrize("write_kind", ["region", "reduce", "padded_copy"])
def test_unknown_region_write_does_not_kill_other_elements(write_kind):
    @T.prim_func
    def main(
        x: T.Tensor((1, 64), "float32"), y: T.Tensor((1,), "float32"), indices: T.Tensor((1,), "int32"), out: T.Tensor((2,), "float32")
    ):
        with T.Kernel(1):
            a = T.alloc_shared((16,), "float32")
            b = T.alloc_shared((16,), "float32")
            src = T.alloc_shared((1, 64), "float32")
            index_ub = T.alloc_shared((1,), "int32")
            T.copy(x, src)
            index_ub[0] = indices[0]
            with T.Stage(0):
                a[7] = x[0, 0]
                b[7] = y[0]
                out[1] = b[7]
                if write_kind == "region":
                    with T.Task():
                        T.evaluate(
                            T.call_extern("int32", "write_value", region(a[T.call_extern("int32", "select_index", indices[0])], "w", 1))
                        )
                elif write_kind == "reduce":
                    with T.SimdVF():
                        T.reduce_sum(
                            src,
                            tvm.tirx.BufferRegion(
                                a, [tvm.ir.Range.from_min_extent(T.call_extern("int32", "select_index", index_ub[0]), 1)]
                            ),
                            dim=1,
                        )
                else:
                    T.copy(
                        x[0, :7],
                        tvm.tirx.BufferRegion(a, [tvm.ir.Range.from_min_extent(T.call_extern("int32", "select_index", indices[0]), 7)]),
                        pad_value=0.0,
                    )
                out[0] = a[7]

    # These are access-analysis inputs. An opaque output offset need not be
    # supported by later vector/DMA lowering to require conservative coverage.
    assert ("a", "b") not in _insert_sync_alias_pairs(main)


@pytest.mark.parametrize("mutable_index", [False, True])
def test_coverage_distinguishes_mutable_index_snapshots(mutable_index):
    @T.prim_func
    def main(x: T.Tensor((2,), "int32"), y: T.Tensor((1,), "int32"), out: T.Tensor((2,), "int32"), slot: T.int32):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            indices = T.alloc_shared((1,), "int32")
            with T.Stage(0):
                a[0] = x[0]
                b[0] = y[0]
                out[1] = b[0]
                if mutable_index:
                    indices[0] = slot
                    a[indices[0]] = x[1]
                    indices[0] = 0
                    out[0] = a[indices[0]]
                else:
                    a[slot] = x[1]
                    out[0] = a[slot]

    pairs = _insert_sync_alias_pairs(main)
    if mutable_index:
        # With slot=1, the later reader still needs the original a[0]. Equal
        # BufferLoad syntax at two different access points is not equal value.
        assert ("a", "b") not in pairs
    else:
        # A genuine immutable coordinate remains a valid coverage proof.
        assert ("a", "b") in pairs


@pytest.mark.parametrize("pointer_kind", ["strided", "beyond_view", "dynamic", "unknown_base", "single", "compact", "dense_view"])
def test_pointer_footprint_must_fit_the_logical_view(pointer_kind):
    view_size = 1 if pointer_kind == "beyond_view" else 8
    old_index = 8 if pointer_kind in ("dynamic", "unknown_base") else 1

    @T.prim_func
    def main(out: T.Tensor((2,), "int32"), count: T.int32):
        with T.Kernel(1):
            a = T.alloc_shared((16,), "int32")
            b = T.alloc_shared((16,), "int32")
            if pointer_kind in ("strided", "single"):
                view = T.StridedTensor((view_size,), (2,), "int32", data=a.data, scope="shared.dyn")
            elif pointer_kind == "dense_view":
                view = T.view(a)
            else:
                view = tvm.tirx.decl_buffer((view_size,), "int32", data=a.data, scope="shared.dyn")
            with T.Stage(0):
                a[old_index] = 100
                b[old_index] = 200
                out[1] = b[old_index]
                with T.Task():
                    for i in T.serial(view_size):
                        view[i] = 300
                if pointer_kind == "dynamic":
                    out[0] = T.call_extern("int32", "consume_view", T.access_ptr(view[0], "r", extent=count))
                elif pointer_kind == "unknown_base":
                    out[0] = T.call_extern(
                        "int32", "consume_view", T.access_ptr(view[T.call_extern("int32", "select_index", count)], "r", extent=1)
                    )
                else:
                    out[0] = T.call_extern("int32", "consume_view", T.access_ptr(view[0], "r", extent=1 if pointer_kind == "single" else 2))

    pairs = _insert_sync_alias_pairs(main)
    if pointer_kind in ("single", "compact"):
        assert ("b", "view") in pairs
    else:
        assert ("b", "view") not in pairs


@pytest.mark.parametrize("kind", ["gather", "gatherb", "vld2_offset", "vld2_compact", "compact"])
def test_indirect_intrinsic_read_footprint_is_not_a_compact_region(kind):
    @T.prim_func
    def main(out: T.Tensor((2,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((512,), "float32")
            b = T.alloc_shared((512,), "float32")
            with T.Stage(0):
                with T.Task():
                    for i in T.serial(512):
                        a[i] = T.float32(100)
                b[64] = T.float32(200)
                out[1] = b[64]
                with T.Task():
                    for i in T.serial(128 if kind.startswith("vld2") else 64):
                        a[i] = T.float32(300)
                with T.SimdVF():
                    if kind == "gather":
                        T.evaluate(T.simd.vgather2(a[0], T.simd.vdup(T.uint32(256), "uint32")))
                    elif kind == "gatherb":
                        T.evaluate(T.simd.vgatherb(a[0], T.simd.vdup(T.uint32(256), "uint32")))
                    elif kind == "vld2_offset":
                        first, second = T.simd.vld2(a[0], dist="DINTLV_B32", off=T.int32(256))
                        T.evaluate(first)
                        T.evaluate(second)
                    elif kind == "vld2_compact":
                        first, second = T.simd.vld2(a[0], dist="DINTLV_B32")
                        T.evaluate(first)
                        T.evaluate(second)
                    else:
                        T.evaluate(T.simd.vld(a[0]))

    pairs = _insert_sync_alias_pairs(main)
    if kind in ("compact", "vld2_compact"):
        assert ("a", "b") in pairs
    else:
        assert ("a", "b") not in pairs


@pytest.mark.parametrize("pointer_kind", ["raw", "carrier", "returned"])
def test_raw_storage_pointer_escape_is_not_hidden_by_a_structured_access(pointer_kind):
    @T.prim_func
    def main(out: T.Tensor((2,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            with T.Stage(0):
                a[1] = 100
                b[1] = 200
                out[1] = b[1]
                a[0] = 300
                with T.Task():
                    if pointer_kind == "carrier":
                        pointer = T.alloc_var("handle", init=T.access_ptr(a[0], "r", extent=1))
                        out[0] = T.call_extern("int32", "read_second", pointer) + a[0]
                    elif pointer_kind == "returned":
                        out[0] = (
                            T.call_extern("int32", "read_value", T.call_extern("handle", "advance", T.access_ptr(a[0], "r", extent=1)))
                            + a[0]
                        )
                    else:
                        out[0] = T.call_extern("int32", "read_second", a.data) + a[0]

    assert ("a", "b") not in _insert_sync_alias_pairs(main)


def test_nd2nz_post_copy_uses_its_dma_read_span():
    @T.prim_func
    def main(out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((32,), "float32")
            b = T.alloc_shared((32,), "float32")
            dst = T.alloc_l1((16,), "float32")
            with T.Stage(0):
                with T.Task():
                    for i in T.serial(32):
                        a[i] = T.float32(100)
                b[16] = T.float32(200)
                out[0] = b[16]
                a[0] = T.float32(300)
                T.ascend_nd2nz_post_copy(dst[0], a[0], 1, 16, 1, "float")

    assert ("a", "b") not in _insert_sync_alias_pairs(main)


def test_nd2nz_statement_intrinsics_keep_source_alias_eligibility():
    from tilelang.layout import make_ascend_nz_layout

    @T.prim_func
    def main(x: T.Tensor((16, 64), "float32"), out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((16, 64), "float32")
            other = T.alloc_shared((16, 64), "float32")
            dst = T.alloc_l1((16, 64), "bfloat16")
            T.annotate_layout({dst: make_ascend_nz_layout(dst)})
            with T.Stage(0):
                T.copy(x, src)
                T.copy(src, dst)
                with T.SimdVF():
                    T.simd.vsts(other[0, 0], T.simd.vdup(T.float32(200), "float32"))
                out[0] = other[0, 0]

    mod = tvm.IRModule({"main": main.with_attr("target", tvm.target.Target("ascend"))})
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    with tvm.target.Target("ascend"):
        mod = tilelang.ascend.transform.AscendLayoutInference()(mod)
        mod = tilelang.ascend.transform.NormalizeAscendFractalStorage()(mod)
        mod = tilelang.ascend.transform.InsertNd2Nz()(mod)
    assert ("dst", "src") in _insert_sync_alias_pairs(mod["main"])


def test_nd2nz_scatter_source_can_alias_a_later_ub_copy_destination():
    from tilelang.layout import make_ascend_compact_nz_layout

    @T.prim_func
    def main(x: T.Tensor((16, 64), "float32"), out: T.Tensor((1,), "bfloat16")):
        with T.Kernel(1):
            src = T.alloc_shared((16, 64), "float32")
            nz = T.alloc_shared((17, 64), "bfloat16")
            other = T.alloc_shared((17, 64), "bfloat16")
            T.annotate_layout({nz: make_ascend_compact_nz_layout(nz)})
            with T.Stage(0):
                T.fill(nz, T.bfloat16(0))
                T.copy(x, src)
                T.copy(src, nz[:16, :])
                T.copy(nz, other)
                out[0] = other[0, 0]

    assert ("other", "src") in _insert_sync_alias_pairs(main)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("footprint", ["base", "short", "exact", "padded"])
def test_nd2nz_post_copy_keeps_the_last_required_element_live(dtype, footprint):
    cols = 16 if dtype == "float32" else 32
    required = 24 if dtype == "float32" else 48
    extent = {"base": 1, "short": required - 1, "exact": required, "padded": 2 * cols}[footprint]
    dtype_name = "float" if dtype == "float32" else "half"

    @T.prim_func
    def main(out: T.Tensor((1,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((2 * cols,), dtype)
            b = T.alloc_shared((2 * cols,), dtype)
            dst = T.alloc_l1((cols,), dtype)
            with T.Stage(0):
                with T.Task():
                    for i in T.serial(2 * cols):
                        a[i] = T.cast(100, dtype)
                b[required - 1] = T.cast(200, dtype)
                out[0] = b[required - 1]
                with T.Task():
                    for i in T.serial(required - 1):
                        a[i] = T.cast(300, dtype)
                T.ascend_nd2nz_post_copy(
                    T.access_ptr(dst[0], "w", extent=cols),
                    T.access_ptr(a[0], "r", extent=extent),
                    1,
                    cols,
                    1,
                    dtype_name,
                )

    assert ("a", "b") not in _insert_sync_alias_pairs(main)


def test_nd2nz_scatter_checks_the_full_vector_read_footprint():
    @T.prim_func
    def main(out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((16 * 64,), "float32")
            b = T.alloc_shared((16 * 64,), "float32")
            dst = T.alloc_shared((17 * 64,), "bfloat16")
            with T.Stage(0):
                with T.Task():
                    for i in T.serial(16 * 64):
                        a[i] = T.float32(100)
                b[1] = T.float32(200)
                out[0] = b[1]
                a[0] = T.float32(300)
                T.evaluate(
                    tvm.tirx.Call(
                        "void",
                        tvm.ir.Op.get("tl.ascend_nd2nz_scatter"),
                        [
                            T.access_ptr(a[0], "r", extent=1),
                            T.access_ptr(dst[0], "w", extent=17 * 64),
                            T.int32(16),
                            T.int32(64),
                            tvm.tirx.StringImm("bfloat16_t"),
                            tvm.tirx.StringImm("float"),
                        ],
                    )
                )

    assert ("a", "b") not in _insert_sync_alias_pairs(main)


@pytest.mark.parametrize("metadata_kind", ["stride", "offset", "shape", "immutable"])
def test_mutable_layout_metadata_does_not_prove_physical_coverage(metadata_kind):
    @T.prim_func
    def main(out: T.Tensor((2,), "int32"), step: T.int32):
        with T.Kernel(1):
            a = T.alloc_shared((16,), "int32")
            b = T.alloc_shared((16,), "int32")
            metadata = T.alloc_shared((1,), "int32")
            with T.Stage(0):
                metadata[0] = 1
                if metadata_kind == "stride":
                    view = T.StridedTensor((8,), (metadata[0],), "int32", data=a.data, scope="shared.dyn")
                elif metadata_kind == "offset":
                    view = tvm.tirx.decl_buffer((8,), "int32", data=a.data, elem_offset=metadata[0], scope="shared.dyn")
                elif metadata_kind == "shape":
                    view = tvm.tirx.decl_buffer((2, metadata[0]), "int32", data=a.data, scope="shared.dyn")
                else:
                    view = T.StridedTensor((8,), (step,), "int32", data=a.data, scope="shared.dyn")
                a[1] = 100
                b[1] = 200
                out[1] = b[1]
                metadata[0] = 2
                if metadata_kind == "shape":
                    view[1, 0] = 300
                elif metadata_kind == "offset":
                    view[0] = 300
                else:
                    view[1] = 300
                metadata[0] = 1
                if metadata_kind == "shape":
                    out[0] = view[1, 0]
                elif metadata_kind == "offset":
                    out[0] = view[0]
                else:
                    out[0] = view[1]

    pairs = _insert_sync_alias_pairs(main)
    if metadata_kind == "immutable":
        assert ("b", "view") in pairs
    else:
        assert ("b", "view") not in pairs


def test_layout_metadata_buffer_stays_live_until_the_access():
    @T.prim_func
    def main(out: T.Tensor((2,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((16,), "int32")
            b = T.alloc_shared((16,), "int32")
            metadata = T.alloc_shared((1,), "int32")
            with T.Stage(0):
                metadata[0] = 1
                view = T.StridedTensor((8,), (metadata[0],), "int32", data=a.data, scope="shared.dyn")
                a[1] = 100
                a[2] = 300
                b[0] = 2
                out[1] = b[0]
                out[0] = view[1]

    assert ("b", "metadata") not in _insert_sync_alias_pairs(main)


@pytest.mark.parametrize("stride", [1, 16])
def test_padded_copy_coverage_uses_physical_rows(stride):
    @T.prim_func
    def main(A: T.Tensor((30,), "float32"), out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((512,), "float32")
            b = T.alloc_shared((512,), "float32")
            view = T.StridedTensor((32,), (stride,), "float32", data=allocation.data, scope="shared.dyn")
            with T.Stage(0):
                view[31] = 100.0
                b[31 * stride] = 200.0
                T.ascend_set_copy_pad_value(b[31 * stride], dtype="float32")
                # The GM round trip supplies a real MTE3->MTE2 completion
                # edge, so missing ordering cannot hide a bad coverage proof.
                with T.Task():
                    for i in T.serial(30):
                        b[i] = 1.0
                T.copy(b[:30], A)
                T.copy(A, view[:30], data_select=True)
                out[0] = view[0] + view[31]

    # For stride=1, padding really overwrites view[31], so the allocation can
    # be reused across its two definitions. For stride=16, each of the 30
    # one-element DMA rows pads only its own seven following physical lanes;
    # view[31] remains live across b's write at the same allocation offset.
    pairs = _insert_sync_alias_pairs(main)
    assert (("b", "view") in pairs) == (stride == 1)


def test_padded_copy_coalesced_aligned_rows_allow_reuse():
    @T.prim_func
    def main(A: T.Tensor((2, 4), "float32"), out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((2, 4), "float32")
            b = T.alloc_shared((2, 4), "float32")
            with T.Stage(0):
                a[1, 3] = 100.0
                b[1, 3] = 200.0
                T.ascend_set_copy_pad_value(b[1, 3], dtype="float32")
                # Complete b's accesses before the new a definition.
                with T.Task():
                    for i, j in T.grid(2, 4):
                        b[i, j] = 1.0
                T.copy(b, A)
                T.copy(A, a, data_select=True)
                out[0] = a[0, 0] + a[1, 3]

    # The two logical four-element rows coalesce into one aligned 32B DMA
    # row: there is no physical padding outside the allocation.
    assert ("a", "b") in _insert_sync_alias_pairs(main)


def test_padded_copy_overflow_excludes_storage_from_reuse():
    @T.prim_func
    def main(A: T.Tensor((4, 30), "float32"), C: T.Tensor((4, 30), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((4, 63), "float32")
            b = T.alloc_shared((4, 63), "float32")
            # The logical [32, 62) range fits, but padding extends it to 64.
            T.copy(A, a[:, 32:62], pad_value=0.0)
            T.copy(a[:, 32:62], C)
            T.copy(A, b[:, :30], pad_value=0.0)
            T.copy(b[:, :30], C)

    # Stop at InsertSync: this checks conservative alias rejection, not an
    # executable out-of-bounds kernel or a new synchronization diagnostic.
    assert ("a", "b") not in _insert_sync_alias_pairs(main)


@pytest.mark.parametrize(
    "make_program, may_alias",
    [
        pytest.param(lambda: _make_internal_conditional_write_program(), False, id="internal_conditional_write_is_not_a_must_definition"),
        pytest.param(lambda: _make_predicated_buffer_store_program(), False, id="predicated_buffer_store_is_not_a_must_definition"),
        pytest.param(lambda: _make_disjoint_task_writes_program(), False, id="disjoint_task_writes_do_not_form_a_must_definition"),
        pytest.param(lambda: _make_strided_task_write_program(), False, id="strided_task_write_does_not_form_a_must_definition"),
        pytest.param(lambda: _make_raw_stepped_task_write_program(), False, id="raw_stepped_task_write_does_not_form_a_must_definition"),
        pytest.param(lambda: _make_loop_break_task_write_program(), False, id="loop_break_task_write_does_not_form_a_must_definition"),
        pytest.param(
            lambda: _make_loop_break_before_writer_program(), False, id="loop_break_before_writer_keeps_incoming_value_live_through"
        ),
        pytest.param(lambda: _make_diagonal_task_write_program(), False, id="diagonal_task_write_does_not_form_a_must_definition"),
        pytest.param(
            lambda: _make_predicated_access_ptr_write_program(), False, id="predicated_access_ptr_write_does_not_form_a_must_definition"
        ),
        pytest.param(lambda: _make_maybe_empty_single_trip_program(), False, id="maybe_empty_single_trip_scope_is_not_promoted"),
        pytest.param(lambda: _make_mixed_opaque_access_program(), False, id="opaque_access_is_not_hidden_by_region_in_same_task"),
        pytest.param(lambda: _make_cross_row_access_ptr_program(), False, id="cross_row_access_ptr_does_not_underestimate_region"),
        pytest.param(
            lambda: _make_cyclic_scalar_program(interleave=True), False, id="auto_schedule_does_not_force_interleaved_lifetimes_to_alias"
        ),
    ],
)
def test_access_footprints_alias_contract(make_program, may_alias):
    assert (("a", "b") in _insert_sync_alias_pairs(make_program())) == may_alias


if __name__ == "__main__":
    tilelang.testing.main()
