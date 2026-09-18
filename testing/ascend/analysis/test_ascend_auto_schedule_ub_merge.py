"""AutoSchedule buffer-alias contract and periodic-lifetime tests."""

import re

import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ascend.transform import _ffi_api as ascend_transform_ffi
from tilelang.engine.lower import lower
from tilelang.language.utils import region


def _make_cyclic_scalar_program(interleave: bool = False, versions: int = 1):
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            if versions > 1:
                T.annotate_buffer_versions({a: versions, b: versions})
            for k in T.Pipelined(8, num_stages=1):
                a[0] = x[k]
                if interleave:
                    b[0] = y[k]
                out_x[k] = a[0]
                if not interleave:
                    b[0] = y[k]
                out_y[k] = b[0]

    return main


def _make_distance_ring_program(versions: int, carried: bool, use_dma: bool, shifted_slot: bool = False):
    @T.prim_func
    def main(
        x: T.Tensor((12,), "int32"),
        y: T.Tensor((12,), "int32"),
        previous: T.Tensor((12,), "int32"),
        out_b: T.Tensor((12,), "int32"),
        local: T.Tensor((12,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((versions, 8), "int32")
            b = T.alloc_shared((versions, 8), "int32")
            T.annotate_manual_multi_buffer(a)
            T.annotate_manual_multi_buffer(b)
            for k in T.Pipelined(12, num_stages=1):
                if carried:
                    previous[k] = T.if_then_else(k < versions, T.int32(0), a[(k + int(shifted_slot)) % versions, 0])
                    b[k % versions, 0] = y[k]
                    out_b[k] = b[k % versions, 0]
                a[k % versions, 0] = x[k]
                if use_dma:
                    T.copy(a[k % versions, :1], local[k : k + 1])
                else:
                    local[k] = a[k % versions, 0]
                if not carried:
                    previous[k] = a[k % versions, 0]
                    # Keep the completion of both a readers before b even if
                    # AutoSchedule reorders independent task bodies.
                    b[k % versions, 0] = y[k] + previous[k] + local[k]
                    out_b[k] = b[k % versions, 0]

    return main


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


def _make_reduce_accumulator_program(clear: bool):
    tile = 64
    iterations = 8

    @T.prim_func
    def main(
        x: T.Tensor((iterations, tile), "float32"),
        initial: T.Tensor((iterations,), "float32"),
        y: T.Tensor((iterations,), "float32"),
        out_accumulator: T.Tensor((iterations,), "float32"),
        out_scratch: T.Tensor((iterations,), "float32"),
    ):
        with T.Kernel(1):
            src = T.alloc_shared((1, tile), "float32")
            accumulator = T.alloc_shared((1,), "float32")
            scratch = T.alloc_shared((1,), "float32")
            for k in T.Pipelined(iterations, num_stages=1):
                T.copy(x[k, :], src[0, :])
                accumulator[0] = initial[k]
                scratch[0] = y[k]
                out_scratch[k] = scratch[0]
                with T.SimdVF():
                    T.reduce_sum(src, accumulator, dim=1, clear=clear)
                out_accumulator[k] = accumulator[0]

    return main


def _make_rw_generation_program(interleave: bool = False):
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                a[0] = x[k]
                if interleave:
                    b[0] = y[k]
                    out_y[k] = b[0]
                a[0] = a[0] + 1
                out_x[k] = a[0]
                if not interleave:
                    b[0] = y[k]
                    out_y[k] = b[0]

    return main


def _make_nested_scope_program(inner_extent: int):
    @T.prim_func
    def main(
        x: T.Tensor((16,), "int32"),
        y: T.Tensor((16,), "int32"),
        out_x: T.Tensor((16,), "int32"),
        out_y: T.Tensor((16,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(4, num_stages=1):
                for i in T.serial(inner_extent):
                    index = k * inner_extent + i
                    a[0] = x[index]
                    out_x[index] = a[0]
                for i in T.serial(inner_extent):
                    index = k * inner_extent + i
                    b[0] = y[index]
                    out_y[index] = b[0]

    return main


def _make_nested_interleaved_program(inner_extent: int):
    @T.prim_func
    def main(
        x: T.Tensor((16,), "int32"),
        y: T.Tensor((16,), "int32"),
        out_x: T.Tensor((16,), "int32"),
        out_y: T.Tensor((16,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(4, num_stages=1):
                for i in T.serial(inner_extent):
                    index = k * inner_extent + i
                    a[0] = x[index]
                    b[0] = y[index]
                    out_x[index] = a[0]
                    out_y[index] = b[0]

    return main


def _make_symbolic_nested_scope_program():
    @T.prim_func
    def main(
        x: T.Tensor((16,), "int32"),
        y: T.Tensor((16,), "int32"),
        out_x: T.Tensor((16,), "int32"),
        out_y: T.Tensor((16,), "int32"),
        inner_extent: T.int32,
    ):
        with T.Kernel(1):
            T.assume(inner_extent >= 1)
            T.assume(inner_extent <= 4)
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(4, num_stages=1):
                for i in T.serial(inner_extent):
                    index = k * inner_extent + i
                    a[0] = x[index]
                    out_x[index] = a[0]
                for i in T.serial(inner_extent):
                    index = k * inner_extent + i
                    b[0] = y[index]
                    out_y[index] = b[0]

    return main


def _make_strided_nested_scope_program():
    @T.prim_func
    def main(
        x: T.Tensor((16,), "int32"),
        y: T.Tensor((16,), "int32"),
        out_x: T.Tensor((16,), "int32"),
        out_y: T.Tensor((16,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(4, num_stages=1):
                for i in T.serial(0, 4, 2):
                    index = k * 4 + i
                    a[0] = x[index]
                    out_x[index] = a[0]
                for i in T.serial(0, 4, 2):
                    index = k * 4 + i
                    b[0] = y[index]
                    out_y[index] = b[0]

    return main


def _make_same_guard_conditional_program():
    tile = 64
    outer = 4

    @T.prim_func
    def main(
        x: T.Tensor((outer * tile,), "int32"),
        y: T.Tensor((outer * tile,), "int32"),
        out_x: T.Tensor((outer * tile,), "int32"),
        out_y: T.Tensor((outer * tile,), "int32"),
        cond: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((tile,), "int32")
            b = T.alloc_shared((tile,), "int32")
            for k in T.Pipelined(outer, num_stages=1):
                if cond > 0:
                    offset = k * tile
                    T.copy(x[offset : offset + tile], a)
                    T.copy(a, out_x[offset : offset + tile])
                    T.copy(y[offset : offset + tile], b)
                    T.copy(b, out_y[offset : offset + tile])

    return main


def _make_conditional_writer_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
        cond: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                if cond > 0:
                    a[0] = x[k]
                out_x[k] = a[0]
                b[0] = y[k]
                out_y[k] = b[0]

    return main


def _make_guard_implication_program(reverse: bool = False):
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
        cond: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                if reverse:
                    if cond > 1:
                        a[0] = x[k]
                    if cond > 0:
                        out_x[k] = a[0]
                else:
                    if cond > 0:
                        a[0] = x[k]
                    if cond > 1:
                        out_x[k] = a[0]
                b[0] = y[k]
                out_y[k] = b[0]

    return main


def _make_nested_guard_implication_program(reverse: bool = False):
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
        cond: T.int32,
        extra: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                if reverse:
                    if cond > 0:  # noqa: SIM102
                        if extra > 0:
                            a[0] = x[k]
                    if cond > 0:
                        out_x[k] = a[0]
                else:
                    if cond > 0:
                        a[0] = x[k]
                    if cond > 0:  # noqa: SIM102
                        if extra > 0:
                            out_x[k] = a[0]
                b[0] = y[k]
                out_y[k] = b[0]

    return main


def _make_mutually_exclusive_program(loop_varying: bool = False):
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
        cond: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                if loop_varying:
                    if k % 2 == 0:
                        a[0] = x[k]
                    else:
                        b[0] = y[k]
                    if k % 2 == 0:
                        out_x[k] = a[0]
                    else:
                        out_y[k] = b[0]
                else:
                    if cond > 0:
                        a[0] = x[k]
                    else:
                        b[0] = y[k]
                    if cond > 0:
                        out_x[k] = a[0]
                    else:
                        out_y[k] = b[0]

    return main


def _make_overlapping_nonexclusive_guard_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
        cond: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                if cond > 0:
                    a[0] = x[k]
                if cond > 1:
                    b[0] = y[k]
                if cond > 0:
                    out_x[k] = a[0]
                if cond > 1:
                    out_y[k] = b[0]

    return main


def _make_nested_conditional_scope_program():
    inner = 2
    outer = 4

    @T.prim_func
    def main(
        x: T.Tensor((outer * inner,), "int32"),
        y: T.Tensor((outer * inner,), "int32"),
        out_x: T.Tensor((outer * inner,), "int32"),
        out_y: T.Tensor((outer * inner,), "int32"),
        cond: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(outer, num_stages=1):
                if cond > 0:
                    for i in T.serial(inner):
                        index = k * inner + i
                        a[0] = x[index]
                        out_x[index] = a[0]
                    for i in T.serial(inner):
                        index = k * inner + i
                        b[0] = y[index]
                        out_y[index] = b[0]

    return main


def _make_nested_live_through_program():
    inner = 2
    outer = 4

    @T.prim_func
    def main(
        x: T.Tensor((outer * inner,), "int32"),
        y: T.Tensor((outer * inner,), "int32"),
        out_x: T.Tensor((outer * inner,), "int32"),
        out_y: T.Tensor((outer * inner,), "int32"),
        cond: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(outer, num_stages=1):
                for i in T.serial(inner):
                    index = k * inner + i
                    if cond > 0:
                        a[0] = x[index]
                    out_x[index] = a[0]
                for i in T.serial(inner):
                    index = k * inner + i
                    b[0] = y[index]
                    out_y[index] = b[0]

    return main


def _make_loop_carried_gap_program(interleave: bool = False):
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                out_x[k] = a[0]
                b[0] = y[k]
                if interleave:
                    a[0] = x[k]
                out_y[k] = b[0]
                if not interleave:
                    a[0] = x[k]

    return main


def _make_loop_carried_local_reader_program(use_dma: bool):
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        previous: T.Tensor((8,), "int32"),
        b_result: T.Tensor((8,), "int32"),
        local_result: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                previous[k] = T.if_then_else(k == 0, T.int32(0), a[0])
                b[0] = y[k]
                b_result[k] = b[0]
                a[0] = x[k]
                if use_dma:
                    T.copy(a[:1], local_result[k : k + 1])
                else:
                    local_result[k] = a[0]

    return main


def _make_guarded_loop_carried_gap_program(loop_varying: bool = False):
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
        cond: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                if loop_varying:
                    if k % 2 == 0:
                        out_x[k] = a[0]
                else:
                    if cond > 0:
                        out_x[k] = a[0]
                b[0] = y[k]
                out_y[k] = b[0]
                if loop_varying:
                    if k % 2 == 0:
                        a[0] = x[k]
                else:
                    if cond > 0:
                        a[0] = x[k]

    return main


def _make_buffer_guarded_loop_carried_gap_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
        predicate: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                if predicate[0] > 0:
                    out_x[k] = a[0]
                b[0] = y[k]
                out_y[k] = b[0]
                if predicate[0] > 0:
                    a[0] = x[k]

    return main


def _make_nested_outer_guard_carried_program():
    tile = 64
    inner = 2
    outer = 4

    @T.prim_func
    def main(
        x: T.Tensor((outer * inner * tile,), "int32"),
        y: T.Tensor((outer * inner * tile,), "int32"),
        out_x: T.Tensor((outer * inner * tile,), "int32"),
        out_y: T.Tensor((outer * inner * tile,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((tile,), "int32")
            b = T.alloc_shared((tile,), "int32")
            for k in T.Pipelined(outer, num_stages=1):
                for i in T.serial(inner):
                    offset = (k * inner + i) * tile
                    if k % 2 == 0:  # noqa: SIM102
                        if k > 0:
                            T.copy(a, out_x[offset : offset + tile])
                    if k % 2 == 1:
                        T.copy(y[offset : offset + tile], b)
                    if k % 2 == 1:
                        T.copy(b, out_y[offset : offset + tile])
                    if k % 2 == 0:
                        T.copy(x[offset : offset + tile], a)

    return main


def _make_two_loop_carried_program():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(8, num_stages=1):
                out_x[k] = a[0]
                out_y[k] = b[0]
                a[0] = x[k]
                b[0] = y[k]

    return main


def _make_root_live_in_program(overlap: bool = False):
    @T.prim_func
    def main(
        x: T.Tensor((1,), "int32"),
        y: T.Tensor((1,), "int32"),
        out_x: T.Tensor((1,), "int32"),
        out_y: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            if overlap:
                b[0] = y[0]
            out_x[0] = a[0]
            if not overlap:
                b[0] = y[0]
            out_y[0] = b[0]
            a[0] = x[0]

    return main


def _make_nested_live_out_program(overlap: bool = False, conditional_writer: bool = False):
    tile = 64
    inner = 2
    outer = 4

    @T.prim_func
    def main(
        x: T.Tensor((outer * inner * tile,), "int32"),
        y: T.Tensor((outer * inner * tile,), "int32"),
        out_x: T.Tensor((outer * tile,), "int32"),
        out_y: T.Tensor((outer * inner * tile,), "int32"),
        cond: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((tile,), "int32")
            b = T.alloc_shared((tile,), "int32")
            for k in T.Pipelined(outer, num_stages=1):
                for i in T.serial(inner):
                    offset = (k * inner + i) * tile
                    T.copy(y[offset : offset + tile], b)
                    if overlap:
                        T.copy(x[offset : offset + tile], a)
                    T.copy(b, out_y[offset : offset + tile])
                    if not overlap:
                        if conditional_writer:
                            if cond > 0:
                                T.copy(x[offset : offset + tile], a)
                        else:
                            T.copy(x[offset : offset + tile], a)
                T.copy(a, out_x[k * tile : (k + 1) * tile])

    return main


def _make_parent_live_out_program(overlap: bool = False):
    tile = 64
    inner = 2
    outer = 4

    @T.prim_func
    def main(
        x: T.Tensor((outer * inner * tile,), "int32"),
        y: T.Tensor((outer * tile,), "int32"),
        out_x: T.Tensor((outer * tile,), "int32"),
        out_y: T.Tensor((outer * tile,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((tile,), "int32")
            b = T.alloc_shared((tile,), "int32")
            for k in T.Pipelined(outer, num_stages=1):
                T.copy(y[k * tile : (k + 1) * tile], b)
                if not overlap:
                    T.copy(b, out_y[k * tile : (k + 1) * tile])
                for i in T.serial(inner):
                    offset = (k * inner + i) * tile
                    T.copy(x[offset : offset + tile], a)
                if overlap:
                    T.copy(b, out_y[k * tile : (k + 1) * tile])
                T.copy(a, out_x[k * tile : (k + 1) * tile])

    return main


def _make_parent_live_in_program(overlap: bool = False):
    tile = 64
    inner = 2
    outer = 4

    @T.prim_func
    def main(
        x: T.Tensor((outer * tile,), "int32"),
        y: T.Tensor((outer * tile,), "int32"),
        out_x: T.Tensor((outer * inner * tile,), "int32"),
        out_y: T.Tensor((outer * tile,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((tile,), "int32")
            b = T.alloc_shared((tile,), "int32")
            for k in T.Pipelined(outer, num_stages=1):
                T.copy(x[k * tile : (k + 1) * tile], a)
                if overlap:
                    T.copy(y[k * tile : (k + 1) * tile], b)
                    T.copy(b, out_y[k * tile : (k + 1) * tile])
                for i in T.serial(inner):
                    offset = (k * inner + i) * tile
                    T.copy(a, out_x[offset : offset + tile])
                if not overlap:
                    T.copy(y[k * tile : (k + 1) * tile], b)
                    T.copy(b, out_y[k * tile : (k + 1) * tile])

    return main


def _make_preloaded_read_only_program():
    tile = 64
    iterations = 4

    @T.prim_func
    def main(
        x: T.Tensor((iterations, tile), "int32"),
        weight: T.Tensor((tile,), "int32"),
        out: T.Tensor((iterations, tile), "int32"),
    ):
        with T.Kernel(1):
            weight_ub = T.alloc_shared((tile,), "int32")
            x_ub = T.alloc_shared((tile,), "int32")
            mid_ub = T.alloc_shared((tile,), "int32")
            out_ub = T.alloc_shared((tile,), "int32")
            T.copy(weight, weight_ub)
            for k in T.Pipelined(iterations, num_stages=1):
                T.copy(x[k, :], x_ub)
                with T.SimtVF(threads=64):
                    for i in T.Parallel(tile):
                        mid_ub[i] = x_ub[i] + weight_ub[i]
                with T.SimtVF(threads=64):
                    for i in T.Parallel(tile):
                        out_ub[i] = mid_ub[i] + 1
                T.copy(out_ub, out[k, :])

    return main


def _make_parent_with_child_loop_carried_program():
    tile = 64
    inner = 2
    outer = 4

    @T.prim_func
    def main(
        x: T.Tensor((outer * inner * tile,), "int32"),
        y: T.Tensor((outer * tile,), "int32"),
        out_x: T.Tensor((outer * inner * tile,), "int32"),
        out_y: T.Tensor((outer * tile,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((tile,), "int32")
            b = T.alloc_shared((tile,), "int32")
            for k in T.Pipelined(outer, num_stages=1):
                for i in T.serial(inner):
                    offset = (k * inner + i) * tile
                    T.copy(a, out_x[offset : offset + tile])
                    T.copy(x[offset : offset + tile], a)
                T.copy(y[k * tile : (k + 1) * tile], b)
                T.copy(b, out_y[k * tile : (k + 1) * tile])

    return main


def _make_deep_nested_parent_live_out_program(overlap: bool = False):
    tile = 64
    middle = 2
    inner = 2
    outer = 4

    @T.prim_func
    def main(
        x: T.Tensor((outer * middle * inner * tile,), "int32"),
        y: T.Tensor((outer * tile,), "int32"),
        out_x: T.Tensor((outer * tile,), "int32"),
        out_y: T.Tensor((outer * tile,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((tile,), "int32")
            b = T.alloc_shared((tile,), "int32")
            for k in T.Pipelined(outer, num_stages=1):
                T.copy(y[k * tile : (k + 1) * tile], b)
                if not overlap:
                    T.copy(b, out_y[k * tile : (k + 1) * tile])
                for j in T.serial(middle):
                    for i in T.serial(inner):
                        offset = ((k * middle + j) * inner + i) * tile
                        T.copy(x[offset : offset + tile], a)
                if overlap:
                    T.copy(b, out_y[k * tile : (k + 1) * tile])
                T.copy(a, out_x[k * tile : (k + 1) * tile])

    return main


def _make_nontransitive_alias_graph_program():
    @T.prim_func
    def main(
        x: T.Tensor((1,), "int32"),
        y: T.Tensor((1,), "int32"),
        z: T.Tensor((1,), "int32"),
        out_x: T.Tensor((1,), "int32"),
        out_y: T.Tensor((1,), "int32"),
        out_z: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((64,), "int32")
            b = T.alloc_shared((32,), "int32")
            c = T.alloc_shared((32,), "int32")
            a[0] = x[0]
            out_x[0] = a[0]
            b[0] = y[0]
            c[0] = z[0]
            out_y[0] = b[0]
            out_z[0] = c[0]

    return main


def _collect_op_names(node) -> set[str]:
    names = set()

    def visit(value):
        if isinstance(value, tvm.tirx.Call) and isinstance(value.op, tvm.ir.Op):
            names.add(str(value.op.name))

    tvm.tirx.stmt_functor.post_order_visit(node, visit)
    return names


def _insert_sync_alias_pairs(program, *, disable_reuse=False, analysis_only=False) -> set[tuple[str, str]]:
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureInsertSync:
        def run_after_pass(self, mod, info):
            if info.name == "tl.InsertSync":
                snapshots[info.name] = mod
                if analysis_only:
                    raise ValueError("test stopped after capturing InsertSync")

    # Direct lower() calls require the caller to hold the target scope.
    target = tvm.target.Target("ascend")
    with (
        target,
        tvm.transform.PassContext(
            opt_level=3,
            instruments=[CaptureInsertSync()],
            config={"tl.disable_shared_memory_reuse": disable_reuse},
        ),
    ):
        try:
            lower(program, target=target)
        except ValueError as error:
            if not analysis_only or "test stopped after capturing InsertSync" not in str(error):
                raise

    pairs = set()

    def collect(node):
        annotations = getattr(node, "annotations", None)
        if not annotations or "tl.buffer_alias_map" not in annotations:
            return
        aliases = annotations["tl.buffer_alias_map"]
        for storage, compatible in aliases.items():
            for other in compatible:
                pairs.add(tuple(sorted((storage.name, other.name))))

    tvm.tirx.stmt_functor.post_order_visit(snapshots["tl.InsertSync"]["main"].body, collect)
    return pairs


def _merged_int32_offsets(source: str) -> set[int]:
    return {int(value) for value in re.findall(r"int32_t\*\)buf_dyn_shmem\)\[(\d+)\]", source)}


def _ub_store_offset(source: str, input_name: str) -> int:
    match = re.search(
        rf"int32_t\*\)buf_dyn_shmem\)\[(\d+)\] = {input_name}\[0\]",
        source,
    )
    assert match is not None
    return int(match.group(1))


def test_internal_conditional_write_is_not_a_must_definition():
    pairs = _insert_sync_alias_pairs(_make_internal_conditional_write_program())
    assert ("a", "b") not in pairs


def test_predicated_buffer_store_is_not_a_must_definition():
    pairs = _insert_sync_alias_pairs(_make_predicated_buffer_store_program())
    assert ("a", "b") not in pairs


def test_disjoint_task_writes_do_not_form_a_must_definition():
    pairs = _insert_sync_alias_pairs(_make_disjoint_task_writes_program())
    assert ("a", "b") not in pairs


def test_strided_task_write_does_not_form_a_must_definition():
    pairs = _insert_sync_alias_pairs(_make_strided_task_write_program())
    assert ("a", "b") not in pairs


def test_raw_stepped_task_write_does_not_form_a_must_definition():
    pairs = _insert_sync_alias_pairs(_make_raw_stepped_task_write_program())
    assert ("a", "b") not in pairs


def test_loop_break_task_write_does_not_form_a_must_definition():
    pairs = _insert_sync_alias_pairs(_make_loop_break_task_write_program())
    assert ("a", "b") not in pairs


def test_loop_break_before_writer_keeps_incoming_value_live_through():
    pairs = _insert_sync_alias_pairs(_make_loop_break_before_writer_program())
    assert ("a", "b") not in pairs


def test_diagonal_task_write_does_not_form_a_must_definition():
    pairs = _insert_sync_alias_pairs(_make_diagonal_task_write_program())
    assert ("a", "b") not in pairs


def test_predicated_access_ptr_write_does_not_form_a_must_definition():
    pairs = _insert_sync_alias_pairs(_make_predicated_access_ptr_write_program())
    assert ("a", "b") not in pairs


def test_maybe_empty_single_trip_scope_is_not_promoted():
    pairs = _insert_sync_alias_pairs(_make_maybe_empty_single_trip_program())
    assert ("a", "b") not in pairs


def test_opaque_access_is_not_hidden_by_region_in_same_task():
    pairs = _insert_sync_alias_pairs(_make_mixed_opaque_access_program())
    assert ("a", "b") not in pairs


def test_address_of_escape_keeps_unknown_read_footprint_live():
    # read_second dereferences p[1], outside the base element named by
    # address_of. Rewriting a[0] cannot kill that still-live value of a[1].
    program = _make_address_of_escape_program()
    assert ("a", "b") not in _insert_sync_alias_pairs(program)
    source = lower(program, target="ascend").kernel_source
    assert _ub_store_offset(source, "x") != _ub_store_offset(source, "y")


@pytest.mark.parametrize("index_kind", ["extern", "load", "wrapped"])
def test_unknown_write_index_does_not_kill_other_elements(index_kind):
    # With an index value of zero, the second store leaves a[1] live. Its
    # may-write hull must not become a definition of every element in a.
    program = _make_unknown_write_index_program(index_kind)
    assert ("a", "b") not in _insert_sync_alias_pairs(program)
    source = lower(program, target="ascend").kernel_source
    assert _ub_store_offset(source, "x") != _ub_store_offset(source, "y")


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
    assert ("a", "b") not in _insert_sync_alias_pairs(main, analysis_only=True)


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


def test_cross_row_access_ptr_does_not_underestimate_region():
    pairs = _insert_sync_alias_pairs(_make_cross_row_access_ptr_program())
    assert ("a", "b") not in pairs


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

    pairs = _insert_sync_alias_pairs(main, analysis_only=True)
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

    assert ("a", "b") not in _insert_sync_alias_pairs(main, analysis_only=True)


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

    assert ("a", "b") not in _insert_sync_alias_pairs(main, analysis_only=True)


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

    assert ("dst", "src") in _insert_sync_alias_pairs(main, analysis_only=True)


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

    assert ("other", "src") in _insert_sync_alias_pairs(main, analysis_only=True)


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

    assert ("a", "b") not in _insert_sync_alias_pairs(main, analysis_only=True)


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

    assert ("a", "b") not in _insert_sync_alias_pairs(main, analysis_only=True)


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
    assert ("a", "b") not in _insert_sync_alias_pairs(main, analysis_only=True)


def test_insert_sync_ffi_accepts_zero_arguments():
    assert ascend_transform_ffi.InsertSync() is not None


@pytest.mark.parametrize("padded", [False, True])
def test_same_pipe_issue_order_does_not_allow_dma_alias(padded):
    width = 7 if padded else 8

    @T.prim_func
    def main(x: T.Tensor((8192,), "int32"), y: T.Tensor((width,), "int32"), out: T.Tensor((width,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((8192,), "int32")
            b = T.alloc_shared((8192,), "int32")
            # Preserve MTE2 issue order without adding a completion dependency.
            with T.Stage(0):
                T.copy(x, a)
            if padded:
                T.copy(y, b[8184 : 8184 + width], pad_value=0)
            else:
                T.copy(y, b[8184 : 8184 + width])
            T.copy(b[8184 : 8184 + width], out)

    assert ("a", "b") not in _insert_sync_alias_pairs(main)
    source = lower(main, target="ascend").kernel_source
    # If the bases alias, the second DMA writes the still-in-flight first
    # DMA's tail. An MTE2 -> MTE3 wait after both writes cannot repair that WAW.
    assert _merged_int32_offsets(source) == {0, 8192 + 8184}


def test_dma_completion_through_another_pipe_still_allows_alias():
    @T.prim_func
    def main(x: T.Tensor((8,), "int32"), intermediate: T.Tensor((8,), "int32"), out: T.Tensor((8,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            T.copy(x, a)
            T.copy(a, intermediate)
            T.copy(intermediate, b)
            T.copy(b, out)

    # The MTE3 -> MTE2 dependency completes a's reader before b is written.
    assert ("a", "b") in _insert_sync_alias_pairs(main)
    assert _merged_int32_offsets(lower(main, target="ascend").kernel_source) == {0}


def test_auto_schedule_does_not_force_interleaved_lifetimes_to_alias():
    pairs = _insert_sync_alias_pairs(_make_cyclic_scalar_program(interleave=True))
    assert ("a", "b") not in pairs


def test_read_modify_write_reduce_keeps_incoming_accumulator_live():
    pairs = _insert_sync_alias_pairs(_make_reduce_accumulator_program(clear=False))
    assert ("accumulator", "scratch") not in pairs


def test_interleaved_lifetimes_do_not_alias_in_auto_schedule():
    artifact = lower(_make_cyclic_scalar_program(interleave=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_rw_consumer_keeps_old_generation_live():
    artifact = lower(_make_rw_generation_program(interleave=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_single_trip_nested_scopes_promote_lifetimes():
    artifact = lower(_make_nested_scope_program(inner_extent=1), target="ascend")
    assert _merged_int32_offsets(artifact.kernel_source) == {0}


def test_single_trip_writer_and_multi_trip_sibling_reader_keep_overlap():
    @T.prim_func
    def main(x: T.Tensor((4,), "int32"), y: T.Tensor((4,), "int32"), out: T.Tensor((4, 3), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(4, num_stages=1):
                b[0] = y[k]
                for _i in T.serial(1):
                    a[0] = x[k] + b[0]
                for j in T.serial(2):
                    out[k, j] = a[0]
                out[k, 2] = b[0]

    assert ("a", "b") not in _insert_sync_alias_pairs(main)


def test_nested_carried_lifetime_preserves_local_dma_reader():
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        previous: T.Tensor((8,), "int32"),
        local: T.Tensor((8,), "int32"),
        out_b: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            for k in T.Pipelined(4, num_stages=1):
                for j in T.serial(2):
                    index = k * 2 + j
                    previous[index] = T.if_then_else(index == 0, T.int32(0), a[0])
                    b[0] = y[index]
                    out_b[index] = b[0]
                    a[0] = x[index]
                    T.copy(a[:1], local[index : index + 1])

    assert ("a", "b") not in _insert_sync_alias_pairs(main)


def test_disabling_reuse_keeps_contract_and_sequential_allocation():
    @T.prim_func
    def main(x: T.Tensor((2,), "int32"), out: T.Tensor((2,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            with T.Stage(0):
                a[0] = x[0]
                out[0] = a[0]
                b[0] = x[1]
                out[1] = b[0]

    assert ("a", "b") in _insert_sync_alias_pairs(main)
    assert not _insert_sync_alias_pairs(main, disable_reuse=True)
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": True}):
        source = lower(main, target="ascend").kernel_source
    assert _merged_int32_offsets(source) == {0, 8}


def test_multi_trip_nested_scopes_use_hierarchical_summary():
    for inner_extent in (2, 3):
        artifact = lower(_make_nested_scope_program(inner_extent), target="ascend")
        assert _merged_int32_offsets(artifact.kernel_source) == {0}


def test_same_multi_trip_child_keeps_child_conflict_result():
    artifact = lower(_make_nested_interleaved_program(inner_extent=2), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_symbolic_nested_extent_uses_hierarchical_transfer():
    artifact = lower(_make_symbolic_nested_scope_program(), target="ascend")
    assert _merged_int32_offsets(artifact.kernel_source) == {0}


def test_strided_nested_extent_uses_logical_trip_count():
    artifact = lower(_make_strided_nested_scope_program(), target="ascend")
    assert _merged_int32_offsets(artifact.kernel_source) == {0}


def test_conditional_writer_with_unconditional_read_is_live_through():
    artifact = lower(_make_conditional_writer_program(), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_reverse_guard_implication_remains_live_through():
    artifact = lower(_make_guard_implication_program(reverse=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_nested_writer_guard_does_not_imply_outer_reader_guard():
    artifact = lower(_make_nested_guard_implication_program(reverse=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_iteration_invariant_mutually_exclusive_phases_alias():
    artifact = lower(_make_mutually_exclusive_program(), target="ascend")
    assert _merged_int32_offsets(artifact.kernel_source) == {0}


def test_loop_varying_mutually_exclusive_phases_keep_distinct_epoch_storage():
    artifact = lower(_make_mutually_exclusive_program(loop_varying=True), target="ascend")
    # The alternating branches use different active-epoch clocks. A distance-1
    # ordering in either domain cannot be projected into the other one.
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_overlapping_nonexclusive_guard_phases_do_not_alias():
    artifact = lower(_make_overlapping_nonexclusive_guard_program(), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_conditioned_nested_scopes_use_hierarchical_summary():
    artifact = lower(_make_nested_conditional_scope_program(), target="ascend")
    assert _merged_int32_offsets(artifact.kernel_source) == {0}


def test_nested_live_through_propagates_to_parent_scope():
    artifact = lower(_make_nested_live_through_program(), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_preloaded_read_only_storage_stays_live_through_loop():
    pairs = _insert_sync_alias_pairs(_make_preloaded_read_only_program())
    assert ("out_ub", "weight_ub") not in pairs


def test_loop_carried_live_in_out_can_alias_through_cyclic_gap():
    artifact = lower(_make_loop_carried_gap_program(), target="ascend")
    assert _merged_int32_offsets(artifact.kernel_source) == {0}


def test_local_phase_overlapping_loop_carried_suffix_does_not_alias():
    artifact = lower(_make_loop_carried_gap_program(interleave=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


@pytest.mark.parametrize("use_dma", [False, True])
def test_loop_carried_lifetime_preserves_local_reader(use_dma):
    program = _make_loop_carried_local_reader_program(use_dma)
    pairs = _insert_sync_alias_pairs(program)
    # The scalar reader finishes before the next write to b. The MTE3 reader
    # has no such ordering, even though the next iteration's scalar read of a
    # finishes before b is written. Both reader lifetimes must be checked.
    assert (("a", "b") in pairs) == (not use_dma)
    source = lower(program, target="ascend").kernel_source
    offsets = _merged_int32_offsets(source)
    if use_dma:
        assert len(offsets) >= 2
    else:
        assert offsets == {0}


@pytest.mark.parametrize("versions", [2, 3, 5])
@pytest.mark.parametrize("use_dma", [False, True])
@pytest.mark.parametrize("shifted_slot", [False, True])
def test_reader_of_earlier_ring_generation_keeps_allocation_live(versions, use_dma, shifted_slot):
    # The carried reader sees the write versions (or versions - 1) iterations
    # ago. Even scalar completion leaves other physical slots live while b
    # executes, so whole-allocation reuse is not legal.
    program = _make_distance_ring_program(versions, True, use_dma, shifted_slot)
    assert ("a", "b") not in _insert_sync_alias_pairs(program)


@pytest.mark.parametrize("versions", [2, 3, 5])
def test_local_ring_generations_can_share_whole_allocation(versions):
    # Slot periodicity must not turn into a blanket ban on multi-buffer reuse.
    program = _make_distance_ring_program(versions, False, False)
    assert ("a", "b") in _insert_sync_alias_pairs(program)


def test_two_loop_carried_lifetimes_overlap_at_iteration_boundary():
    artifact = lower(_make_two_loop_carried_program(), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_loop_varying_guard_does_not_form_distance_one_lifetime():
    artifact = lower(_make_guarded_loop_carried_gap_program(loop_varying=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_buffer_guard_does_not_form_distance_one_lifetime():
    pairs = _insert_sync_alias_pairs(_make_buffer_guarded_loop_carried_gap_program())
    assert ("a", "b") not in pairs


def test_outer_loop_guard_does_not_form_inner_distance_one_lifetime():
    artifact = lower(_make_nested_outer_guard_carried_program(), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_live_in_prefix_can_release_storage_after_last_incoming_read():
    artifact = lower(_make_root_live_in_program(), target="ascend")
    assert _merged_int32_offsets(artifact.kernel_source) == {0}


def test_phase_overlapping_live_in_prefix_does_not_alias():
    artifact = lower(_make_root_live_in_program(overlap=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_phase_overlapping_live_out_suffix_does_not_alias():
    artifact = lower(_make_nested_live_out_program(overlap=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_conditional_writer_with_outside_read_is_live_through():
    artifact = lower(_make_nested_live_out_program(conditional_writer=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_parent_phase_overlapping_child_live_out_does_not_alias():
    artifact = lower(_make_parent_live_out_program(overlap=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_parent_phase_overlapping_child_live_in_does_not_alias():
    artifact = lower(_make_parent_live_in_program(overlap=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_child_loop_carried_phase_promotes_across_parent_boundary():
    artifact = lower(_make_parent_with_child_loop_carried_program(), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_deep_nested_live_out_rejects_overlapping_parent_phase():
    artifact = lower(_make_deep_nested_parent_live_out_program(overlap=True), target="ascend")
    assert len(_merged_int32_offsets(artifact.kernel_source)) >= 2


def test_auto_schedule_packer_uses_whole_allocation_cliques():
    source = lower(_make_nontransitive_alias_graph_program(), target="ascend").kernel_source
    a_offset = _ub_store_offset(source, "x")
    b_offset = _ub_store_offset(source, "y")
    c_offset = _ub_store_offset(source, "z")
    assert a_offset == 0
    assert {b_offset, c_offset} == {0, 64}


def test_inserted_sync_stays_outside_native_conditions():
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureInsertSync:
        def run_after_pass(self, mod, info):
            if info.name == "tl.InsertSync":
                snapshots[info.name] = mod

    with tvm.transform.PassContext(opt_level=3, instruments=[CaptureInsertSync()]):
        lower(_make_same_guard_conditional_program(), target="ascend")

    body = snapshots["tl.InsertSync"]["main"].body
    sync_ops = {
        "tl.ascend_pipe_barrier",
        "tl.ascend_set_flag",
        "tl.ascend_wait_flag",
    }
    assert _collect_op_names(body) & sync_ops

    conditioned_tasks = 0

    def check_condition(node):
        nonlocal conditioned_tasks
        if not isinstance(node, tvm.tirx.IfThenElse):
            return
        body_ops = _collect_op_names(node.then_case)
        if "tl.tileop.ascend_copy" not in body_ops:
            return
        conditioned_tasks += 1
        assert not body_ops & sync_ops

    tvm.tirx.stmt_functor.post_order_visit(body, check_condition)
    assert conditioned_tasks == 4


def test_alias_contract_lifetime():
    snapshots = {}
    wanted = {
        "tl.InsertSync",
        "tl.LowerScheduledTIR",
        "tl.MergeUBAllocations",
    }

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod.script()

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        tilelang.lower(_make_cyclic_scalar_program(), target="ascend")

    assert "tl.buffer_alias_map" in snapshots["tl.InsertSync"]
    assert "tl.buffer_alias_map" in snapshots["tl.LowerScheduledTIR"]
    assert "tl.buffer_alias_map" not in snapshots["tl.MergeUBAllocations"]


if __name__ == "__main__":
    tilelang.testing.main()
