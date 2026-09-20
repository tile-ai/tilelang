"""Buffer lifetimes contracts."""

import pytest
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from testing.ascend.auto_schedule._reuse_utils import _insert_sync_alias_pairs


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
                    # Include a completion dependency when one reader uses DMA;
                    # scalar source order alone does not finish that transfer.
                    b[k % versions, 0] = y[k] + previous[k] + local[k]
                    out_b[k] = b[k % versions, 0]

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


def test_read_modify_write_reduce_keeps_incoming_accumulator_live():
    pairs = _insert_sync_alias_pairs(_make_reduce_accumulator_program(clear=False))
    assert ("accumulator", "scratch") not in pairs


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


def test_multi_trip_nested_scopes_use_hierarchical_summary():
    for inner_extent in (2, 3):
        assert ("a", "b") in _insert_sync_alias_pairs(_make_nested_scope_program(inner_extent))


def test_loop_varying_mutually_exclusive_phases_keep_distinct_epoch_storage():
    pairs = _insert_sync_alias_pairs(_make_mutually_exclusive_program(loop_varying=True))
    # The alternating branches use different active-epoch clocks. A distance-1
    # ordering in either domain cannot be projected into the other one.
    assert ("a", "b") not in pairs


def test_preloaded_read_only_storage_stays_live_through_loop():
    pairs = _insert_sync_alias_pairs(_make_preloaded_read_only_program())
    assert ("out_ub", "weight_ub") not in pairs


@pytest.mark.parametrize("use_dma", [False, True])
def test_loop_carried_lifetime_preserves_local_reader(use_dma):
    program = _make_loop_carried_local_reader_program(use_dma)
    pairs = _insert_sync_alias_pairs(program)
    # The scalar reader finishes before the next write to b. The MTE3 reader
    # has no such ordering, even though the next iteration's scalar read of a
    # finishes before b is written. Both reader lifetimes must be checked.
    assert (("a", "b") in pairs) == (not use_dma)


@pytest.mark.parametrize("versions", [2, 3, 5])
@pytest.mark.parametrize("use_dma", [False, True])
@pytest.mark.parametrize("shifted_slot", [False, True])
def test_ring_generation_lifetimes(versions, use_dma, shifted_slot):
    # With two slots and a one-slot shift, the scalar read ends the only
    # remaining live generation before b writes. More slots or asynchronous
    # readers leave a generation live across that write.
    program = _make_distance_ring_program(versions, True, use_dma, shifted_slot)
    may_alias = versions == 2 and shifted_slot and not use_dma
    assert (("a", "b") in _insert_sync_alias_pairs(program)) == may_alias


@pytest.mark.parametrize("versions", [2, 3, 5])
def test_local_ring_generations_can_share_whole_allocation(versions):
    # Slot periodicity must not turn into a blanket ban on multi-buffer reuse.
    program = _make_distance_ring_program(versions, False, False)
    assert ("a", "b") in _insert_sync_alias_pairs(program)


@pytest.mark.parametrize(
    "make_program, may_alias",
    [
        pytest.param(lambda: _make_rw_generation_program(interleave=True), False, id="rw_consumer_keeps_old_generation_live"),
        pytest.param(lambda: _make_nested_scope_program(inner_extent=1), True, id="single_trip_nested_scopes_promote_lifetimes"),
        pytest.param(
            lambda: _make_nested_interleaved_program(inner_extent=2), False, id="same_multi_trip_child_keeps_child_conflict_result"
        ),
        pytest.param(lambda: _make_symbolic_nested_scope_program(), True, id="symbolic_nested_extent_uses_hierarchical_transfer"),
        pytest.param(lambda: _make_strided_nested_scope_program(), True, id="strided_nested_extent_uses_logical_trip_count"),
        pytest.param(lambda: _make_conditional_writer_program(), False, id="conditional_writer_with_unconditional_read_is_live_through"),
        pytest.param(lambda: _make_guard_implication_program(reverse=True), False, id="reverse_guard_implication_remains_live_through"),
        pytest.param(
            lambda: _make_nested_guard_implication_program(reverse=True), False, id="nested_writer_guard_does_not_imply_outer_reader_guard"
        ),
        pytest.param(lambda: _make_mutually_exclusive_program(), True, id="iteration_invariant_mutually_exclusive_phases_alias"),
        pytest.param(
            lambda: _make_overlapping_nonexclusive_guard_program(), False, id="overlapping_nonexclusive_guard_phases_do_not_alias"
        ),
        pytest.param(lambda: _make_nested_conditional_scope_program(), True, id="conditioned_nested_scopes_use_hierarchical_summary"),
        pytest.param(lambda: _make_nested_live_through_program(), False, id="nested_live_through_propagates_to_parent_scope"),
        pytest.param(lambda: _make_loop_carried_gap_program(), True, id="loop_carried_live_in_out_can_alias_through_cyclic_gap"),
        pytest.param(
            lambda: _make_loop_carried_gap_program(interleave=True), False, id="local_phase_overlapping_loop_carried_suffix_does_not_alias"
        ),
        pytest.param(lambda: _make_two_loop_carried_program(), False, id="two_loop_carried_lifetimes_overlap_at_iteration_boundary"),
        pytest.param(
            lambda: _make_guarded_loop_carried_gap_program(loop_varying=True),
            False,
            id="loop_varying_guard_does_not_form_distance_one_lifetime",
        ),
        pytest.param(lambda: _make_buffer_guarded_loop_carried_gap_program(), False, id="buffer_guard_does_not_form_distance_one_lifetime"),
        pytest.param(
            lambda: _make_nested_outer_guard_carried_program(), False, id="outer_loop_guard_does_not_form_inner_distance_one_lifetime"
        ),
        pytest.param(lambda: _make_root_live_in_program(), True, id="live_in_prefix_can_release_storage_after_last_incoming_read"),
        pytest.param(lambda: _make_root_live_in_program(overlap=True), False, id="phase_overlapping_live_in_prefix_does_not_alias"),
        pytest.param(lambda: _make_nested_live_out_program(overlap=True), False, id="phase_overlapping_live_out_suffix_does_not_alias"),
        pytest.param(
            lambda: _make_nested_live_out_program(conditional_writer=True), False, id="conditional_writer_with_outside_read_is_live_through"
        ),
        pytest.param(
            lambda: _make_parent_live_out_program(overlap=True), False, id="parent_phase_overlapping_child_live_out_does_not_alias"
        ),
        pytest.param(lambda: _make_parent_live_in_program(overlap=True), False, id="parent_phase_overlapping_child_live_in_does_not_alias"),
        pytest.param(
            lambda: _make_parent_with_child_loop_carried_program(), False, id="child_loop_carried_phase_promotes_across_parent_boundary"
        ),
        pytest.param(
            lambda: _make_deep_nested_parent_live_out_program(overlap=True),
            False,
            id="deep_nested_live_out_rejects_overlapping_parent_phase",
        ),
    ],
)
def test_buffer_lifetimes_alias_contract(make_program, may_alias):
    assert (("a", "b") in _insert_sync_alias_pairs(make_program())) == may_alias


if __name__ == "__main__":
    tilelang.testing.main()
