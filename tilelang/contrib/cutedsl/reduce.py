"""
Reduce operations for CuTeDSL backend.
Based on tl_templates/cuda/reduce.h
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Float32
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import nvvm
from cutlass.cute.arch.nvvm_wrappers import shuffle_sync_op


@dsl_user_op
def min(a: float | Float32, b: float | Float32, c: float | Float32 | None = None, *, loc=None, ip=None) -> Float32:
    return Float32(
        nvvm.fmin(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max(a: float | Float32, b: float | Float32, c: float | Float32 | None = None, *, loc=None, ip=None) -> Float32:
    return Float32(
        nvvm.fmax(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


class SumOp:
    """Sum reduction operator"""

    @staticmethod
    def __call__(x, y):
        return x + y


class MaxOp:
    """Max reduction operator"""

    @staticmethod
    def __call__(x, y):
        return max(x, y)


class MinOp:
    """Min reduction operator"""

    @staticmethod
    def __call__(x, y):
        # Use cutlass.min which is JIT-friendly
        return min(x, y)


class BitAndOp:
    """Bitwise AND reduction operator"""

    @staticmethod
    def __call__(x, y):
        return x & y


class BitOrOp:
    """Bitwise OR reduction operator"""

    @staticmethod
    def __call__(x, y):
        return x | y


class BitXorOp:
    """Bitwise XOR reduction operator"""

    @staticmethod
    def __call__(x, y):
        return x ^ y


def bar_sync(barrier_id, number_of_threads):
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=number_of_threads)


def bar_sync_ptx(barrier_id, number_of_threads):
    from cutlass._mlir.dialects import llvm

    llvm.inline_asm(
        None,
        [Int32(barrier_id).ir_value(), Int32(number_of_threads).ir_value()],
        "bar.sync $0, $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def AllReduce(reducer, threads, scale, thread_offset, all_threads=None):
    """
    AllReduce operation implementing warp/block-level reduction.
    Based on tl::AllReduce from reduce.h

    Args:
        reducer: Reducer operator class (SumOp, MaxOp, etc.)
        threads: Number of threads participating in reduction
        scale: Reduction scale factor
        thread_offset: Thread ID offset
        all_threads: Total number of threads in block

    Returns:
        A callable object with run() and run_hopper() methods
    """

    class AllReduceInstance:
        def __init__(self, reducer, threads, scale, thread_offset: cutlass.Constexpr[int], all_threads: cutlass.Constexpr[int]):
            self.reducer = reducer
            self.threads = threads
            self.scale = scale
            self.thread_offset = thread_offset
            self.all_threads = all_threads if all_threads is not None else threads

        def run(self, x, red_buf: cute.Pointer = None):
            """
            Perform all-reduce across threads.
            Based on tl::AllReduce<...>::run from reduce.h
            """
            offset = self.threads // 2

            if offset >= 32:
                # Use shared memory for large thread counts
                cute.arch.sync_threads()
                tidx, _, _ = cute.arch.thread_idx()
                cute.make_tensor(red_buf + tidx - self.thread_offset, (1,))[0] = x
                cute.arch.sync_threads()
                x = self.reducer()(x, cute.make_tensor(red_buf + ((tidx - self.thread_offset) ^ offset), (1,))[0])
            else:
                # Use warp shuffle for small thread counts
                # Use the pre-existing shuffle_sync_op with butterfly (XOR) mode
                other = shuffle_sync_op(x, offset, mask=0xFFFFFFFF, mask_and_clamp=0x1F, kind=nvvm.ShflKind.bfly)
                x = self.reducer()(x, other)

            return (
                x
                if offset == self.scale
                else AllReduce(self.reducer, offset, self.scale, self.thread_offset, self.all_threads).run(x, red_buf)
            )

        def run_hopper(self, x, red_buf: cute.Pointer = None):
            """
            Perform all-reduce on Hopper architecture using bar.sync.
            Based on tl::AllReduce<...>::run_hopper from reduce.h
            """
            offset = self.threads // 2
            tidx, _, _ = cute.arch.thread_idx()
            if offset >= 32:
                # Use inlined asm for bar.sync to avoid instruction reordering
                bar_sync_ptx(1, self.all_threads)
                cute.make_tensor(red_buf + tidx - self.thread_offset, (1,))[0] = x
                bar_sync_ptx(2, self.all_threads)
                x = self.reducer()(x, cute.make_tensor(red_buf + ((tidx - self.thread_offset) ^ offset), (1,))[0])
            else:
                # Use warp shuffle for small thread counts
                # Use the pre-existing shuffle_sync_op with butterfly (XOR) mode
                other = shuffle_sync_op(x, offset, mask=0xFFFFFFFF, mask_and_clamp=0x1F, kind=nvvm.ShflKind.bfly)
                x = self.reducer()(x, other)

            return (
                x
                if offset == self.scale
                else AllReduce(self.reducer, offset, self.scale, self.thread_offset, self.all_threads).run_hopper(x, red_buf)
            )

    return AllReduceInstance(reducer, threads, scale, thread_offset, all_threads)
